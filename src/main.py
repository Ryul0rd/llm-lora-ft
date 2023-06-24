from dataclasses import dataclass, field, asdict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
from datasets import load_dataset
from datasets.arrow_dataset import Example, Batch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from torchmetrics import MeanMetric, Accuracy
from tqdm import tqdm, trange


def main():
    # hyperparameters and config
    config = Config(
        project_name = "llm-peft",
        model_name = "tiiuae/falcon-7b",
        n_training_iterations = 3000,
        effective_batch_size = 2,
        gradient_accumulation_steps = 1,
        learning_rate = 2e-4,
        lora_rank = 16,
        lora_alpha = 1.0,
        lora_dropout = 0.1,
        max_sequence_length = 512,
        log_every_n_steps = 10,
        validate_every_n_steps = 200,
    )

    # data preparation
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    negative_token_id = tokenizer.encode("Negative")[0]
    positive_token_id = tokenizer.encode("Positive")[0]
    label_indices = torch.tensor([negative_token_id, positive_token_id], dtype=torch.int32)

    def preprocess(data: Example | Batch) -> BatchEncoding:
        labeled_reviews = []
        for review, label in zip(data["text"], iter(data["label"])):
            label_text = "Positive" if label.item() else "Negative"
            labeled_review = f"Review:\n{review}Sentiment:\n{label_text}"
            labeled_reviews.append(labeled_review)
        data["labeled_reviews"] = labeled_reviews
        tokens = tokenizer(
            data["labeled_reviews"],
            return_tensors="pt",
            max_length=config.max_sequence_length,
            padding="max_length",
            truncation=True,
        )
        data["input_ids"] = tokens["input_ids"]
        data["attention_mask"] = tokens["attention_mask"]
        return data

    training_dataset = (
        load_dataset("imdb", split="train")
        .with_format("torch")
        .map(preprocess, batched=True)
        .shuffle()
        .select(range(8000))
    )
    validation_dataset = (
        load_dataset("imdb", split="test")
        .with_format("torch")
        .map(preprocess, batched=True)
        .shuffle()
        .select(range(1000))
    )

    training_dataloader = DataLoader(training_dataset, config.batch_size, pin_memory=True, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, config.batch_size, pin_memory=True)
    iter_training_dataloader = iter(training_dataloader)

    # model and optimizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
    )
    lora_config = LoraConfig(
        r = config.lora_rank,
        lora_alpha = config.lora_alpha,
        target_modules = ["query", "value"],
        lora_dropout = config.lora_dropout,
        bias = "none",
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype = torch.bfloat16,
        trust_remote_code = True,
        quantization_config = bnb_config,
        device_map = "auto",
    )
    #model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    optimizer = AdamW(model.parameters(), config.learning_rate)

    accelerator = Accelerator(
        mixed_precision = "bf16",
        log_with = "aim",
        project_dir = "../logs",
    )
    device = accelerator.device
    model, optimizer, training_dataloader, validation_dataloader = accelerator.prepare(
        model,
        optimizer,
        training_dataloader,
        validation_dataloader,
    )

    # training loop
    training_loss_metric = MeanMetric().to(device)
    validation_loss_metric = MeanMetric().to(device)
    validation_accuracy = Accuracy("binary").to(device)
    accelerator.init_trackers(config.project_name, asdict(config))

    for step in trange(config.n_training_iterations, desc="Training", leave=True):
        # training
        for _ in range(config.gradient_accumulation_steps):
            try:
                batch = next(iter_training_dataloader)
            except:
                iter_training_dataloader = iter(training_dataloader)
                batch = next(iter_training_dataloader)
            output = model.forward(
                input_ids = batch["input_ids"],
                attention_mask = batch["attention_mask"],
            )
            loss = torch.nn.functional.cross_entropy(output.logits[:, -1, label_indices], batch["label"])
            training_loss_metric.update(loss)
            accelerator.backward(loss / config.gradient_accumulation_steps)
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % config.log_every_n_steps == 0:
            accelerator.log({"training_loss": training_loss_metric.compute()}, step)
            training_loss_metric.reset()

        # validation
        if (step + 1) % config.validate_every_n_steps == 0:
            with torch.no_grad():
                for batch in tqdm(validation_dataloader, desc="Validating", leave=False):
                    output = model.forward(
                        input_ids = batch["input_ids"],
                        attention_mask = batch["attention_mask"],
                    )
                    loss = torch.nn.functional.cross_entropy(output.logits[:, -1, label_indices], batch["label"])
                    preds = torch.argmax(output.logits[:, -1, label_indices], dim=1)
                    validation_loss_metric.update(loss)
                    validation_accuracy.update(preds, batch["label"])
            accelerator.log(
                {
                    "validation_loss": validation_loss_metric.compute(),
                    "validation_accuracy": validation_accuracy.compute(),
                },
                step,
            )
            validation_loss_metric.reset()
            validation_accuracy.reset()

    accelerator.end_training()


@dataclass
class Config:
    project_name: str
    model_name: str
    n_training_iterations: int
    effective_batch_size: int
    batch_size: int = field(init=False)
    learning_rate: float
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    gradient_accumulation_steps: int
    max_sequence_length: int
    log_every_n_steps: int
    validate_every_n_steps: int

    def __post_init__(self):
        if self.effective_batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError("effective_batch_size must be evenly divisble by gradient_accumulation_steps")
        
        self.batch_size = self.effective_batch_size // self.gradient_accumulation_steps
    

if __name__ == "__main__":
    main()
