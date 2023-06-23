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
from torchmetrics import MeanMetric
from tqdm import tqdm, trange


def main():
    # hyperparameters and config
    config = Config(
        project_name = "llm-peft",
        model_name = "tiiuae/falcon-7b",
        n_effective_training_iterations = 200,
        effective_batch_size = 2,
        learning_rate = 1e-4,
        gradient_accumulation_steps = 1,
        max_sequence_length = 512,
        log_every_n_updates = 10,
        validate_every_n_updates = 50,
    )

    # data preparation
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenize(data: Example | Batch) -> BatchEncoding:
        return tokenizer(
            data["text"],
            return_tensors="pt",
            max_length=config.max_sequence_length,
            padding="max_length",
            truncation=True,
        )
    
    def add_labels(data: Example | Batch) -> BatchEncoding:
        CROSS_ENTROPY_IGNORE_LABEL = -100
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        data["labels"] = torch.where(data["input_ids"] == pad_token_id, CROSS_ENTROPY_IGNORE_LABEL, data["input_ids"])
        return data

    training_dataset = (
        load_dataset("imdb", split="train[:1000]")
        .with_format("torch")
        .map(tokenize, batched=True)
        .map(add_labels, batched=True)
    )
    validation_dataset = (
        load_dataset("imdb", split="test[:100]")
        .with_format("torch")
        .map(tokenize, batched=True)
        .map(add_labels, batched=True)
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
        r = 16,
        lora_alpha = 16,
        target_modules = ["query", "value"],
        lora_dropout = 0.1,
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
    accelerator.init_trackers(config.project_name, asdict(config))

    for step in trange(config.n_training_iterations, desc="Training", leave=True):
        # training
        try:
            batch = next(iter_training_dataloader)
        except:
            iter_training_dataloader = iter(training_dataloader)
            batch = next(iter_training_dataloader)
        output = model.forward(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            labels = batch["labels"],
        )
        training_loss_metric.update(output.loss)
        accelerator.backward(output.loss / config.gradient_accumulation_steps)
        if (step + 1) % config.gradient_accumulation_steps == 0:
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
                        labels = batch["labels"],
                    )
                    validation_loss_metric.update(output.loss)
            accelerator.log({"validation_loss": validation_loss_metric.compute()}, step)
            validation_loss_metric.reset()

    accelerator.end_training()


@dataclass
class Config:
    project_name: str
    model_name: str
    n_effective_training_iterations: int
    n_training_iterations: int = field(init=False)
    effective_batch_size: int
    batch_size: int = field(init=False)
    learning_rate: float
    gradient_accumulation_steps: int
    max_sequence_length: int
    log_every_n_updates: int
    log_every_n_steps: int = field(init=False)
    validate_every_n_updates: int
    validate_every_n_steps: int = field(init=False)

    def __post_init__(self):
        if self.effective_batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError("effective_batch_size must be evenly divisble by gradient_accumulation_steps")
        
        self.batch_size = self.effective_batch_size // self.gradient_accumulation_steps
        self.n_training_iterations = self.n_effective_training_iterations * self.gradient_accumulation_steps
        self.log_every_n_steps = self.log_every_n_updates * self.gradient_accumulation_steps
        self.validate_every_n_steps = self.validate_every_n_updates * self.gradient_accumulation_steps
    

if __name__ == "__main__":
    main()
