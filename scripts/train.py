# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
# from datasets import load_dataset

# def fine_tune(output_dir):
#     dataset = load_dataset("csv", data_files={"train": "data/train.csv", "validation": "data/validation.csv"})
#     model_name = "t5-small"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     def preprocess_function(examples):
#         inputs = ["summarize: " + doc for doc in examples["article"]]
#         model_inputs = tokenizer(inputs, max_length=512, truncation=True)
#         labels = tokenizer(examples["summary"], max_length=150, truncation=True)
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
    
#     print(dataset["train"].column_names)

#     tokenized_datasets = dataset.map(preprocess_function, batched=True)

#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         logging_dir="./logs",
#         logging_steps=10,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_datasets["train"],
#         eval_dataset=tokenized_datasets["validation"],
#     )

#     trainer.train()

#     model.save_pretrained(f"{output_dir}/fine_tuned_model")
#     tokenizer.save_pretrained(f"{output_dir}/fine_tuned_model")

# if __name__ == "__main__":
#     fine_tune("models")

import os
from inspect import signature

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

def preprocess_function(examples, tokenizer):
    max_input_length = 64  # Reduced input length
    max_target_length = 64
    inputs = ["summarize: " + doc for doc in examples["article"]]
    targets = examples["highlights"]
    
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=max_target_length, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune(output_dir):
    # Load a smaller subset of the dataset
    train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    validation_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokenized_datasets = {
        "train": train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True),
        "validation": validation_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    }

    # Load smaller model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    # Training arguments
    training_args_kwargs = {
        "output_dir": os.path.join(output_dir, "results"),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 2,  # Smaller batch size
        "per_device_eval_batch_size": 2,
        "num_train_epochs": 1,  # Fewer epochs
        "weight_decay": 0.01,
        "save_steps": 100,
        "save_total_limit": 1,
        "logging_dir": os.path.join(output_dir, "logs"),
        "logging_steps": 10,
        "gradient_accumulation_steps": 16,  # Accumulating gradients
        "fp16": torch.cuda.is_available(),  # Mixed precision
        "dataloader_num_workers": 0,  # Compatibility with Windows
        "report_to": "none",
    }

    training_args_params = signature(TrainingArguments).parameters
    if "evaluation_strategy" in training_args_params:
        training_args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_args_params:
        training_args_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    fine_tune("models")
