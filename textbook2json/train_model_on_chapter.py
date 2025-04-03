#!/usr/bin/env python3
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import logging
from copy import deepcopy
import re
import io
import time
import json
import base64
import requests
import fitz
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from json_repair import repair_json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rewrite import Config as RewriteConfig, run_rewrite

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- Utility Functions and Dataset Loaders ---------

def load_dataset_from_file(filename: str):
    """
    Load a dataset from a JSON file.
    The JSON file should contain records with at least a "prompt" key,
    and for SFT, a "response" key.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file {filename} not found.")
    try:
        dataset = load_dataset("json", data_files={"train": filename})["train"]
        # Check for required keys
        if "prompt" not in dataset.column_names:
            raise ValueError(f"Dataset {filename} must contain a 'prompt' key.")
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {filename}: {str(e)}")

def load_sft_dataset_step1():
    return load_dataset_from_file("sft_dataset_step1.json")

def load_rl_dataset_step2():
    return load_dataset_from_file("rl_dataset_step2.json")

def load_sft_dataset_step3():
    return load_dataset_from_file("sft_dataset_step3.json")

def load_rl_dataset_step4():
    return load_dataset_from_file("rl_dataset_step4.json")

def compute_reward(generated_texts, batch):
    """
    Compute rewards for generated responses.
    This is a placeholder function that rewards longer responses.
    In practice, implement your own evaluation (e.g., using a symbolic checker).
    """
    rewards = []
    for text in generated_texts:
        # Example: reward based on the number of words in the response
        word_count = len(text.split())
        reward = min(word_count / 10.0, 1.0)  # Normalize to [0,1]
        rewards.append(reward)
    return rewards

# --------- Step 1: SFT Training Function ---------

def update_progress(progress_file: str, progress: dict):
    with open(progress_file, "w") as f:
        json.dump(progress, f)


def sft_training(model, tokenizer, dataset, output_dir, num_epochs=3, progress_file="progress.json"):
    # Check for 'response' key in SFT dataset

    if "response" not in dataset.column_names:
        raise ValueError("SFT dataset must contain a 'response' key.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    total_steps = len(dataset) // training_args.per_device_train_batch_size * num_epochs

    logger.info(f"Starting SFT training for {output_dir} ...")
    step = 0
    for epoch in range(num_epochs):
        for batch in trainer.get_train_dataloader():
            trainer.training_step(model, batch)
            step += 1
            progress = {"epoch": epoch+1, "step": step, "total_steps": total_steps,
                        "percentage": round(100 * step / total_steps, 2)}
            update_progress(progress_file, progress)
            if step % 50 == 0:
                logger.info(f"Progress: {progress['percentage']}%")
    logger.info(f"SFT training complete for {output_dir}. Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# --------- Step 2 & 4: RL Training Function ---------

def rl_training(model, tokenizer, dataset, output_dir, num_epochs=3):
    """
    Reinforcement Learning (RL) training using PPO (via the TRL library).
    This function performs a simplified RL loop over the dataset.
    """
    try:
        from trl import PPOConfig, PPOTrainer
    except ImportError:
        raise ImportError("Please install the TRL library: pip install trl")

    # Configure PPO; tune hyperparameters as needed.
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        log_with="tensorboard",
    )

    # Create a frozen reference model
    ref_model = deepcopy(model).eval()

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        **ppo_config.__dict__,
    )

    logger.info(f"Starting RL training for {output_dir} ...")
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataset):
            prompt = batch["prompt"]
            # Tokenize the prompt and move to model device
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Generate responses; adjust max_length as needed
            generated_ids = model.generate(inputs.input_ids, max_length=256)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Compute rewards for each generated response
            rewards = compute_reward(generated_texts, batch)
            # Convert rewards to tensor on the same device
            rewards_tensor = torch.tensor(rewards, device=model.device)
            # Perform a PPO optimization step
            stats = ppo_trainer.step([inputs.input_ids], [generated_ids], [rewards_tensor])
            if i % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {i} -- PPO stats: {stats}")

    logger.info(f"RL training complete for {output_dir}. Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# --------- Train Model on Chapter Function ---------

def train_model_on_chapter(model_name: str, file: str):
    """
    Fine-tunes a model on chapter training data from a JSONL file using the same SFT process as in main.
    
    Parameters:
      - model_name: The folder name (and model identifier) where the trained model is saved.
      - file: The path to a JSONL file containing training data. Each line must contain an "instruction" key,
              and (optionally) an "output" key. If "output" is missing, it is set to an empty string.
              
    This function is tuned for a tutoring service. It assumes the training examples are in tutoring format
    (e.g., questions, explanations, and instructions extracted from a textbook).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    # Check if a model already exists in the given folder.

    if os.path.isdir(model_name) and os.listdir(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logging.info(f"Loaded existing chapter model from {model_name}")
    else:
        os.makedirs(model_name, exist_ok=True)
        base_model_name = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        logging.info(f"Initialized new model from {base_model_name} for chapter training.")

    # Load dataset from JSONL file.
    dataset = load_dataset("json", data_files={"train": file})["train"]

    # Ensure each record has an "instruction" key and an "output" key.
    def ensure_keys(example):
        if "instruction" not in example:
            raise ValueError("Each record must have an 'instruction' key.")
        if "output" not in example:
            example["output"] = ""
        return example

    dataset = dataset.map(ensure_keys)

    # For tutoring service training, increase the number of epochs for better tuning.
    tutoring_epochs = 5

    # Fine-tune the model using the same SFT training process as in main.
    sft_training(model, tokenizer, dataset, output_dir=model_name, num_epochs=tutoring_epochs)


