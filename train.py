import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import json 
from tqdm import dqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AdamW,
    get_scheduler
)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def train(seed, data_file):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForMaskedLM.from_pretrained(config.model_name).to(device)

    # Load and tokenize your dataset (here using HuggingFace datasets for demo)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # Save model
        if config.save_path:
            os.makedirs(config.save_path, exist_ok=True)
            model.save_pretrained(config.save_path)
            tokenizer.save_pretrained(config.save_path)