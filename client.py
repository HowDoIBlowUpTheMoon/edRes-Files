from collections import OrderedDict
import warnings

import flwr as fl
import torch
import numpy as np

import random
from torch.utils.data import DataLoader

from datasets import load_dataset
from evaluate import load as load_metric

import argparse
import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PeftModel
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda")
BATCH_SIZE = 32
MODEL_NAME_OR_PATH = "bert-base-uncased"
LOCAL_EPOCHS = 2
TASK = "sst2"
PEFT_ID = 0

def load_data(PEFT_ID: int) -> [DataLoader, DataLoader]: # Can't figure out the return type for evaluate.load

    task = TASK

    model_name_or_path = MODEL_NAME_OR_PATH
    batch_size = BATCH_SIZE
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", use_fast=False)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    trainString = f'train[{PEFT_ID * 10}%:{(PEFT_ID + 1) * 10}%]' 
    valString = f'validation[{PEFT_ID * 10}%:{(PEFT_ID + 1) * 10}%]' 
    
    trainset = load_dataset("glue", task, split = trainString)
    testset = load_dataset("glue", task, split = valString)
 
    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
        return outputs
    
    tokenized_train = trainset.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
    )

    tokenized_test = testset.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
    )
    
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(tokenized_train, shuffle=True, collate_fn=collate_fn, batch_size=batch_size) #, generator=g)
    eval_dataloader = DataLoader(
        tokenized_test, shuffle=False, collate_fn=collate_fn, batch_size=batch_size) #, generator=g
    
    return train_dataloader, eval_dataloader


def train(net, trainloader, epochs):
    lr = 0.3
    optimizer = AdamW(params=net.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(trainloader) * epochs),
        num_training_steps=(len(trainloader) * epochs),
    )
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


def test(net, testloader):
    task = TASK
    metric = evaluate.load("glue", task)
    loss = 0.0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

def client_fn(cid):
    trainloader, testloader = load_data(cid)
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
    net = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, return_dict=True)
    net = get_peft_model(net, peft_config).to(DEVICE)

    return PeftClient(net, trainloader, testloader, LOCAL_EPOCHS)


class PeftClient(fl.client.NumPyClient):
        def __init__(
            self,
            model,
            trainloader: DataLoader,
            testloader: DataLoader,
            localEpochs: int, 
        ) -> None:
            self.model = model
            self.trainloader = trainloader
            self.testloader = testloader
            self.num_local_epochs = localEpochs
            
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in get_peft_model_state_dict(self.model).items()]

        def set_parameters(self, parameters):
            keys = [k for k in get_peft_model_state_dict(self.model).keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            set_peft_model_state_dict(self.model, state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(self.model, self.trainloader, self.num_local_epochs)
            print("Training Finished.")
            return self.get_parameters(config={}), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(self.model, self.testloader)
            return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
                                                                         
def main(): 
    PEFT_ID = int(sys.argv[1])
    
    trainloader, testloader = load_data(PEFT_ID)
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
    net = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH, return_dict=True)
    net = get_peft_model(net, peft_config).to(DEVICE)

    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PeftClient(net, trainloader, testloader, LOCAL_EPOCHS))


if __name__ == "__main__":
    main()
