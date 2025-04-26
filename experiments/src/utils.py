import torch
import torch.nn as nn
import numpy as np
import yaml
import os
from datetime import datetime


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"seed {seed} set")
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.10f}'.format})
    torch.set_float32_matmul_precision("high")


def apply_glorot_xavier(model):
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


def inspect_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_count = parameter.numel()
        total_params += param_count
        #print(f"{name}: {param_count} parameters")
    #print(f"Total number of parameters: {total_params}")
    return total_params


def get_torch_device(device=None):

    if device:
        torch.device(device)
        print(device)
        return device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


class Logger:
    def __init__(self, log_name):
        os.makedirs(f"logs/{log_name}", exist_ok=True)
        log_filename = f"logs/{log_name}/logs_{log_name}.log"
        self.log_file = open(log_filename, "a")
        self.log_name = log_name
        
        self.log_file.write(f"{self.log_name}\n")
        self.log_file.write(f"logging started: {datetime.now()}\n")
        self.log_file.write("-" * 80 + "\n")
        self.log_file.flush() 
        
    def log_message(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {msg}\n")
        self.log_file.flush() 
        
    def __del__(self):
        self.log_file.write(f"logging ended: {datetime.now()}\n")
        self.log_file.write("-" * 80 + "\n")
        self.log_file.flush() 
        self.log_file.close()