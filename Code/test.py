import os
import glob
import random
import shutil

import torch
from torch.autograd import Variable
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric import utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from network import Net


from dataset_m2n_test import GraphDataset

num_dataset_files = 200
case_num = 4

# Use os.path.join for path construction, and add a wildcard to match files
file_pattern = os.path.join('/home/emu/Documents/surrogate/dataset/pickle/m2n/cs0-test', '*')

# This will now match all files in the cs8test directory
files = glob.glob(file_pattern)

# Ensure we're only dealing with files, not directories
files = [f for f in files if os.path.isfile(f)]

target_files = random.sample(files, min(len(files), num_dataset_files))
target_dataset_dir = '/tmp/cs' + str(case_num)

if os.path.exists(target_dataset_dir):
    shutil.rmtree(target_dataset_dir)
os.makedirs(target_dataset_dir)

for f in target_files:
    shutil.copy2(f, os.path.join(target_dataset_dir, os.path.basename(f)))

# dataset = GraphDataset(target_dataset_dir)
dataset = GraphDataset(target_dataset_dir)

# Set the device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# if device.type != 'cuda':
#     raise RuntimeError("CUDA is not available. Please ensure GPU and CUDA are set up.")
device = 'cpu'

num_classes = 2  # Since feasibility is binary (True/False)
num_features = dataset.num_features
print(num_features)

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)

training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

model = Net(num_features=num_features,
            nhid=128,
            num_classes=32,
            pooling_ratio=0.5,
            dropout_ratio=0.5).to(device)

# Train & Validation Loader
train_loader = DataLoader(training_set, batch_size=1, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=1, shuffle=True)

# Initialization of the final classifier
in_feat = 32
pred_layers = []
pred_layers.append(nn.Linear(in_feat, 64).to(device))
pred_layers.append(nn.LeakyReLU())
pred_layers.append(nn.Linear(64, 32).to(device))
pred_layers.append(nn.LeakyReLU())
pred_layers.append(nn.Linear(32, 2).to(device))
pred_model = nn.Sequential(*pred_layers)

# The to-be-finetuned model and the optimizer
model.map2_model = pred_model

# model.load_state_dict(torch.load('/home/emu/Documents/surrogate/Code/best_machine.pth', weights_only=True))
model.load_state_dict(torch.load('/home/emu/Documents/surrogate/Code/cs4.pth', weights_only=True))
model.eval()
correct = 0.
for data in test_loader:
    data = data.to(device)
    out = model(data)
    pred = out.max(dim=1)[1]
    correct += pred.eq(data.y).sum().item()
accuracy = correct / len(test_loader.dataset)
print("Accuracy [%]: " + str(accuracy * 100.))
