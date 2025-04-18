from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import torch
from torch.autograd import Variable

from torch_geometric.loader import DataLoader
from torch_geometric import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from network import Net
from tripletnet import tripletnet
from triplet_sampler import TripletSampler

from dataset import GraphDataset

import pandas as pd


# Test & Evaluation code
def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

def load_model(path, num_features, num_classes, args, device):
    model = Net(
        num_features=num_features,
        nhid=args.nhid,
        num_classes=args.final_dim,
        pooling_ratio=args.pooling_ratio,
        dropout_ratio=args.dropout_ratio
    ).to(device)
    # def __init__(self, num_features, nhid, num_classes, pooling_ratio, dropout_ratio):
    # model = Net(num_features, args.nhid, args.final_dim, args.pooling_ratio, args.dropout_ratio).to(device)

    # Load the state dict
    state_dict = torch.load(path, map_location=device, weights_only=False)
    # total_params = 0
    # for param_tensor in state_dict.values():
    #     total_params += param_tensor.numel()

    # print(f"Total number of parameters: {total_params}")

    # def convert_bytes(size):
    #     for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
    #         if size < 1024.0:
    #             return f"{size:.2f} {unit}"
    #         size /= 1024.0
    # file_size = os.path.getsize(model_path)
    # print(f"File size: {convert_bytes(file_size)}")


    
    # Remove 'map2_model' keys from the state dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('map2_model')}

    # Load the state dict, ignoring missing and unexpected keys
    model.load_state_dict(state_dict, strict=False)
    
    # Recreate the map2_model
    in_feat = args.final_dim
    pred_layers = []
    pred_layers.append(nn.Linear(in_feat, 64).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(64, 32).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(32, 2).to(device))
    model.map2_model = nn.Sequential(*pred_layers)
    
    model.eval()
    return model

# def load_model(path, num_features, num_classes, args, device):
#     model = Net(
#         num_features=num_features,
#         nhid=args.nhid,
#         num_classes=num_classes,
#         pooling_ratio=args.pooling_ratio,
#         dropout_ratio=args.dropout_ratio
#     ).to(device)
    
#     # Load the state dict
#     state_dict = torch.load(path, map_location=device, weights_only=False)
    
#     # Remove 'map2_model' keys from the state dict
#     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('map2_model')}

#     # Load the state dict, ignoring missing and unexpected keys
#     model.load_state_dict(state_dict, strict=False)
    
#     # Dynamically recreate the map2_model
#     in_feat = args.final_dim
#     hidden_layers = args.map2_hidden_layers if hasattr(args, 'map2_hidden_layers') else [64, 32]
    
#     pred_layers = []
#     prev_dim = in_feat
#     for hidden_dim in hidden_layers:
#         pred_layers.append(nn.Linear(prev_dim, hidden_dim).to(device))
#         pred_layers.append(nn.LeakyReLU())
#         prev_dim = hidden_dim
    
#     # Add final layer
#     pred_layers.append(nn.Linear(prev_dim, num_classes).to(device))
    
#     model.map2_model = nn.Sequential(*pred_layers)
    
#     model.eval()
#     return model


def evaluate(val_loader, model_path, num_features, num_classes, args, device):
    # Load the model
    model = load_model(model_path, num_features, num_classes, args, device)
    model.eval()

    val_embeddings = []
    val_labels = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            learned_feat = out.cpu().numpy()
            val_embeddings.append(learned_feat)
            val_labels.append(data.y.long().cpu().numpy())

    val_embeddings = np.vstack(val_embeddings)
    val_labels = np.concatenate(val_labels)

    # Use KNN for prediction
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(val_embeddings, val_labels)
    val_preds = neigh.predict(val_embeddings)

    result = {
        'prec': metrics.precision_score(val_labels, val_preds, average='macro'),
        'recall': metrics.recall_score(val_labels, val_preds, average='macro'),
        'acc': metrics.accuracy_score(val_labels, val_preds),
        'F1': metrics.f1_score(val_labels, val_preds, average="micro")
    }

    return result


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=32, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD')
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')
parser.add_argument('--num_features', type=int, default=3, help='Dimension of input features')
parser.add_argument('--final_dim', type=int, default=32, help='Dimension of final embeddings')
parser.add_argument('--alpha', type=float, default=1.5, help='Margin in the triplet loss')

args = parser.parse_args()

print("here")
# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise RuntimeError("CUDA is not available. Please ensure GPU and CUDA are set up.")

# Load dataset

# dataset_dir = os.path.join('data', args.dataset)
dataset_dir = '/home/emu/Documents/surrogate/dataset/pickle/data-set/cs8test/'
dataset = GraphDataset(dataset_dir)

num_classes = 2  # Since feasibility is binary (True/False)
num_features = dataset.num_features

# Train
min_loss = 1e10
patience = 0

val_accs = []
test_accs = []

val_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '6th_m2n_model_latest.pth'
results = evaluate(val_loader, model_path, num_features, num_classes, args, device)
print(results)



