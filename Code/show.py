from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import torch
from torch.autograd import Variable
from torch.utils.data import random_split

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



def load_model(path, num_features, num_classes, args, device):
    model = Net(
        num_features=num_features,
        nhid=args.nhid,
        num_classes=args.final_dim,
        pooling_ratio=args.pooling_ratio,
        dropout_ratio=args.dropout_ratio
    ).to(device)
    
    # Load the state dict
    state_dict = torch.load(path, map_location=device, weights_only=False)
    
    # Remove 'map2_model' keys from the state dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('map2_model')}

    # Remove 'lin3' keys from the state dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('lin3')}

    # Load the state dict, ignoring missing and unexpected keys
    model.load_state_dict(state_dict, strict=False)
    
    # Reinitialize lin3 layer
    model.lin3 = nn.Linear(model.nhid // 2, num_classes).to(device)
    
    # Recreate the map2_model
    in_feat = args.final_dim
    pred_layers = []
    pred_layers.append(nn.Linear(in_feat, 64).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(64, 32).to(device))
    pred_layers.append(nn.LeakyReLU())
    pred_layers.append(nn.Linear(32, num_classes).to(device))
    model.map2_model = nn.Sequential(*pred_layers)
    
    model.eval()
    return model

def test_1(val_loader, model_path, num_features, num_classes, args, device):
    model = load_model(model_path, num_features, num_classes, args, device)
    model.eval()
    correct = 0
    loss = 0

    # Lists to store embeddings and labels for potential further use
    val_embeddings = []
    val_labels = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            
            # Collect embeddings and labels
            learned_feat = out.cpu().numpy()
            val_embeddings.append(learned_feat)
            val_labels.append(data.y.long().cpu().numpy())

            # Calculate predictions and loss
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out, data.y, reduction='sum').item()

    # Calculate accuracy and average loss
    accuracy = correct / len(val_loader.dataset)
    average_loss = loss / len(val_loader.dataset)

    # Return predictions, ground truth labels, and metrics
    val_embeddings = np.vstack(val_embeddings)
    val_labels = np.concatenate(val_labels)

    result = {
        'accuracy': accuracy,
        'average_loss': average_loss
    }

    return val_embeddings, val_labels, result

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

def evaluate(train_loader, val_loader, model_path, num_features, num_classes, args, device):
    # Load the model
    model = load_model(model_path, num_features, num_classes, args, device)
    model.eval()

    train_embeddings = []
    train_labels = []
    val_embeddings = []
    val_labels = []

    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            out = model(data)
            learned_feat = out[0].cpu().data.numpy()
            train_embeddings.append(learned_feat)
            train_labels.append(data.y.long().cpu().numpy())

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
    neigh.fit(train_embeddings, train_labels)

    val_preds = neigh.predict(val_embeddings)
    train_preds = neigh.predict(train_embeddings)

    result = {
        'prec': metrics.precision_score(val_labels, val_preds, average='macro'),
        'recall': metrics.recall_score(val_labels, val_preds, average='macro'),
        'acc': metrics.accuracy_score(val_labels, val_preds),
        'F1': metrics.f1_score(val_labels, val_preds, average="micro"),
        'train acc': metrics.accuracy_score(train_labels, train_preds)
    }

    # Return predictions and ground truth labels
    return val_preds, val_labels, result

def evaluate_mlp(train_loader, val_loader, model_path, num_features, num_classes, args, device):
    # Load the model
    model = load_model(model_path, num_features, num_classes, args, device)
    model.eval()

    train_embeddings = []
    train_labels = []
    val_embeddings = []
    val_labels = []

    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            out = model(data)
            learned_feat = out[0].cpu().data.numpy()
            train_embeddings.append(learned_feat)
            train_labels.append(data.y.long().cpu().numpy())


        for data in val_loader:
            data = data.to(device)
            out = model(data)
            learned_feat = out[0].cpu().data.numpy()
            val_embeddings.append(learned_feat)
            val_labels.append(data.y.long().cpu().numpy())

    val_embeddings = np.vstack(val_embeddings)
    val_labels = np.concatenate(val_labels)

    # Initialization of the MLP classifier
    in_feat = val_embeddings.shape[1]
    pred_layers = [
        nn.Linear(in_feat, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 32),
        nn.LeakyReLU(),
        nn.Linear(32, num_classes)
    ]
    pred_model = nn.Sequential(*pred_layers).to(device)

    # Train the MLP classifier
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    pred_model.train()
    for epoch in range(100):  # You can adjust the number of epochs
        optimizer.zero_grad()
        outputs = pred_model(torch.FloatTensor(val_embeddings).to(device))
        loss = criterion(outputs, torch.LongTensor(val_labels).to(device))
        loss.backward()
        optimizer.step()

    # Evaluate the MLP classifier
    pred_model.eval()
    with torch.no_grad():
        outputs = pred_model(torch.FloatTensor(val_embeddings).to(device))
        _, val_preds = torch.max(outputs, 1)
        val_preds = val_preds.cpu().numpy()

    result = {
        'prec': metrics.precision_score(val_labels, val_preds, average='macro'),
        'recall': metrics.recall_score(val_labels, val_preds, average='macro'),
        'acc': metrics.accuracy_score(val_labels, val_preds),
        'F1': metrics.f1_score(val_labels, val_preds, average="micro")
    }

    # Return predictions and ground truth labels
    return val_preds, val_labels, result


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=32, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')
parser.add_argument('--num_features', type=int, default=3, help='Dimension of input features')
parser.add_argument('--final_dim', type=int, default=32, help='Dimension of final embeddings')
parser.add_argument('--alpha', type=float, default=1.5, help='Margin in the triplet loss')

args = parser.parse_args()

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise RuntimeError("CUDA is not available. Please ensure GPU and CUDA are set up.")

# Load dataset

# dataset_dir = os.path.join('data', args.dataset)
dataset_dir = '/home/emu/Documents/surrogate/cs9'
dataset = GraphDataset(dataset_dir)

num_classes = dataset.num_classes  # Since feasibility is binary (True/False)
num_features = dataset.num_features

print("num_features : ", num_features)
print("num_classes : ", num_classes)
print(len(dataset))

num_train = int(len(dataset)*0.1)
num_val = len(dataset) - num_train

print("num_train : ", num_train)
print("num_val : ", num_val)

train_set, val_set = random_split(dataset, [num_train, num_val])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'best_machine.pth'
val_preds, val_labels, result= evaluate(train_loader, val_loader, model_path, num_features, num_classes, args, device)

# val_preds_mlp, val_labels_mlp, result_mlp= evaluate_mlp(train_loader, val_loader, model_path, num_features, num_classes, args, device)

val_preds_, val_labels_, result_= test_1(val_loader, model_path, num_features, num_classes, args, device)

# Create a DataFrame for the first 20 results
results_df = pd.DataFrame({
    'Ground Truth': val_labels[:20],
    'Predicted': val_preds[:20]
})

# results_df_mlp = pd.DataFrame({
#     'Ground Truth': val_labels_mlp[:20],
#     'Predicted': val_preds_mlp[:20]
# })

# results_df_ = pd.DataFrame({
#     'Ground Truth': val_labels_[:20],
#     'Predicted': val_preds_[:20]
# })

# Print the table
print(results_df)
print(result)

# print("MLP RESULT")
# print(results_df_mlp)
# print(result_mlp)

print("NORM RESULT")
# print(results_df_)
print(result_)

