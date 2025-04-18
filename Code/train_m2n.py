from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import torch
from torch.autograd import Variable
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric import utils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import argparse
import os
from network import Net
from tripletnet import tripletnet
from triplet_sampler import TripletSampler


from dataset_m2n import GraphDataset
from store_result import save_history_results
from tqdm import tqdm
import logging

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

# Hyperparameters & Setup

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD')
parser.add_argument('--iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--epochs', type=int, default=100000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50, help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')
parser.add_argument('--num_features', type=int, default=64, help='Dimension of input features')
parser.add_argument('--final_dim', type=int, default=64, help='Dimension of final embeddings')
parser.add_argument('--alpha', type=float, default=1.5, help='Margin in the triplet loss')

args = parser.parse_args()

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise RuntimeError("CUDA is not available. Please ensure GPU and CUDA are set up.")

base_folder = "experiment_results_m2n/cs8/"
model_name = "non_label1"
model_folder = f"{base_folder}/{model_name}"
os.makedirs(model_folder, exist_ok=True)

# Load dataset

# dataset_dir = os.path.join('data', args.dataset)
case_num = 8
folder = "cs8"
dataset_dir = '/home/emu/Documents/surrogate/dataset'

# print(dataset_dir)
# exit()
dataset = GraphDataset(dataset_dir, case_num, folder)

num_classes = dataset.num_classes  # Since feasibility is binary (True/False)
num_features = dataset.num_features

print("num_features : ", num_features)
print("num_classes : ", num_classes)

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)

# Train
min_loss = 1e10
patience = 0

val_accs = []
test_accs = []
num_iter=0

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        filename=f"{base_folder}/experiment_log.txt",
        filemode='a'  # 'w' will overwrite, use 'a' to append
    )



for iter in range(args.iterations):
    threshold = 0.6
    if num_training == 100:
            threshold = 0.65

    while True:
        print("val accs: ", val_accs)
        print("test accs: ", test_accs)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        model = Net(num_features, args.nhid, args.final_dim, args.pooling_ratio, args.dropout_ratio).to(device)

        # Triplet Net
        TNet = tripletnet(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Triplet Sampler of train graphs
        tripletsampler_tr = TripletSampler(training_set)

        # Triplet loss function
        criterion = torch.nn.MarginRankingLoss(margin=args.alpha)

        # FIRST STAGE: TRAINING WITH TRIPLET LOSS
        for epoch in range(args.epochs):

            tripletsampler_tr.shuffle()

            model.train()
            while not tripletsampler_tr.end():
                one_triplet = tripletsampler_tr.sampler()

                # data = data.to(args.device)
                dist_p, dist_n, embed_a, embed_p, embed_n = TNet(one_triplet['anchor'].to(device),
                                                                one_triplet['pos'].to(device),
                                                                one_triplet['neg'].to(device))
                target = torch.FloatTensor(dist_p.size()).fill_(-1)
                target = Variable(target).to(device)

                loss = criterion(dist_p, dist_n, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # SECOND STAGE: FINE-TUNING

        # Triplet Sampler of train graphs
        tripletsampler_tr = TripletSampler(training_set)
        # Train & Validation Loader
        train_loader = DataLoader(training_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=1, shuffle=True)

        # Initialization of the final classifier
        in_feat = args.final_dim
        pred_layers = []
        pred_layers.append(nn.Linear(in_feat, 64).to(device))
        pred_layers.append(nn.LeakyReLU())
        pred_layers.append(nn.Linear(64, 32).to(device))
        pred_layers.append(nn.LeakyReLU())
        pred_layers.append(nn.Linear(32, 2).to(device))
        pred_model = nn.Sequential(*pred_layers).to(device)

        # The to-be-finetuned model and the optimizer
        model.map2_model = pred_model
        optimizer_2 = torch.optim.Adam(model.parameters(), lr=0.001)

        # result = evaluate(training_set, training_set, model, device)
        train_acc, train_loss = test(model, val_loader)

        # result = evaluate(training_set, validation_set, model, device)
        # train_acc = result['acc']
        print("train acc before post-train : " + str(train_acc))

        # Fine-tuning

        epoch_results = []

        epoch_result = {
                'Epoch': 0,
                'Train Accuracy': 0,
                'Train Loss': 2,
                'Validation Accuracy': 0,
                'Validation Loss': 2
            }
        
        epoch_results.append(epoch_result)

        pbar = tqdm(total=args.epochs, desc="Training Progress")

        for epoch in range(args.epochs):
            posttrain_loss = 0.0
            iter = 0

            model.train()

            for i, data in enumerate(train_loader):
                optimizer_2.zero_grad()
                data = data.to(device)
                out = model(data)

                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer_2.step()
                optimizer_2.zero_grad()

            val_acc, val_loss = test(model, val_loader)
            
            if val_loss < min_loss:
                model_filename = f"{model_folder}/{model_name}_latest.pth"
                torch.save(model.state_dict(), model_filename)
                # print(f"Model saved at epoch {epoch}")
                min_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience > patience:
                break

            # result = evaluate(training_set, training_set, model, device)
            train_acc, train_loss = test(model, train_loader)

            epoch_result = {
                'Epoch': epoch + 1,
                'Train Accuracy': train_acc,
                'Train Loss': train_loss,
                'Validation Accuracy': val_acc,
                'Validation Loss': val_loss
            }

            epoch_results.append(epoch_result)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({'train_acc': f'{train_acc:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}'})

        # Close the progress bar
        pbar.close()

        print("ITERATIONS NUMBER : ",num_iter)

        # Evaluate the results
        # test_result = evaluate(training_set, test_set, model, name='Test', max_num_examples=100)
        test_result, _  = test(model, test_loader)
        

        # val_result = evaluate(training_set, validation_set, model, name='Validation', max_num_examples=100)
        val_result, _  = test(model, val_loader)

        print(str(test_result),str(val_result))
        logging.info(f"Iteration {num_iter}:")
        logging.info(f"num_training: {len(training_set)}")
        logging.info(f"num_val: {len(validation_set)}")
        logging.info(f"num_test: {len(test_set)}")
        logging.info(f"Test Accuracy: {test_result}")
        logging.info(f"Validation Accuracy: {val_result}")

        if (test_result > threshold):
            break
    num_iter +=1
    # In your iteration loop
    logging.info('###SAVE#####')
    save_history_results(epoch_results, model_folder, model_name, num_iter + 1, model)
    val_accs.append(val_result)
    test_accs.append(test_result)


# After iterations
logging.info('Validation accuracy summary:')
logging.info(f'Iterations: {args.iterations}')
logging.info(f'Mean Validation Accuracy: {np.mean(val_accs)}')
logging.info(f'Std Validation Accuracy: {np.std(val_accs)}')

logging.info('Test accuracy summary:')
logging.info(f'Iterations: {args.iterations}')
logging.info(f'Mean Test Accuracy: {np.mean(test_accs)}')
logging.info(f'Std Test Accuracy: {np.std(test_accs)}')

logging.info(f'Test Accuracy: {test_accs}')
logging.info(f'Validation Accuracy: {val_accs}')


# Print avg and std
# Validation avg. and std
print('Validation accuracy of ' + str(args.iterations) + ' times is: ' + str(np.mean(val_accs)) + ' with std: ' + str(np.std(val_accs)))
# Test avg. and std
print('Test accuracy of ' + str(args.iterations) + ' times is: ' + str(np.mean(test_accs)) + ' with std: ' + str(np.std(test_accs)))
#best_par = python train_test.py --epochs=60 --dropout_ratio=0.5 --pooling_ratio=0.5 --num_features=32 --nhid=128 --final_dim=32 --alpha=1.5 --batch_size=16