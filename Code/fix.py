import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime

def recreate_graphs_from_csv(csv_file_path, new_model_name):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Extract information from the original file name
    original_file_name = os.path.basename(csv_file_path)
    parts = original_file_name.split('_')
    iteration = parts[-3]  # Assuming the iteration is the third-to-last part
    
    # Create the new folder path
    base_folder = os.path.dirname(os.path.dirname(csv_file_path))
    model_folder = f"{base_folder}/{new_model_name}"
    os.makedirs(model_folder, exist_ok=True)

    # Save to CSV with new model name
    csv_filename = f"{model_folder}/{new_model_name}_{iteration}.csv"
    df.to_csv(csv_filename, index=False)

    # Set up Times New Roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 24})  # Global font size increase

    # Create and save accuracy graph
    plt.figure(figsize=(12, 6))
    plt.plot(df['Epoch'].to_numpy(), df['Train Accuracy'].to_numpy() * 100, label='Train Accuracy', marker='o')
    plt.plot(df['Epoch'].to_numpy(), df['Validation Accuracy'].to_numpy() *100, label='Validation Accuracy', marker='x')
    plt.title(f"{new_model_name} Training History - Accuracy {iteration}", fontsize=24, pad=20)
    plt.xlabel("Epoch", fontsize=24, labelpad=10)
    plt.ylabel("Accuracy [%]", fontsize=24, labelpad=10)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9)

    accuracy_graph_filename = f"{model_folder}/{new_model_name}_{iteration}_accuracy.png"
    plt.savefig(accuracy_graph_filename, bbox_inches='tight')
    plt.close()

    # Create and save loss graph
    plt.figure(figsize=(12, 6))
    plt.plot(df['Epoch'].to_numpy(), df['Train Loss'].to_numpy(), label='Train Loss', marker='o')
    plt.plot(df['Epoch'].to_numpy(), df['Validation Loss'].to_numpy(), label='Validation Loss', marker='x')
    plt.title(f"{new_model_name} Training History - Loss {iteration}", fontsize=24, pad=20)
    plt.xlabel("Epoch", fontsize=24, labelpad=10)
    plt.ylabel("Loss", fontsize=24, labelpad=10)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9)

    loss_graph_filename = f"{model_folder}/{new_model_name}_{iteration}_loss.png"
    plt.savefig(loss_graph_filename, bbox_inches='tight')
    plt.close()


def process_folder(folder_path, new_model_name):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file_path = os.path.join(folder_path, filename)
            recreate_graphs_from_csv(csv_file_path, new_model_name)

# Usage example:
csv_file_path = "/home/emu/Documents/surrogate/Code/experiment_results_m2n/cs8/6th_m2n_model"
new_model_name = "edge-based_model"
# base_folder = "/home/emu/Documents/two-stage-gnn/Code/sag/experiment_results_1/cs8/new_model/"
process_folder(csv_file_path, new_model_name)
