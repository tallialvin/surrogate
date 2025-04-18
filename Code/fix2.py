import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime
import numpy as np

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
    plt.rcParams.update({'font.size': 18})  # Global font size

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    # Capitalize the first letter of each word in the model name
    capitalized_model_name = new_model_name.capitalize()

    fig.suptitle(f"Training History of {capitalized_model_name} : Model No.{iteration}", fontsize=24, y=1.0)

    # Get the maximum epoch number
    max_epoch = df['Epoch'].max()

    # Accuracy subplot (left)
    ax1.plot(df['Epoch'].to_numpy(), df['Train Accuracy'].to_numpy() * 100, label='Train Accuracy', marker='o')
    ax1.plot(df['Epoch'].to_numpy(), df['Validation Accuracy'].to_numpy() * 100, label='Validation Accuracy', marker='x')
    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Accuracy [%]", fontsize=20)
    ax1.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    ax1.set_xlim(0, max_epoch)  # Set x-axis limits from 0 to max epoch
    ax1.legend(fontsize=16)
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Loss subplot (right)
    ax2.plot(df['Epoch'].to_numpy(), df['Train Loss'].to_numpy(), label='Train Loss', marker='o')
    ax2.plot(df['Epoch'].to_numpy(), df['Validation Loss'].to_numpy(), label='Validation Loss', marker='x')
    ax2.set_xlabel("Epoch", fontsize=20)
    ax2.set_ylabel("Loss", fontsize=20)
    ax2.set_ylim(0, 2)  # Set y-axis limits from 0 to 2
    ax2.set_xlim(0, max_epoch)  # Set x-axis limits from 0 to max epoch
    ax2.legend(fontsize=16)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # Set integer ticks for x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Adjust top margin to accommodate title

    # Save the combined graph
    combined_graph_filename = f"{model_folder}/{new_model_name}_{iteration}_combined.png"
    plt.savefig(combined_graph_filename, bbox_inches='tight')
    plt.close()

def process_folder(folder_path, new_model_name):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_file_path = os.path.join(folder_path, filename)
            recreate_graphs_from_csv(csv_file_path, new_model_name)

# Usage example:
csv_file_path = "/home/emu/Documents/surrogate/Code/experiment_results_m2n/cs8/6th_m2n_model"
new_model_name = "edge-based"
process_folder(csv_file_path, new_model_name)
