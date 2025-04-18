import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import torch

def save_history_results(epoch_results, model_folder, model_name, iteration, model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(model_folder, exist_ok=True)

    # Prepare data
    df = pd.DataFrame(epoch_results)
    
    # Save to CSV
    csv_filename = f"{model_folder}/{model_name}_{iteration}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)

        # Set up Times New Roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 24})  # Global font size increase

    # Create and save accuracy graph
    plt.figure(figsize=(12, 6))
    plt.plot(df['Epoch'].to_numpy(), df['Train Accuracy'].to_numpy() * 100, label='Train Accuracy', marker='o')
    plt.plot(df['Epoch'].to_numpy(), df['Validation Accuracy'].to_numpy() *100, label='Validation Accuracy', marker='x')
    plt.title(f"{model_name} Training History - Accuracy {iteration}", fontsize=24, pad=20)
    plt.xlabel("Epoch", fontsize=24, labelpad=10)
    plt.ylabel("Accuracy [%]", fontsize=24, labelpad=10)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9)

    accuracy_graph_filename = f"{model_folder}/{model_name}_{iteration}_accuracy.png"
    plt.savefig(accuracy_graph_filename, bbox_inches='tight')
    plt.close()

    # Create and save loss graph
    plt.figure(figsize=(12, 6))
    plt.plot(df['Epoch'].to_numpy(), df['Train Loss'].to_numpy(), label='Train Loss', marker='o')
    plt.plot(df['Epoch'].to_numpy(), df['Validation Loss'].to_numpy(), label='Validation Loss', marker='x')
    plt.title(f"{model_name} Training History - Loss {iteration}", fontsize=24, pad=20)
    plt.xlabel("Epoch", fontsize=24, labelpad=10)
    plt.ylabel("Loss", fontsize=24, labelpad=10)
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9)

    loss_graph_filename = f"{model_folder}/{model_name}_{iteration}_loss.png"
    plt.savefig(loss_graph_filename, bbox_inches='tight')
    plt.close()


    # Save the model
    model_filename = f"{model_folder}/{model_name}_{iteration}_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)

    print(f"Results for iteration {iteration} saved to {csv_filename}")
    print(f"Accuracy Graph for iteration {iteration} saved to {accuracy_graph_filename}")
    print(f"Loss Graph for iteration {iteration} saved to {loss_graph_filename}")
    print(f"Model for iteration {iteration} saved to {model_filename}")

# Example usage in your main script:
# for iteration in range(args.iterations):
#     epoch_results = []
#     for epoch in range(num_epochs):
#         # Your training code here
#         epoch_result = {
#             'Epoch': epoch + 1,
#             'Train Accuracy': train_acc,
#             'Validation Accuracy': val_acc,
#             'Test Accuracy': test_acc
#         }
#         epoch_results.append(epoch_result)
#     
#     save_epoch_results(epoch_results, "YourModelName", iteration + 1)
