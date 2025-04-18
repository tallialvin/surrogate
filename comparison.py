fes = [1,1,1,1,1,1,1,1,1,1]
man_fes = [0,0,0,0,0,0,0,0,0,0]
GT_1 = [0,1,0,0,0,0,1,1,0,0]
GT_2 = [0,0,0,0,0,0,0,1,0,0,0]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_comparison(model_name, predictions, gt, ax):
    cm = confusion_matrix(gt[:len(predictions)], predictions)
    
    # Create custom annotations with False/True labels and larger font
    annot = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            if i == 0 and j == 0:
                row.append(f"TN\n{cm[i,j]}")
            elif i == 0 and j == 1:
                row.append(f"FP\n{cm[i,j]}")
            elif i == 1 and j == 0:
                row.append(f"FN\n{cm[i,j]}")
            elif i == 1 and j == 1:
                row.append(f"TP\n{cm[i,j]}")
        annot.append(row)
    
    # Configure all font sizes
    sns.set(font_scale=1.5)  # Controls base font size
    heatmap = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['False', 'True'],
                yticklabels=['False', 'True'],
                annot_kws={"size": 16})  # Annotation font size
    
    # Set title and labels with larger font
    ax.set_title(f'{model_name}\nAccuracy: {accuracy_score(gt[:len(predictions)], predictions):.2f}', 
                 fontsize=14, pad=20)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    
    # Adjust tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)

# Create comparison plots (same size)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# GT_1 Comparison
plot_comparison('FES vs GT_1', fes, GT_1, axes[0,0])
plot_comparison('MAN_FES vs GT_1', man_fes, GT_1, axes[0,1])

# GT_2 Comparison
plot_comparison('FES vs GT_2', fes, GT_2, axes[1,0])
plot_comparison('MAN_FES vs GT_2', man_fes, GT_2, axes[1,1])

plt.tight_layout()
plt.show()
