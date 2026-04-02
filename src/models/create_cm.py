import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 1. Setup Data
cm_data = np.array([[1059, 361], 
                    [777, 577]]) 

# 2. Calculate percentages by ROW (Recall-based)
row_sums = cm_data.sum(axis=0, keepdims=True)
percentages = cm_data / row_sums

# 3. Plotting
plt.figure(figsize=(10, 8))
plt.rcParams.update(plt.rcParamsDefault)

# Use 'percentages' for the heatmap color intensities
ax = sns.heatmap(
    percentages, 
    annot=False, 
    cmap="Blues", 
    xticklabels=["person", "background"], 
    yticklabels=["person", "background"],
    vmin=0, vmax=1,
    cbar=True
)

# 4. Manually add the combined labels
for i in range(2):
    for j in range(2):
        count = cm_data[i, j]
        pct = percentages[i, j]
        
        # Color logic: use white text on dark backgrounds
        color = "white" if pct > 0.5 else "black"
        
        # Draw Percentage (BOLD)
        ax.text(j + 0.5, i + 0.45, f"{pct:.1%}", 
                ha="center", va="center", 
                fontsize=18, color=color, weight="bold")
        
        # Draw Absolute Count (NORMAL)
        ax.text(j + 0.5, i + 0.55, f"({count})", 
                ha="center", va="center", 
                fontsize=18, color=color, weight="normal")

# 5. Labels and Titles
plt.xlabel("True", fontsize=18)
plt.ylabel("Predicted", fontsize=18)
plt.tight_layout(pad=1.0)

# Save for your RMarkdown
output_dir = "/home/nele_pauline_suffo/outputs/person_detection/yolo12l_20260322_204343/yolo12l_det_validation_20260326_230132_best_conf"
plt.savefig(f"{output_dir}/confusion_matrix_enhanced.png", dpi=300)
print(f"Confusion matrix saved to: {output_dir}/confusion_matrix_enhanced.png")