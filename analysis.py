import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_heatmap(csv_path, grid_size, save_path=None):
    df = pd.read_csv(csv_path)
    df = df[df["Grid Rows"] == grid_size[0]]  # assumes square grid

    heatmap = np.zeros(grid_size)

    for _, row in df.iterrows():
        idx = int(row["Patch Index"])
        r = idx // grid_size[1]
        c = idx % grid_size[1]
        heatmap[r][c] = row["LPIPS Score"]

    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title(f"LPIPS Heatmap ({grid_size[0]}x{grid_size[1]})")
    plt.colorbar(label='LPIPS Score')
    plt.xlabel("Column")
    plt.ylabel("Row")

    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
        plt.close()
    else:
        plt.show()

def analyze_threshold_grid(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    grouped = df.groupby(["Grid Rows", "Threshold"]).size().unstack(fill_value=0)
    
    grouped.plot(kind='bar', stacked=True)
    plt.title("Damage Detection Counts by Grid Size and Threshold")
    plt.xlabel("Grid Rows")
    plt.ylabel("Patches Detected (LPIPS > Threshold)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    csv_path = "lpips_log.csv"

    if not os.path.exists(csv_path):
        print("CSV log not found. Run LPIPS scoring first.")
        exit()

    os.makedirs("images/heatmaps", exist_ok=True)
    os.makedirs("images/threshold_grids", exist_ok=True)

    # Show heatmap for a specific grid

    for grid in [(2, 2), (4, 4), (8, 8), (16, 16)]:
        heatmap_path = f"images/heatmaps/heatmap_{grid[0]}x{grid[1]}.png"
        visualize_heatmap(csv_path, grid_size=grid, save_path=heatmap_path)

    # Show aggregate analysis
    analyze_threshold_grid(csv_path, save_path="images/threshold_grids/threshold_analysis.png")
