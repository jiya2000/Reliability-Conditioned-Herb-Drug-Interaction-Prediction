"""
Generate Publication-Quality Architecture Diagram for Reliability-Conditioned HDI Prediction.
Outputs a high-resolution PNG (300 DPI) to results/figures/architecture_diagram.png.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, x, y, width, height, label, sublabel="", color="#E3F2FD", edgecolor="#1E88E5", fontsize=10):
    """Draws a rounded rectangular box with title and subtitle."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor=color, edgecolor=edgecolor, linewidth=1.5,
        zorder=2
    )
    ax.add_patch(box)
    
    if sublabel:
        ax.text(x + width/2, y + height*0.62, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="#1A237E", zorder=3)
        ax.text(x + width/2, y + height*0.28, sublabel, ha="center", va="center", 
                fontsize=fontsize-2, color="#37474F", zorder=3, style="italic")
    else:
        ax.text(x + width/2, y + height/2, label, ha="center", va="center", 
                fontsize=fontsize, fontweight="bold", color="#1A237E", zorder=3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, label="", color="#455A64", rad=0.0, lw=1.5):
    """Draws a connecting arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.3,head_length=0.4",
            color=color, linewidth=lw,
            connectionstyle=f"arc3,rad={rad}" if rad != 0.0 else "arc3"
        ),
        zorder=1
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.03, label, ha="center", va="bottom", 
                fontsize=8, color="#263238", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
                zorder=4)

def generate_architecture_diagram(output_path="results/figures/architecture_diagram.png"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    # Background panel dividing the 4 main columns
    # Panel 1: Data Sources
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.20, 0.90, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor="#E0E0E0", zorder=0))
    ax.text(0.12, 0.92, "1. Data Sources & Extraction", ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")
    
    # Panel 2: Representation & Metadata
    ax.add_patch(FancyBboxPatch((0.24, 0.05), 0.24, 0.90, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor="#E0E0E0", zorder=0))
    ax.text(0.36, 0.92, "2. Encoders & 5-Tuple Metadata", ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")

    # Panel 3: Reliability & Gated Fusion (Novelties)
    ax.add_patch(FancyBboxPatch((0.50, 0.05), 0.27, 0.90, boxstyle="round,pad=0.02", facecolor="#FFF8E1", edgecolor="#FFE082", zorder=0))
    ax.text(0.635, 0.92, "3. Novel Reliability Gating & Losses", ha="center", va="center", fontsize=13, fontweight="bold", color="#F57F17")

    # Panel 4: Prediction & Explanation
    ax.add_patch(FancyBboxPatch((0.79, 0.05), 0.19, 0.90, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor="#E0E0E0", zorder=0))
    ax.text(0.885, 0.92, "4. Downstream Tasks", ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")

    # --- Column 1: Data Sources ---
    draw_box(ax, 0.04, 0.72, 0.16, 0.12, "Knowledge Graph (KG)", "DrugBank / ChEMBL / IMPPAT\n(229 Nodes, 2400 Edges)", color="#E1F5FE", edgecolor="#0288D1")
    draw_box(ax, 0.04, 0.45, 0.16, 0.12, "Code-Mixed Forums", "Practo / 1mg / Reddit\n(Hindi-English Text)", color="#F3E5F5", edgecolor="#8E24AA")
    draw_box(ax, 0.04, 0.18, 0.16, 0.12, "Entity Linker & NER", "Maps colloquial names\nto Chemical IDs", color="#E8F5E9", edgecolor="#388E3C")

    # --- Column 2: Encoders & Metadata ---
    draw_box(ax, 0.26, 0.72, 0.20, 0.12, "RGCN Graph Encoder", "Relational Topological\nMolecular Embeddings (H_G)", color="#E1F5FE", edgecolor="#0288D1")
    draw_box(ax, 0.26, 0.45, 0.20, 0.12, "BiomedBERT Encoder", "Semantic Biomedical\nText Embeddings (H_T)", color="#F3E5F5", edgecolor="#8E24AA")
    draw_box(ax, 0.26, 0.18, 0.20, 0.14, "Candidate Metadata (5-Tuple)", "C: Corroboration Count | T: Recency\nB: Biomedical Quality | M: Mol. Plausibility\nS: Source Type Code (1-6)", color="#FFF9C4", edgecolor="#FBC02D")

    # --- Column 3: Novel Contributions ---
    draw_box(ax, 0.52, 0.68, 0.23, 0.14, "Multi-Granularity Attention", "Token / Sentence / Document\nHierarchical Cross-Attention Gate", color="#FFE0B2", edgecolor="#F57C00", fontsize=11)
    draw_box(ax, 0.52, 0.42, 0.23, 0.14, "Reliability Scorer + MC Dropout", "Stochastic Forward Passes (K=10)\nOutputs: Mean R ± Uncertainty σ_R", color="#FFCCBC", edgecolor="#D84315", fontsize=11)
    
    # Novel Losses Box
    draw_box(ax, 0.52, 0.12, 0.23, 0.22, "Novel Training Objectives & Alerts", 
             "1. ARC Loss: Corr(R,Acc) + Div + Ord\n2. Contrastive InfoNCE: R-weighted margin\n3. Temporal Drift: CUSUM + KL Shift Alerts", 
             color="#FFCDD2", edgecolor="#C62828", fontsize=10)

    # --- Column 4: Prediction ---
    draw_box(ax, 0.81, 0.65, 0.15, 0.12, "Link Predictor MLP", "Predicts HDI Interaction\nProbability P ∈ [0, 1]", color="#C8E6C9", edgecolor="#2E7D32", fontsize=11)
    draw_box(ax, 0.81, 0.35, 0.15, 0.14, "Explainability Engine", "Surfaces Corroborating\nSentences & Reliability R\nMaps Pharmacological Pathway", color="#E0F2F1", edgecolor="#00897B", fontsize=10)

    # --- ARROWS ---
    # Col 1 -> Col 2
    draw_arrow(ax, 0.20, 0.78, 0.26, 0.78, color="#0288D1", lw=2)
    draw_arrow(ax, 0.20, 0.51, 0.26, 0.51, color="#8E24AA", lw=2)
    draw_arrow(ax, 0.12, 0.45, 0.12, 0.30, color="#666666", lw=1.5)
    draw_arrow(ax, 0.20, 0.24, 0.26, 0.24, color="#FBC02D", lw=2)
    
    # Col 2 -> Col 3
    draw_arrow(ax, 0.46, 0.78, 0.52, 0.78, label="H_G", color="#0288D1", lw=2)
    draw_arrow(ax, 0.46, 0.51, 0.52, 0.73, label="H_T", color="#8E24AA", lw=2, rad=-0.1)
    draw_arrow(ax, 0.46, 0.25, 0.52, 0.49, label="(C,T,B,M,S)", color="#F57F17", lw=2, rad=-0.1)

    # Scorer -> Attention Gate (The core novelty!)
    draw_arrow(ax, 0.635, 0.56, 0.635, 0.68, label="Gate = (1 - α·σ_R)·R", color="#D84315", lw=2.5)

    # Scorer -> Novel Losses
    draw_arrow(ax, 0.635, 0.42, 0.635, 0.34, label="R, σ_R", color="#C62828", lw=2)

    # Attention -> Prediction & Explanations
    draw_arrow(ax, 0.75, 0.75, 0.81, 0.71, label="Fused Vector", color="#2E7D32", lw=2.5)
    draw_arrow(ax, 0.75, 0.72, 0.81, 0.45, label="Attention Weights", color="#00897B", lw=2, rad=0.1)
    draw_arrow(ax, 0.75, 0.49, 0.81, 0.42, label="R_mean", color="#D84315", lw=1.5)

    # Title
    fig.suptitle("Reliability-Conditioned HDI Prediction Architecture (Current System)", fontsize=18, fontweight="bold", color="#1A237E", y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated high-resolution architecture diagram at: {Path(output_path).resolve()}")

if __name__ == "__main__":
    generate_architecture_diagram()
