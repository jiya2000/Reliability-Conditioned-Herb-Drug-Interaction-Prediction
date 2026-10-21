"""
Generate Official Patent-Grade Architectural Flowchart (Figure 1)
Strictly compliant with Indian Patent Office (Rule 15) and USPTO (37 CFR 1.84).
Generates high-resolution 300 DPI PNG and vector PDF with perfect typography and balanced margins.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_patent_figure():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "patent_figure1_official_sheet.png"
    pdf_path = output_dir / "patent_figure1_official_sheet.pdf"

    # Standard A4/US Letter portrait proportion (8.5 x 11 inches)
    fig, ax = plt.subplots(figsize=(9, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.axis("off")

    def draw_box(x, y, w, h, title, body="", fill="#FFFFFF", edge="#000000", lw=1.5, title_fs=10, body_fs=8.5):
        # Draw clean rectangle with slight rounding
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.0",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3
        )
        ax.add_patch(rect)
        cx = x + w / 2
        if body:
            ax.text(cx, y + h * 0.70, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#000000", zorder=4)
            ax.text(cx, y + h * 0.32, body, ha="center", va="center", fontsize=body_fs, color="#222222", zorder=4, linespacing=1.2)
        else:
            ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#000000", zorder=4)

    def draw_diamond(cx, cy, w, h, title, body="", fill="#F8F9FA", edge="#000000", lw=1.5, title_fs=9, body_fs=7.5):
        pts = [
            [cx, cy + h / 2],
            [cx + w / 2, cy],
            [cx, cy - h / 2],
            [cx - w / 2, cy]
        ]
        poly = patches.Polygon(pts, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3)
        ax.add_patch(poly)
        if body:
            ax.text(cx, cy + h * 0.12, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#000000", zorder=4)
            ax.text(cx, cy - h * 0.18, body, ha="center", va="center", fontsize=body_fs, color="#222222", zorder=4)
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color="#000000", zorder=4)

    def draw_arrow(x1, y1, x2, y2, label="", rad=0.0, label_offset=(0, 1.2), label_fs=8):
        conn = f"arc3,rad={rad}" if rad != 0 else "arc3"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", color="#000000", lw=1.4, connectionstyle=conn),
            zorder=2
        )
        if label:
            mx = (x1 + x2) / 2 + label_offset[0]
            my = (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center", fontsize=label_fs, fontweight="bold", color="#000000",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="#888888", lw=0.6), zorder=5)

    # Header / Sheet Label
    ax.text(50, 127, "FIG. 1: SYSTEM ARCHITECTURE & OPERATIONAL FLOW", ha="center", va="center", fontsize=11, fontweight="bold", color="#000000")

    # 100: Query
    draw_box(30, 116, 40, 7, "100: CANDIDATE QUERY", "(Botanical Entity h, Pharmaceutical Drug d)", title_fs=9.5, body_fs=8)

    # Branches
    # 102: Bio Branch (Left)
    draw_box(4, 95, 42, 14, "102: BIOLOGICAL MECHANISM BRANCH",
             "• 104: Tripartite Knowledge Graph\n  CYP450s (3A4, 2D6, 2C9) & ABCB1/P-gp\n• 106: 2-Layer Heterogeneous GAT Encoder",
             title_fs=9, body_fs=8)

    # 108: Literature Branch (Right)
    draw_box(54, 95, 42, 14, "108: EVIDENTIARY LINEAGE BRANCH",
             "• 110: Audited Registry (DOIs/PMIDs)\n• 112: Citation DAG (Echo Discount α = 0.15)\n• 114: Context Compatibility (Species, Prep, Route)\n• 116: BiomedBERT Contextual Text Encoder",
             title_fs=9, body_fs=7.5)

    # Arrows from Query to Branches
    draw_arrow(40, 116, 25, 109)
    draw_arrow(60, 116, 75, 109)

    # 118: Scorer (under 108)
    draw_box(56, 78, 38, 8, "118: NEURAL RELIABILITY SCORER",
             "Inputs: Lineage Stats & Contradiction Index\nOutput: Scalar Reliability Gate R ∈ [0, 1]",
             title_fs=8.5, body_fs=7.5)
    draw_arrow(75, 95, 75, 86, label="Lineage Stats", label_offset=(5, 0))

    # 120: Fusion
    draw_box(22, 63, 56, 8, "120: RELIABILITY-GATED MULTIMODAL FUSION",
             r"$\mathbf{z}_{\mathrm{fused}} = (1 - R)\,\mathbf{z}_{\mathrm{graph}} + R\,\mathbf{z}_{\mathrm{contextual}}$",
             title_fs=9.5, body_fs=9)

    # Connections to Fusion
    draw_arrow(25, 95, 36, 71, label=r"$\mathbf{z}_{\mathrm{graph}}$", label_offset=(-3, 2))
    draw_arrow(65, 95, 52, 71, label=r"$\mathbf{z}_{\mathrm{contextual}}$", label_offset=(4, 6), rad=0.1)
    draw_arrow(65, 78, 56, 71, label="Gate R", label_offset=(0, 2))

    # 122: Temperature Calibration
    draw_box(25, 49, 50, 7.5, "122: TEMPERATURE-SCALED POST-HOC CALIBRATION",
             r"Calibrated Probability $\hat{p} = \sigma(z / T^*)$, where $T^* = 1.48$",
             title_fs=9, body_fs=8.5)
    draw_arrow(50, 63, 50, 56.5)

    # 124: Tri-state decision diamond
    draw_diamond(50, 36, 58, 16, "124: TRI-STATE DECISION ENGINE",
                 r"Epistemic $u > \tau_u$ OR (Conflict $C > \tau_c$ AND $|\hat{p} - 0.5| < \tau_m$)?",
                 title_fs=8.5, body_fs=7.0)
    draw_arrow(50, 49, 50, 44.0)

    # Left Branch: 126 Abstain
    draw_box(4, 20, 38, 7.5, "126: STATE 3: ABSTAIN", "Selective Clinical Withholding", title_fs=8.5, body_fs=7.5)
    draw_arrow(32, 36, 23, 27.5, label="YES (Uncertain)", label_offset=(-4, 2))

    # 128 Review Queue
    draw_box(4, 7, 38, 8, "128: CLINICAL PHARMACIST QUEUE", "Asynchronous Review Queue\n(Non-Intrusive EHR Advisory Flag)", title_fs=8.5, body_fs=7)
    draw_arrow(23, 20, 23, 15)

    # Right Branch: 130 Probability Threshold
    draw_diamond(74, 23.5, 34, 10, "130: Probability Threshold", r"Is Calibrated $\hat{p} \geq 0.50$?", title_fs=8, body_fs=7.5)
    draw_arrow(68, 36, 74, 28.5, label="NO (Definitive)", label_offset=(4, 2))

    # 132 Positive Interaction
    draw_box(48, 7, 24, 8, "132: STATE 1: INTERACTION", "Automated Prescription\nInterlock Alert", title_fs=8, body_fs=7)
    draw_arrow(65, 18.5, 60, 15, label="YES", label_offset=(-2, 0.5))

    # 134 Negative Safe
    draw_box(74, 7, 24, 8, "134: STATE 2: NO INTERACTION", "Safe Co-Administration\nClearance", title_fs=8, body_fs=7)
    draw_arrow(83, 18.5, 86, 15, label="NO", label_offset=(2, 0.5))

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.savefig(output_dir / "patent_system_architecture_diagram.png", dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"Generated official patent sheets: {png_path} and {pdf_path}")

if __name__ == "__main__":
    generate_patent_figure()
