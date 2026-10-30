import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os
import shutil
import textwrap

# Set global styles
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

fig = plt.figure(figsize=(22, 13.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

# Header
fig.text(0.04, 0.965, "FOUR-TIER GROUPED PARTITIONING & SPLIT-CONFORMAL CALIBRATION", 
         fontsize=19, fontweight='bold', color='#0F172A')
fig.text(0.04, 0.940, "Scientific rationale for Bemis–Murcko scaffold clustering (preventing AI memorization) and 4-way data allocation (guaranteed error bars)", 
         fontsize=12, color='#475569')

gs = gridspec.GridSpec(2, 2, figure=fig, top=0.90, bottom=0.05, left=0.04, right=0.96, 
                       hspace=0.26, wspace=0.18)

# -------------------------------------------------------------
# Panel A: Why Grouped Scaffold Splits vs Random? (Top-Left)
# -------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_axis_off()
ax_a.set_title("A. Why Clustered (Scaffold-Disjoint) Instead of Random Splits?", 
               fontsize=14, fontweight='bold', color='#1E293B', loc='left', pad=12)

# Card 1: The Problem with Random Splits
rect_a1 = FancyBboxPatch((0.01, 0.52), 0.98, 0.45, transform=ax_a.transAxes,
                         boxstyle="round,pad=0.015,rounding_size=0.025",
                         facecolor='#FFF1F2', edgecolor='#FECDD3', linewidth=1.3, zorder=1)
ax_a.add_patch(rect_a1)

ax_a.text(0.035, 0.915, "[FAILURE MODE] Standard Random Splits (AI 'Cheating')", 
          transform=ax_a.transAxes, fontsize=12.5, fontweight='bold', color='#BE123C', zorder=2)

desc_a1 = (
    "• High Structural Redundancy: In medicinal chemistry, research groups synthesize dozens of close chemical\n"
    "   analogues sharing identical core skeletons to optimize a single lead molecule.\n\n"
    "• The 'Cheating' Trap: Random 80/20 splitting scatters near-identical chemical analogues across both sets.\n"
    "   The machine learning model merely memorizes the scaffold rather than learning generalizable chemistry.\n\n"
    "• The False Sense of Performance: The algorithm achieves deceptively high scores (e.g. R² > 0.90) on\n"
    "   test analogues, yet suffers catastrophic failure when evaluated on novel botanical phytochemicals."
)
ax_a.text(0.035, 0.85, desc_a1, transform=ax_a.transAxes, fontsize=10.2, color='#4C0519', 
          va='top', zorder=2, linespacing=1.35)

# Card 2: Our Solution
rect_a2 = FancyBboxPatch((0.01, 0.02), 0.98, 0.46, transform=ax_a.transAxes,
                         boxstyle="round,pad=0.015,rounding_size=0.025",
                         facecolor='#F0FDF4', edgecolor='#BBF7D0', linewidth=1.3, zorder=1)
ax_a.add_patch(rect_a2)

ax_a.text(0.035, 0.425, "[SOLUTION] Bemis–Murcko Scaffold Grouping", 
          transform=ax_a.transAxes, fontsize=12.5, fontweight='bold', color='#15803D', zorder=2)

desc_a2 = (
    "• Ring-and-Linker Clustering: Molecules are reduced to their core skeleton via RDKit MurckoScaffold.\n\n"
    "• Zero Cross-Set Leakage: All analogues sharing a core skeleton are locked strictly into ONE role (60/15/10/15).\n\n"
    "• A True Chemical Generalization Test: The held-out test set contains 100% unseen molecular scaffolds.\n\n"
    "• Authentic Botanical Screening: Accurately evaluates whether the model can predict the metabolic\n"
    "   inhibition of novel herbal molecules (Hyperforin, Curcuminoids, Echinacoside) without prior exposure."
)
ax_a.text(0.035, 0.365, desc_a2, transform=ax_a.transAxes, fontsize=10.2, color='#14532D', 
          va='top', zorder=2, linespacing=1.35)


# -------------------------------------------------------------
# Panel B: Why 4 Splits Instead of 3? (Top-Right)
# -------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_axis_off()
ax_b.set_title("B. Why 4 Splits Instead of 3? (The Conformal Guarantee)", 
               fontsize=14, fontweight='bold', color='#1E293B', loc='left', pad=12)

# Card 1: Standard 3-Way Split
rect_b1 = FancyBboxPatch((0.01, 0.52), 0.98, 0.45, transform=ax_b.transAxes,
                         boxstyle="round,pad=0.015,rounding_size=0.025",
                         facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.3, zorder=1)
ax_b.add_patch(rect_b1)

ax_b.text(0.035, 0.915, "Standard 3-Way ML Split (Train / Validation / Test)", 
          transform=ax_b.transAxes, fontsize=12.5, fontweight='bold', color='#475569', zorder=2)

desc_b1 = (
    "• Conventional Practice: Divides data into Train (fitting), Validation (tuning), and Test (audit).\n\n"
    "• Point-Prediction Only: Produces a single scalar potency: e.g. 'pIC50 = 6.2'.\n\n"
    "• Zero Uncertainty Awareness: The model cannot convey whether its confidence is high or low.\n\n"
    "• Clinically Dangerous: A single number gives no warning if the true potency could realistically vary\n"
    "   by ±0.2 log units (modest) versus ±2.5 log units (a dangerous 300-fold drug concentration spread!)."
)
ax_b.text(0.035, 0.85, desc_b1, transform=ax_b.transAxes, fontsize=10.2, color='#334155', 
          va='top', zorder=2, linespacing=1.35)

# Card 2: 4-Way Conformal Split
rect_b2 = FancyBboxPatch((0.01, 0.02), 0.98, 0.46, transform=ax_b.transAxes,
                         boxstyle="round,pad=0.015,rounding_size=0.025",
                         facecolor='#EFF6FF', edgecolor='#BFDBFE', linewidth=1.3, zorder=1)
ax_b.add_patch(rect_b2)

ax_b.text(0.035, 0.425, "Our 4-Way Split (+ Dedicated Calibration Role)", 
          transform=ax_b.transAxes, fontsize=12.5, fontweight='bold', color='#1D4ED8', zorder=2)

desc_b2 = (
    "• Independent Calibration Role: Reserves a dedicated 10% slice strictly for conformal calibration.\n\n"
    "• Distribution-Free Error Bars: Sorts nonconformity residuals s_j = |y_j - f(x_j)| to compute radius q:\n"
    "      C_alpha(x) = [ f(x) - q,   f(x) + q ]      where      P( Y in C_alpha(X) ) >= 90%\n\n"
    "• Finite-Sample Guarantee: Statistically guarantees that true laboratory potency falls inside the\n"
    "   interval in at least 90% of unseen assays under exchangeability, without Gaussian assumptions."
)
ax_b.text(0.035, 0.365, desc_b2, transform=ax_b.transAxes, fontsize=10.2, color='#1E3A8A', 
          va='top', zorder=2, linespacing=1.35)


# -------------------------------------------------------------
# Panel C: What Each of the 4 Partitions is Used For (Bottom-Left)
# -------------------------------------------------------------
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_axis_off()
ax_c.set_title("C. What Each of the 4 Partitions is Used For (Scaffold Regime)", 
               fontsize=14, fontweight='bold', color='#1E293B', loc='left', pad=12)

# Segmented Horizontal Split Bar
bar_y = 0.92
bar_h = 0.065
splits_bar = [
    ("TRAIN (60%)", 0.01, 0.588, "#0284C7"),
    ("VAL (15%)", 0.602, 0.147, "#0D9488"),
    ("CAL (10%)", 0.753, 0.098, "#7C3AED"),
    ("TEST (15%)", 0.855, 0.135, "#059669")
]
for text_label, x_pos, width, color in splits_bar:
    bar_rect = FancyBboxPatch((x_pos, bar_y), width, bar_h, transform=ax_c.transAxes,
                              boxstyle="round,pad=0.005,rounding_size=0.012",
                              facecolor=color, edgecolor='none', zorder=3)
    ax_c.add_patch(bar_rect)
    ax_c.text(x_pos + width/2, bar_y + bar_h/2, text_label, transform=ax_c.transAxes,
              fontsize=9.2, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=4)

splits = [
    {
        "name": "1. TRAIN SET",
        "pct": "60%",
        "count": "7,937 samples",
        "color": "#0284C7",
        "bg": "#F0F9FF",
        "border": "#BAE6FD",
        "role": "Model Parameter Fitting",
        "desc": "Used strictly by learning algorithms (Ridge, ExtraTrees, Weighted ExtraTrees) to optimize decision tree splits and regression weights mapping 2,048-bit Morgan fingerprints to pIC50."
    },
    {
        "name": "2. VALIDATION SET",
        "pct": "15%",
        "count": "1,863 samples",
        "color": "#0D9488",
        "bg": "#F0FDFA",
        "border": "#99F6E4",
        "role": "Model Selection Contest",
        "desc": "Used as an unbiased tournament to compare the 4 candidate models per target. The algorithm achieving the lowest Validation MAE is selected as the winner, preventing any test set peeking."
    },
    {
        "name": "3. CALIBRATION SET",
        "pct": "10%",
        "count": "1,326 samples",
        "color": "#7C3AED",
        "bg": "#FAF5FF",
        "border": "#DDD6FE",
        "role": "Uncertainty Interval Tuning",
        "desc": "The selected best model predicts on this data. The pipeline sorts absolute residuals to compute the finite-sample quantile radius q = s_(k), guaranteeing nominal 90% prediction intervals."
    },
    {
        "name": "4. TEST SET",
        "pct": "15%",
        "count": "2,116 samples",
        "color": "#059669",
        "bg": "#ECFDF5",
        "border": "#A7F3D0",
        "role": "Final Independent Audit Vault",
        "desc": "Completely untouched until final reporting. Verifies both out-of-distribution point accuracy (pooled MAE 0.4974) and empirical interval coverage (87.4%–95.5% across targets vs nominal 90%)."
    }
]

y_c_top = 0.88
c_height = 0.20
c_gap = 0.025

for i, s in enumerate(splits):
    y = y_c_top - i * (c_height + c_gap)
    rect = FancyBboxPatch((0.01, y - c_height), 0.98, c_height, transform=ax_c.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.025",
                          facecolor=s['bg'], edgecolor=s['border'], linewidth=1.2, zorder=1)
    ax_c.add_patch(rect)
    
    # Title badge & percentage
    ax_c.text(0.035, y - 0.042, f"{s['name']} — {s['pct']} ({s['count']})", transform=ax_c.transAxes, 
              fontsize=11.2, fontweight='bold', color=s['color'], zorder=2)
    # Role
    ax_c.text(0.035, y - 0.078, f"Pipeline Function: {s['role']}", transform=ax_c.transAxes, 
              fontsize=9.4, fontweight='bold', color='#1E293B', zorder=2)
    # Description
    desc_w = textwrap.fill(s['desc'], width=72)
    ax_c.text(0.035, y - 0.115, desc_w, transform=ax_c.transAxes, fontsize=8.8, 
              color='#475569', va='top', zorder=2, linespacing=1.28)


# -------------------------------------------------------------
# Panel D: Why Can't We Just Combine Any of Them? (Bottom-Right)
# -------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_axis_off()
ax_d.set_title("D. Why Can't We Just Combine Any of Them? (The 3 Failure Modes)", 
               fontsize=14, fontweight='bold', color='#1E293B', loc='left', pad=12)

reasons = [
    {
        "q": "[RISK 1] Why NOT Calibrate on the Training Set?",
        "ans": "• Models naturally fit training samples with minimal error (residuals are artificially small).\n"
               "• If calibrated on training data, the computed conformal radius (q) will be severely underestimated.\n"
               "• Result: Real-world error bars become deceptively narrow. In deployment, biological potency will\n"
               "   fall outside the predicted interval far more frequently than the promised 90% (severe undercoverage)."
    },
    {
        "q": "[RISK 2] Why NOT Calibrate on the Validation Set?",
        "ans": "• In Section 3.5, the winning model was explicitly chosen because it achieved the lowest error on validation.\n"
               "• 'Winner's Curse' / Selection Bias: The selected model is unnaturally optimistic on the validation set.\n"
               "• Consequence: Using validation residuals violates the foundational exchangeability assumption of conformal\n"
               "   prediction theory, invalidating the mathematical proof of guaranteed coverage."
    },
    {
        "q": "[RISK 3] Why NOT Calibrate on the Test Set?",
        "ans": "• The Test Set represents the uncompromised final scientific benchmark.\n"
               "• If you calculate or tune error intervals using test samples, you have committed direct Data Leakage.\n"
               "• Result: You would have zero independent data remaining to prove to peer reviewers or clinical evaluators\n"
               "   that your calibrated intervals actually hold up on completely unseen chemicals."
    }
]

y_d_top = 0.98
d_height = 0.295
d_gap = 0.04

for i, r in enumerate(reasons):
    y = y_d_top - i * (d_height + d_gap)
    rect = FancyBboxPatch((0.01, y - d_height), 0.98, d_height, transform=ax_d.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.025",
                          facecolor='#FFFBEB', edgecolor='#FDE68A', linewidth=1.2, zorder=1)
    ax_d.add_patch(rect)
    
    # Question Header
    ax_d.text(0.035, y - 0.045, r['q'], transform=ax_d.transAxes, 
              fontsize=11.2, fontweight='bold', color='#B45309', zorder=2)
    # Explanation
    ax_d.text(0.035, y - 0.082, r['ans'], transform=ax_d.transAxes, 
              fontsize=9.2, color='#78350F', va='top', zorder=2, linespacing=1.35)

# Save to artifacts directory and docs/figures
artifact_dir = "/Users/aditisharma/.gemini/antigravity/brain/5dae2fdf-ffcf-4680-b70f-d6013fecfa19"
artifact_path = os.path.join(artifact_dir, "four_way_partitioning_and_conformal_calibration.png")
docs_path = "docs/figures/four_way_partitioning_and_conformal_calibration.png"

plt.savefig(artifact_path, bbox_inches='tight', dpi=300)
os.makedirs("docs/figures", exist_ok=True)
shutil.copyfile(artifact_path, docs_path)
print(f"Generated successfully: {artifact_path} and {docs_path}")
