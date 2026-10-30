import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
import shutil
import textwrap

# Set global styles
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

fig = plt.figure(figsize=(20, 13), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

# Header
fig.text(0.04, 0.965, "MODEL BENCHMARK, ARCHITECTURAL RATIONALE & VALIDATION SELECTION", 
         fontsize=18, fontweight='bold', color='#0F172A')
fig.text(0.04, 0.942, "Evaluation across 6 FDA-mandated metabolic targets under strict Bemis–Murcko scaffold holdout (2,116 held-out test samples)", 
         fontsize=11.5, color='#475569')

gs = gridspec.GridSpec(2, 2, figure=fig, top=0.89, bottom=0.06, left=0.04, right=0.96, 
                       hspace=0.28, wspace=0.18)

# -------------------------------------------------------------
# Panel A: Why Only These 4 Models? (Top-Left)
# -------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_axis_off()
ax_a.set_title("A. Architectural Rationale: Why These 4 Models?", fontsize=13.5, fontweight='bold', color='#1E293B', loc='left', pad=12)

cards = [
    {
        "title": "1. Dummy Median Regressor (Naive Baseline)",
        "color": "#475569",
        "bg": "#F8FAFC",
        "border": "#CBD5E1",
        "role": "Sanity Check / Zero-Information Reference",
        "desc": "Predicts the median pIC50 of training assays. Any viable ML model MUST decisively beat this naive guesser. If an algorithm fails to outperform the median, it has captured no meaningful structure–activity relationship."
    },
    {
        "title": "2. Ridge Linear Regression (Linear Benchmark)",
        "color": "#0284C7",
        "bg": "#F0F9FF",
        "border": "#BAE6FD",
        "role": "Parametric Additive Benchmark (L2 Regularized, α=100, LSQR)",
        "desc": "Assumes each bit of the 2,048-bit Morgan fingerprint contributes additively and independently to potency. Solved with LSQR. Establishes how much predictive accuracy is achievable through pure linear feature contributions."
    },
    {
        "title": "3. Extra-Trees Regressor (Non-Linear Workhorse)",
        "color": "#0D9488",
        "bg": "#F0FDFA",
        "border": "#99F6E4",
        "role": "Non-Parametric Ensemble (128 trees, min_leaf=2, max_features=0.5)",
        "desc": "Extremely Randomized Trees draw random split thresholds across feature subsets. This heavily suppresses overfitting on sparse 2,048-bit binary fingerprints and captures non-linear pharmacophoric interactions."
    },
    {
        "title": "4. Consistency-Weighted Extra-Trees (Quality Ablation)",
        "color": "#7C3AED",
        "bg": "#FAF5FF",
        "border": "#DDD6FE",
        "role": "Noise-Conditioned Model (Sample weight = 1 / (1 + MAD))",
        "desc": "Repeated literature assays exhibit inter-lab dispersion. This variant down-weights compounds with high experimental conflict across papers, testing whether conditioning on assay reproducibility enhances true generalization."
    }
]

y_top = 0.98
card_height = 0.22
card_gap = 0.03

for i, c in enumerate(cards):
    y = y_top - i * (card_height + card_gap)
    # Background Box
    rect = FancyBboxPatch((0.01, y - card_height), 0.98, card_height, transform=ax_a.transAxes,
                          boxstyle="round,pad=0.015,rounding_size=0.025",
                          facecolor=c['bg'], edgecolor=c['border'], linewidth=1.2, zorder=1)
    ax_a.add_patch(rect)
    
    # Title badge
    ax_a.text(0.035, y - 0.045, c['title'], transform=ax_a.transAxes, fontsize=10.5, 
              fontweight='bold', color=c['color'], zorder=2)
    # Role subtitle
    ax_a.text(0.035, y - 0.082, f"Role: {c['role']}", transform=ax_a.transAxes, fontsize=8.8, 
              fontstyle='italic', color='#334155', zorder=2)
    # Description wrapped cleanly
    wrapped_desc = textwrap.fill(c['desc'], width=76)
    ax_a.text(0.035, y - 0.12, wrapped_desc, transform=ax_a.transAxes, fontsize=8.2, 
              color='#475569', va='top', zorder=2, linespacing=1.25)

# -------------------------------------------------------------
# Panel B: How Model Selection Was Done (Top-Right)
# -------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_title("B. Model Selection Contest: Validation MAE Across 6 Targets", fontsize=13.5, fontweight='bold', color='#1E293B', loc='left', pad=12)

targets = ['ABCB1', 'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4']
val_median = [0.7984, 0.7465, 0.4959, 0.5375, 0.6455, 0.7247]
val_ridge = [0.7284, 0.6232, 0.5060, 0.4826, 0.5418, 0.5833]
val_et = [0.6102, 0.5839, 0.4903, 0.4691, 0.5409, 0.5267]
val_et_w = [0.6115, 0.5946, 0.4948, 0.4751, 0.5403, 0.5246]

x = np.arange(len(targets))
width = 0.19

b1 = ax_b.bar(x - 1.5*width, val_median, width, label='Median Baseline', color='#94A3B8', edgecolor='#64748B', linewidth=0.8)
b2 = ax_b.bar(x - 0.5*width, val_ridge, width, label='Ridge Regression', color='#38BDF8', edgecolor='#0284C7', linewidth=0.8)
b3 = ax_b.bar(x + 0.5*width, val_et, width, label='Extra-Trees (Unweighted)', color='#14B8A6', edgecolor='#0F766E', linewidth=0.8)
b4 = ax_b.bar(x + 1.5*width, val_et_w, width, label='Consistency-Weighted ET', color='#A855F7', edgecolor='#7E22CE', linewidth=0.8)

# Add winning stars/callouts
win_y = [0.6102, 0.5839, 0.4903, 0.4691, 0.5403, 0.5246]
for i in range(len(targets)):
    pos_x = x[i] + 0.5*width if i < 4 else x[i] + 1.5*width
    ax_b.scatter(pos_x, win_y[i] + 0.035, marker='*', s=150, color='#EAB308', edgecolor='#A16207', zorder=5)
    ax_b.text(pos_x, win_y[i] + 0.055, f"{win_y[i]:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1E293B')

ax_b.set_ylabel("Validation MAE in pIC50 (Lower is Better)", fontsize=10.5, color='#334155')
ax_b.set_xticks(x)
ax_b.set_xticklabels(targets, fontsize=10.5, fontweight='bold', color='#1E293B')
ax_b.set_ylim(0, 0.95)
ax_b.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
ax_b.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=8.8, loc='upper right')

# Selection rule callout
ax_b.text(0.03, 0.93, "Selection Rule: min(Validation MAE) before evaluating Calibration or Test", 
          transform=ax_b.transAxes, fontsize=8.8, fontweight='bold', 
          bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF3C7', edgecolor='#F59E0B', alpha=0.95))

# -------------------------------------------------------------
# Panel C: Target-by-Target Selected Model Performance (Bottom-Left)
# -------------------------------------------------------------
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_axis_off()
ax_c.set_title("C. Selected Model Specifications & Held-Out Test Evaluation", fontsize=13.5, fontweight='bold', color='#1E293B', loc='left', pad=12)

table_data = [
    ["Target", "Selected Model", "Why Selected (Validation Contest)", "Test n", "Test MAE", "Test RMSE", "Test R²", "90% Interval Radius", "Test Coverage"],
    ["ABCB1 (P-gp)", "ExtraTrees", "Lowest Val MAE (0.6102 vs 0.7284 Ridge)", "155", "0.495", "0.658", "0.507", "± 1.348 pIC50", "95.5% (Nom: 90%)"],
    ["CYP1A2", "ExtraTrees", "Lowest Val MAE (0.5839 vs 0.6232 Ridge)", "251", "0.584", "0.756", "0.221", "± 1.457 pIC50", "93.6% (Nom: 90%)"],
    ["CYP2C19", "ExtraTrees", "Lowest Val MAE (0.4903 vs 0.5060 Ridge)", "217", "0.474", "0.600", "0.129", "± 1.088 pIC50", "93.5% (Nom: 90%)"],
    ["CYP2C9", "ExtraTrees", "Lowest Val MAE (0.4691 vs 0.4826 Ridge)", "358", "0.468", "0.607", "0.242", "± 0.953 pIC50", "87.4% (Nom: 90%)"],
    ["CYP2D6", "Weighted ExtraTrees", "Lowest Val MAE (0.5403 vs 0.5409 unweighted)", "375", "0.513", "0.666", "0.322", "± 1.276 pIC50", "94.4% (Nom: 90%)"],
    ["CYP3A4", "Weighted ExtraTrees", "Lowest Val MAE (0.5246 vs 0.5267 unweighted)", "760", "0.482", "0.641", "0.350", "± 1.011 pIC50", "88.0% (Nom: 90%)"],
    ["POOLED TOTAL", "Selected Ensemble", "Optimal Architecture per Target", "2,116", "0.497", "0.652", "0.344", "± 1.156 (mean)", "90.8% (Target 90%)"]
]

col_widths = [0.13, 0.17, 0.26, 0.065, 0.08, 0.08, 0.065, 0.15, 0.14]

tab = ax_c.table(cellText=table_data, loc='center', cellLoc='center', colWidths=col_widths)
tab.auto_set_font_size(False)
tab.set_fontsize(8.2)
tab.scale(1.0, 1.85)

for (r, c_idx), cell in tab.get_celld().items():
    cell.set_edgecolor('#CBD5E1')
    if r == 0:
        cell.set_facecolor('#1E293B')
        cell.get_text().set_color('#FFFFFF')
        cell.get_text().set_fontweight('bold')
    elif r == len(table_data) - 1:
        cell.set_facecolor('#E2E8F0')
        cell.get_text().set_color('#0F172A')
        cell.get_text().set_fontweight('bold')
    elif r % 2 == 1:
        cell.set_facecolor('#FFFFFF')
    else:
        cell.set_facecolor('#F8FAFC')
        
    if c_idx == 1 and r > 0 and r < len(table_data) - 1:
        cell.get_text().set_fontweight('bold')
        if "Weighted" in table_data[r][c_idx]:
            cell.get_text().set_color('#7C3AED')
        else:
            cell.get_text().set_color('#0D9488')
    elif c_idx == 2 and r > 0:
        cell.get_text().set_fontsize(7.8)

# -------------------------------------------------------------
# Panel D: Overall Pooled Model Comparison (Bottom-Right)
# -------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_title("D. Comparative Performance: Pooled Test MAE across Model Families", fontsize=13.5, fontweight='bold', color='#1E293B', loc='left', pad=12)

models_eval = [
    'Training Median\n(Naive Baseline)', 
    'Ridge Linear\nRegression', 
    'Extra-Trees\n(Unweighted)', 
    'Consistency-Weighted\nExtra-Trees', 
    'Selected Ensemble\n(Winner Combination)'
]
mae_scaffold = [0.6212, 0.5235, 0.4981, 0.4995, 0.4974]
mae_source = [0.6875, 0.6398, 0.6595, 0.6611, 0.6613]

x_m = np.arange(len(models_eval))
w_m = 0.35

bar_s = ax_d.bar(x_m - w_m/2, mae_scaffold, w_m, label='Scaffold Holdout (Primary Benchmark)', 
                 color=['#94A3B8', '#38BDF8', '#14B8A6', '#A855F7', '#059669'], 
                 edgecolor='#0F172A', linewidth=0.9)
bar_p = ax_d.bar(x_m + w_m/2, mae_source, w_m, label='Publication Holdout (Sensitivity Test)', 
                 color=['#E2E8F0', '#BAE6FD', '#CCFBF1', '#F3E8FF', '#D1FAE5'], 
                 edgecolor='#64748B', linewidth=0.7, hatch='//')

# Annotate Scaffold values
for bar, val in zip(bar_s, mae_scaffold):
    ax_d.text(bar.get_x() + bar.get_width()/2, val + 0.012, f"{val:.4f}", 
              ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0F172A')

ax_d.set_ylabel("Pooled Mean Absolute Error (MAE in pIC50)", fontsize=10.5, color='#334155')
ax_d.set_xticks(x_m)
ax_d.set_xticklabels(models_eval, fontsize=8.8, color='#1E293B')
ax_d.set_ylim(0, 0.82)
ax_d.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
ax_d.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=8.8, loc='upper right')

# Annotation for selected ensemble advantage
ax_d.annotate('Selected Models:\nLowest MAE (0.4974)\n~3.1-fold error', 
            xy=(4 - w_m/2, 0.4974), xytext=(3.15, 0.30),
            arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.8, headwidth=6),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#D1FAE5', edgecolor='#059669', alpha=0.95),
            fontsize=8.8, fontweight='bold', color='#065F46')

# Save to artifacts directory and docs/figures
artifact_dir = "/Users/aditisharma/.gemini/antigravity/brain/5dae2fdf-ffcf-4680-b70f-d6013fecfa19"
artifact_path = os.path.join(artifact_dir, "model_selection_and_comparative_benchmark.png")
docs_path = "docs/figures/model_selection_and_comparative_benchmark.png"

plt.savefig(artifact_path, bbox_inches='tight', dpi=300)
os.makedirs("docs/figures", exist_ok=True)
shutil.copyfile(artifact_path, docs_path)
print(f"Generated successfully: {artifact_path} and {docs_path}")
