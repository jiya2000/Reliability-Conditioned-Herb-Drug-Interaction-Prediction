"""
Generates interactive and publication-quality Knowledge Graph visualizations
from the biological features and entity profiles for presentations, viva, and reports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.features.biological_features import (
    HERB_BIOLOGICAL_PROFILES,
    DRUG_BIOLOGICAL_PROFILES,
    CANONICAL_TARGETS,
)

# Friendly display names
HERB_NAMES = {
    "HERB_HYPERICUM": "St. John's Wort",
    "HERB_GINKGO": "Ginkgo Biloba",
    "HERB_GINSENG": "Panax Ginseng",
    "HERB_GARLIC": "Garlic (Allium)",
    "HERB_GOLDENSEAL": "Goldenseal",
    "HERB_CURCUMA": "Turmeric (Curcumin)",
    "HERB_GREENTEA": "Green Tea (EGCG)",
    "HERB_LICORICE": "Licorice Root",
    "HERB_MILKTHISTLE": "Milk Thistle (Silymarin)",
    "HERB_VALERIAN": "Valerian Root",
    "HERB_KAVA": "Kava Kava",
    "HERB_ECHINACEA": "Echinacea",
    "HERB_GRAPEFRUIT": "Grapefruit Extract",
    "HERB_CHAMOMILE": "Chamomile",
    "HERB_PEPPERMINT": "Peppermint",
    "HERB_CRANBERRY": "Cranberry",
    "HERB_BLACKCOHOSH": "Black Cohosh",
    "HERB_SAWPALMETTO": "Saw Palmetto",
}

DRUG_NAMES = {
    "DRUG_WARFARIN": "Warfarin",
    "DRUG_DIGOXIN": "Digoxin",
    "DRUG_MIDAZOLAM": "Midazolam",
    "DRUG_CYCLOSPORINE": "Cyclosporine",
    "DRUG_THEOPHYLLINE": "Theophylline",
    "DRUG_CLOPIDOGREL": "Clopidogrel",
    "DRUG_OMEPRAZOLE": "Omeprazole",
    "DRUG_SIMVASTATIN": "Simvastatin",
    "DRUG_METFORMIN": "Metformin",
    "DRUG_LISINOPRIL": "Lisinopril",
    "DRUG_TACROLIMUS": "Tacrolimus",
    "DRUG_NADOLOL": "Nadolol",
    "DRUG_ROSUVASTATIN": "Rosuvastatin",
    "DRUG_SAQUINAVIR": "Saquinavir",
    "DRUG_INDINAVIR": "Indinavir",
    "DRUG_ALPRAZOLAM": "Alprazolam",
    "DRUG_ASPIRIN": "Aspirin",
    "DRUG_PREDNISOLONE": "Prednisolone",
    "DRUG_HYDROCHLOROTHIAZIDE": "Hydrochlorothiazide",
    "DRUG_RITONAVIR": "Ritonavir",
    "DRUG_DOCETAXEL": "Docetaxel",
    "DRUG_ETHINYLESTRADIOL": "Ethinylestradiol",
}


def build_full_networkx_graph(core_only: bool = True) -> nx.Graph:
    """Constructs the multi-entity bipartite biological knowledge graph."""
    G = nx.Graph()

    # If core_only, focus on the primary 10 benchmark herbs and 10 drugs
    core_herbs = list(HERB_NAMES.keys())[:10] if core_only else list(HERB_BIOLOGICAL_PROFILES.keys())
    core_drugs = list(DRUG_NAMES.keys())[:10] if core_only else list(DRUG_BIOLOGICAL_PROFILES.keys())

    # 1. Add Herb Nodes
    for herb_id in core_herbs:
        name = HERB_NAMES.get(herb_id, herb_id.replace("HERB_", "").title())
        G.add_node(name, node_type="herb", id=herb_id, color="#10b981", size=24)

    # 2. Add Target Nodes
    for target in CANONICAL_TARGETS:
        G.add_node(target, node_type="target", id=target, color="#f59e0b", size=18)

    # 3. Add Drug Nodes
    for drug_id in core_drugs:
        name = DRUG_NAMES.get(drug_id, drug_id.replace("DRUG_", "").title())
        G.add_node(name, node_type="drug", id=drug_id, color="#3b82f6", size=24)

    # 4. Add Herb -> Target Edges
    for herb_id in core_herbs:
        name = HERB_NAMES.get(herb_id, herb_id.replace("HERB_", "").title())
        profile = HERB_BIOLOGICAL_PROFILES.get(herb_id, {})
        for target, val in profile.items():
            if target in CANONICAL_TARGETS and abs(val) > 0.05:
                role = "Induction" if val > 0 else "Inhibition"
                G.add_edge(name, target, weight=abs(val), val=val, role=f"Herb {role} ({val:+.2f})", edge_type="herb_target")

    # 5. Add Drug -> Target Edges
    for drug_id in core_drugs:
        name = DRUG_NAMES.get(drug_id, drug_id.replace("DRUG_", "").title())
        profile = DRUG_BIOLOGICAL_PROFILES.get(drug_id, {})
        for target, val in profile.items():
            if target in CANONICAL_TARGETS and abs(val) > 0.05:
                G.add_edge(name, target, weight=abs(val), val=val, role=f"Drug Substrate/Target ({val:+.2f})", edge_type="drug_target")

    # Prune isolated targets if any
    isolated = [n for n in G.nodes() if G.degree[n] == 0]
    for n in isolated:
        G.remove_node(n)

    return G


def generate_interactive_plotly_html(output_path: str):
    """Generates a rich, interactive 2D HTML network graph with zoom/pan/hover."""
    G = build_full_networkx_graph(core_only=True)
    pos = nx.spring_layout(G, seed=42, k=0.55, iterations=80)

    herb_edge_x, herb_edge_y = [], []
    drug_edge_x, drug_edge_y = [], []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if data.get("edge_type") == "herb_target":
            herb_edge_x.extend([x0, x1, None])
            herb_edge_y.extend([y0, y1, None])
        else:
            drug_edge_x.extend([x0, x1, None])
            drug_edge_y.extend([y0, y1, None])

    trace_herb_edges = go.Scatter(
        x=herb_edge_x, y=herb_edge_y,
        line=dict(width=1.8, color="rgba(16, 185, 129, 0.45)"),
        hoverinfo="none",
        mode="lines",
        name="Herb-Target Regulatory Link"
    )

    trace_drug_edges = go.Scatter(
        x=drug_edge_x, y=drug_edge_y,
        line=dict(width=1.8, color="rgba(59, 130, 246, 0.45)"),
        hoverinfo="none",
        mode="lines",
        name="Drug-Target Clearance Link"
    )

    node_types = ["herb", "target", "drug"]
    type_labels = {
        "herb": "Botanical Entities (Herbs)",
        "target": "Human Biological Targets (CYPs / Transporters)",
        "drug": "Prescription Pharmaceuticals (Drugs)"
    }
    type_colors = {"herb": "#10b981", "target": "#f59e0b", "drug": "#3b82f6"}
    type_sizes = {"herb": 26, "target": 20, "drug": 26}

    node_traces = []
    for ntype in node_types:
        nx_list, ny_list, ntext_list, nhover_list = [], [], [], []
        for n, data in G.nodes(data=True):
            if data.get("node_type") == ntype:
                x, y = pos[n]
                nx_list.append(x)
                ny_list.append(y)
                ntext_list.append(n)
                degree = G.degree[n]
                connected = [f"{v} ({G[n][v].get('role', '')})" for v in G.neighbors(n)]
                conn_preview = "<br> • ".join(connected[:6])
                if len(connected) > 6:
                    conn_preview += f"<br> ... (+{len(connected)-6} more)"
                hover_str = (
                    f"<b>{n}</b> [{ntype.upper()}]<br>"
                    f"Connected Nodes ({degree}):<br> • {conn_preview}"
                )
                nhover_list.append(hover_str)

        node_traces.append(go.Scatter(
            x=nx_list, y=ny_list,
            mode="markers+text",
            text=ntext_list,
            textposition="top center",
            textfont=dict(color="#f8fafc", size=10, family="sans-serif"),
            hovertext=nhover_list,
            hoverinfo="text",
            marker=dict(
                color=type_colors[ntype],
                size=type_sizes[ntype],
                line=dict(width=2, color="#ffffff")
            ),
            name=type_labels[ntype]
        ))

    fig = go.Figure(data=[trace_herb_edges, trace_drug_edges, *node_traces])
    fig.update_layout(
        title={
            "text": "<b>Heterogeneous Biological Knowledge Graph</b><br><span style='font-size: 13px; color: #94a3b8;'>Tri-partite Network: Botanical Sources ↔ Pharmacokinetic Targets (CYP Enzymes / Transporters) ↔ Prescription Drugs</span>",
            "x": 0.05,
            "y": 0.95
        },
        template="plotly_dark",
        paper_bgcolor="#0b1329",
        plot_bgcolor="#0b1329",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#e2e8f0")
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=90, b=20),
        height=850,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"Saved interactive HTML to {output_path}")


def generate_static_png(output_path: str):
    """Generates a publication-quality static PNG visualization."""
    G = build_full_networkx_graph(core_only=True)
    plt.figure(figsize=(16, 12), facecolor="#0f172a")
    ax = plt.gca()
    ax.set_facecolor("#0f172a")

    pos = nx.spring_layout(G, seed=42, k=0.55, iterations=80)

    herb_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "herb_target"]
    drug_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "drug_target"]

    nx.draw_networkx_edges(G, pos, edgelist=herb_edges, edge_color="#10b981", alpha=0.4, width=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=drug_edges, edge_color="#3b82f6", alpha=0.4, width=1.5, ax=ax)

    herbs = [n for n, d in G.nodes(data=True) if d.get("node_type") == "herb"]
    targets = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]
    drugs = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]

    nx.draw_networkx_nodes(G, pos, nodelist=herbs, node_color="#10b981", node_size=650, edgecolors="white", linewidths=1.5, label="Botanicals (Herbs)", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=targets, node_color="#f59e0b", node_size=450, edgecolors="white", linewidths=1.5, label="Targets (CYP/Transporters)", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=drugs, node_color="#3b82f6", node_size=650, edgecolors="white", linewidths=1.5, label="Prescription Drugs", ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=9, font_color="#f8fafc", font_weight="bold", ax=ax)

    plt.title(
        "Heterogeneous Biological Knowledge Graph Architecture (GNN Substrate)\n10 Botanical Entities | 14 Active CYP & Transporter Targets | 10 Prescription Drugs",
        color="#f8fafc",
        fontsize=14,
        pad=20,
        fontweight="bold"
    )
    plt.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=11, loc="upper right")
    plt.axis("off")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication PNG to {output_path}")


if __name__ == "__main__":
    generate_interactive_plotly_html("reports/figures/interactive_knowledge_graph.html")
    generate_static_png("reports/figures/biological_knowledge_graph.png")
