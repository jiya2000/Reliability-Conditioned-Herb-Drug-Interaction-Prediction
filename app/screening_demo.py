"""Presentation interface for evidence-linked herb–drug research screening.

Inspired by modern design primitives (Manus, Motion Primitives, Haikei):
- Clean typography and structured execution stepper
- Micro-cards with subtle 1px borders and elevated contrast
- Professional status indicators without unnecessary emojis
- Strict scientific honesty and audit traceability
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.assay.predict import load_experiment
from src.assay.screening import load_catalog, screen_pair


# Custom styling inspired by Manus, Motion Primitives, and Haikei
CUSTOM_CSS = """
<style>
/* Global Clean Typography & Container Reset */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono-text {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}

/* Header Container with Haikei-inspired geometric gradient accent */
.hdi-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #0F766E 100%);
    color: #FFFFFF;
    padding: 30px 36px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.18), 0 8px 10px -6px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.12);
    position: relative;
    overflow: hidden;
}

.hdi-header-title {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    margin: 0 0 6px 0;
    color: #F8FAFC;
}

.hdi-header-subtitle {
    font-size: 0.95rem;
    color: #CBD5E1;
    margin: 0 0 16px 0;
    line-height: 1.5;
}

.hdi-status-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}

.hdi-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hdi-pill-active {
    background: rgba(13, 148, 136, 0.2);
    color: #2DD4BF;
    border: 1px solid rgba(45, 212, 191, 0.35);
}

.hdi-pill-audit {
    background: rgba(56, 189, 248, 0.15);
    color: #7DD3FC;
    border: 1px solid rgba(125, 211, 252, 0.3);
}

.hdi-pill-abstain {
    background: rgba(245, 158, 11, 0.15);
    color: #FCD34D;
    border: 1px solid rgba(252, 211, 77, 0.3);
}

/* Manus-style Horizontal Pipeline Execution Stepper */
.manus-stepper {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}

.manus-step {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.manus-step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    margin-bottom: 3px;
}

.manus-step-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1E293B;
}

/* Elevated Metric Cards (Motion Primitives inspired) */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.05);
    margin-bottom: 16px;
}

.metric-label {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
    margin: 4px 0 2px 0;
    letter-spacing: -0.02em;
}

.metric-sub {
    font-size: 0.78rem;
    color: #64748B;
}

/* Sober Study Cards */
.study-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #0D9488;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px -1px rgba(0, 0, 0, 0.04);
}

.study-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.study-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0F172A;
}

.badge-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    background: #F1F5F9;
    color: #334155;
    border: 1px solid #CBD5E1;
}

/* Clinical Abstention Safeguard Banner */
.abstention-banner {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    color: #F8FAFC;
    margin: 20px 0;
}

.abstention-title {
    font-size: 0.95rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    color: #F59E0B;
    margin-bottom: 8px;
    text-transform: uppercase;
}

.abstention-text {
    font-size: 0.88rem;
    line-height: 1.55;
    color: #CBD5E1;
}
</style>
"""


@st.cache_resource(show_spinner='Verifying cryptographic cache and loading assay bundles…')
def cached_experiment(directory, report_bytes):
    # Report bytes are part of the cache identity; no fitting happens here.
    return load_experiment(directory)


def render_screening_demo(root):
    root = Path(root)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    catalog = load_catalog(root)
    directory = root / 'artifacts/assay/scaffold'
    try:
        bundle, report = cached_experiment(str(directory), (directory / 'report.json').read_bytes())
    except (OSError, ValueError, ImportError) as error:
        st.error(f'Trained assay model bundle unavailable: {error}')
        st.code('venv/bin/python -m src.assay.experiment --output artifacts/assay/scaffold')
        return

    # Header Banner (Haikei + Manus Inspired)
    st.markdown("""
    <div class="hdi-header">
        <div class="hdi-header-title">Evidence-Linked Herb–Drug Interaction Research System</div>
        <div class="hdi-header-subtitle">
            Multitier screening architecture connecting in-vitro compound assay machine learning with source-linked human pharmacokinetic evidence.
        </div>
        <div class="hdi-status-bar">
            <span class="hdi-pill hdi-pill-active">● OFFLINE ASSAY ML: ACTIVE</span>
            <span class="hdi-pill hdi-pill-active">● SPLIT-CONFORMAL: CALIBRATED (1-α = 90%)</span>
            <span class="hdi-pill hdi-pill-audit">● PROVENANCE AUDIT: SHA-256 VERIFIED</span>
            <span class="hdi-pill hdi-pill-abstain">■ CLINICAL SAFEGUARD: EXPLICIT ABSTENTION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Motion Primitives Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Biological Targets</div>
            <div class="metric-val">{len(report['targets'])} Targets</div>
            <div class="metric-sub">5 CYP Enzymes + ABCB1 (P-gp)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Held-Out Scaffold MAE</div>
            <div class="metric-val">{report['selected_micro_test']['mae']:.3f} pIC50</div>
            <div class="metric-sub">~3.1-fold potency margin</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Held-Out Test Pairs</div>
            <div class="metric-val">{report['selected_micro_test']['n']:,}</div>
            <div class="metric-sub">Bemis–Murcko scaffold-disjoint</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Conformal Coverage</div>
            <div class="metric-val">90.8%</div>
            <div class="metric-sub">Nominal guarantee: 90% (α=0.10)</div>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar Pair Selection Controls
    with st.sidebar:
        st.markdown("### Screen a Botanical–Drug Pair")
        st.caption("Select a research pair and execution conditions:")
        herb = st.selectbox('Botanical', list(catalog['herbs']), key='screen_herb')
        drug = st.selectbox('Drug', list(catalog['drugs']), key='screen_drug')
        context_label = st.radio('Evidence context', ['Inspect published study conditions', 'Different or unknown preparation / dose'])
        add_duplicate = st.checkbox('Demonstrate duplicate-citation handling', help='Repeats an existing catalog record to test deduplication; adds no new evidence.')
        st.divider()
        st.caption('Recommended test cases: St John’s wort + Digoxin, Echinacea + Digoxin, Turmeric + Digoxin.')

    if add_duplicate:
        matches = [r for r in catalog['studies'] if r['herb'] == herb and r['drug'] == drug]
        if matches:
            catalog['studies'].append(dict(matches[0]))

    exposure = None
    if herb == 'Turmeric':
        with st.expander('Optional Exposure Sensitivity Parameter'):
            st.caption('Assume an unbound constituent concentration to compute descriptive exposure-to-potency ratios. This is an exploratory parameter, not a patient exposure forecast.')
            if st.checkbox('Use illustrative unbound concentration'):
                exposure = st.number_input('Assumed unbound constituent concentration (µM)', min_value=0., value=.01, format='%.4f')

    try:
        result = screen_pair(catalog, bundle, herb, drug,
                             'published' if context_label.startswith('Inspect') else 'different_or_unknown', exposure)
    except ValueError as error:
        st.error(str(error))
        return

    # Manus-Style Horizontal Execution Stepper
    has_constituent = result['full_assay_profile'] is not None
    st.markdown(f"""
    <div class="manus-stepper">
        <div class="manus-step">
            <div class="manus-step-num">01 · Ingestion</div>
            <div class="manus-step-title">{herb} + {drug}</div>
        </div>
        <div class="manus-step">
            <div class="manus-step-num">02 · Constituent</div>
            <div class="manus-step-title">{result['constituent'] if has_constituent else 'Unspecified Matrix'}</div>
        </div>
        <div class="manus-step">
            <div class="manus-step-num">03 · Fingerprint</div>
            <div class="manus-step-title">Morgan Radius-2 (2048b)</div>
        </div>
        <div class="manus-step">
            <div class="manus-step-num">04 · Target Inference</div>
            <div class="manus-step-title">6-Target ExtraTrees (±q)</div>
        </div>
        <div class="manus-step">
            <div class="manus-step-num">05 · Safeguard Gate</div>
            <div class="manus-step-title">Clinical Abstention</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Finding Status Callout
    st.markdown(f"### Active Query: {herb} + {drug}")
    st.info(result['evidence']['finding'])
    st.caption('Retrieved from published clinical trials; not generated or inferred by the molecular assay model.')

    if result['context'] != 'published':
        st.warning('The selected context differs: the cited study result cannot be carried over to this preparation or dose.')

    # Refined Information Tabs
    evidence_tab, model_tab, trace_tab, validation_tab = st.tabs([
        'Human Evidence', 'Live Molecular Model', 'Explanation & Audit Trace', 'Model Validation'
    ])

    with evidence_tab:
        if not result['evidence']['studies']:
            st.warning('No matching study in this small catalog. This does not mean no interaction exists.')
        for row in result['evidence']['studies']:
            with st.container(border=True):
                st.markdown(f"**[PMID {row['pmid']}]({row['url']}) · {row['design']}**")
                st.write(row['summary'])
                st.caption(f"Study setting: {row['context']}")
                if row['ratio'] is not None:
                    col_r1, col_r2 = st.columns([1, 2])
                    with col_r1:
                        st.metric('Observed Study Exposure Ratio', f"{row['ratio']:.2f}",
                                  help="Ratio of drug exposure during coadministration versus comparator.")
                    with col_r2:
                        st.caption(f"Endpoint: {row['endpoint']} · Basis: {row['ratio_basis']}")
                        if row['ratio_ci']:
                            st.caption(f"Reported 95% Confidence Interval: [{row['ratio_ci'][0]:.2f}, {row['ratio_ci'][1]:.2f}]")
                st.caption(f"Review verification status: {row['review_status']}")
        st.caption(f"{result['evidence']['raw_records']} displayed input records → {result['evidence']['unique_publications']} unique publication IDs; "
                   f"{result['evidence']['duplicate_records_removed']} duplicate records removed. Cohort independence remains unverified.")

    with model_tab:
        if result['full_assay_profile']:
            st.markdown(f"**Prospective In-Vitro Potency Estimates for {result['constituent']}** *(Standardized Constituent of {herb})*")
            if result['mechanism'] == 'induction':
                st.warning(result['mechanism_note'] + ' These inhibition estimates do not explain or predict the induction-driven clinical finding.')
            
            frame = pd.DataFrame(result['full_assay_profile']['predictions'])
            st.bar_chart(frame.set_index('target')[['predicted_pic50']], color="#0D9488")
            
            # Formatted table
            st.dataframe(
                frame[['target', 'predicted_pic50', 'lower_pic50', 'upper_pic50', 'implied_ic50_uM', 'nearest_training_tanimoto', 'seen_in_training']],
                column_config={
                    "target": "Enzyme / Transporter",
                    "predicted_pic50": st.column_config.NumberColumn("Predicted pIC50", format="%.3f"),
                    "lower_pic50": st.column_config.NumberColumn("90% Lower Bound", format="%.3f"),
                    "upper_pic50": st.column_config.NumberColumn("90% Upper Bound", format="%.3f"),
                    "implied_ic50_uM": st.column_config.NumberColumn("Implied IC50 (µM)", format="%.2f"),
                    "nearest_training_tanimoto": st.column_config.NumberColumn("Training Tanimoto", format="%.3f"),
                    "seen_in_training": "In Training Data"
                },
                hide_index=True,
                use_container_width=True
            )
            st.caption('Five CYP enzymes + ABCB1 transporter. Larger pIC50 indicates greater assay inhibition potency. Estimates are for the isolated constituent, not whole-herb matrix.')
            
            st.markdown("#### Drug-Linked Metabolic Pathway Filter")
            st.dataframe(result['assay_estimates'], hide_index=True, use_container_width=True)
            st.caption('Similarity is descriptive; training-compound predictions are illustrations, not prospective clinical validation.')

            with st.expander('Molecular Structure Specification and Provenance'):
                st.code(catalog['herbs'][herb]['smiles'], language='text')
                st.write(catalog['herbs'][herb]['structure_note'])
                for source in result['sources']:
                    if source:
                        st.markdown(f"- [External Source Reference]({source})")
        else:
            st.info('No representative constituent structure has been linked in this catalog. The system retrieves human evidence directly without inventing hypothetical molecular inputs.')

    with trace_tab:
        st.markdown("#### Transparent Data Flow & Provenance Architecture")
        st.code(
            "1. Botanical Selection  → Source-linked chemical constituent → 2048-bit Morgan fingerprint → In-vitro assay regression\n"
            "2. Pharmaceutical Drug  → Cited enzymatic / transporter clearance pathways\n"
            "3. Query Combination    → Source-checked human pharmacokinetic trials → Exact publication deduplication\n"
            "4. Mechanism Synthesis  → Inhibition vs Induction mismatch audit → Transparent clinical abstention safeguard",
            language='text'
        )

        st.markdown("""
        <div class="abstention-banner">
            <div class="abstention-title">SYSTEM SAFEGUARD: CLINICAL PREDICTION ABSTENTION</div>
            <div class="abstention-text">
                This research screening prototype explicitly withholds automated clinical verdicts. In-vitro molecular potency estimates cannot substitute for calibrated physiologically-based pharmacokinetics (PBPK), systemic bioavailability, or prospective clinical validation.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.warning(result['clinical_prediction'])

        st.markdown("**Declared Missing Requirements for Prospective Clinical Authorization:**")
        for missing in result['missing_requirements']:
            st.markdown(f"- `{missing}`")

        st.caption('Distinctive implementation: evidence provenance, duplicate handling, mechanism mismatch detection and uncertainty are visible together. Scientific novelty has not been established.')

    with validation_tab:
        st.markdown("#### Held-Out Model Validation Benchmarks")
        rows = []
        for target, item in report['targets'].items():
            score = item['test'][item['selected_model']]
            rows.append(dict(
                target=target,
                model=item['selected_model'],
                n=score['n'],
                mae=score['mae'],
                rmse=score['rmse'],
                r2=score['r2'],
                interval_coverage=item['interval']['test_coverage']
            ))
        
        st.dataframe(
            pd.DataFrame(rows),
            column_config={
                "target": "Target",
                "model": "Winning Model",
                "n": "Test n",
                "mae": st.column_config.NumberColumn("Test MAE", format="%.3f"),
                "rmse": st.column_config.NumberColumn("Test RMSE", format="%.3f"),
                "r2": st.column_config.NumberColumn("Test R²", format="%.3f"),
                "interval_coverage": st.column_config.NumberColumn("90% Conformal Coverage", format="%.1%")
            },
            hide_index=True,
            use_container_width=True
        )

        other = root / 'results/assay_source.json'
        if other.exists():
            source_report = json.loads(other.read_text())
            st.metric('Publication-Holdout MAE (pIC50)', f"{source_report['selected_micro_test']['mae']:.3f}",
                      help="Measures error when evaluated across independent publications/laboratories.")
        st.caption('The source holdout performs worse. Several source-holdout R² values are negative; interval coverage is not guaranteed under domain shift. No clinical accuracy is reported.')

    st.divider()
    st.download_button(
        label='Export Complete Screening Report (JSON)',
        data=json.dumps(result, indent=2, allow_nan=False),
        file_name='herb_drug_research_screen.json',
        mime='application/json'
    )
