"""
Streamlit Demo Frontend for HDI Prediction
Interactive UI for querying herb-drug interactions with a premium, pro-level design.
"""

import streamlit as st
import requests
import json
import time

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="HDI Prediction Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Advanced Custom CSS ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Default Streamlit UI */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Hero Gradient Text */
    .hero-text {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    /* Subtitle */
    .subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0px;
        margin-bottom: 2rem;
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 1.2rem;
    }
    .risk-high { background: rgba(255, 75, 75, 0.15); color: #ff4b4b; border: 1px solid #ff4b4b; }
    .risk-moderate { background: rgba(255, 167, 38, 0.15); color: #ffa726; border: 1px solid #ffa726; }
    .risk-low { background: rgba(102, 187, 106, 0.15); color: #66bb6a; border: 1px solid #66bb6a; }
    .risk-minimal { background: rgba(66, 165, 245, 0.15); color: #42a5f5; border: 1px solid #42a5f5; }
    
    /* Progress bars styling */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 242, 254, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API endpoint
API_URL = "http://localhost:8000"

# ---------- Fetch Known Entities ----------
@st.cache_data(ttl=60)
def fetch_entities():
    try:
        response = requests.get(f"{API_URL}/known_entities", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    # Fallback if API is down
    return {
        "drugs": [
            "Warfarin", "Metformin", "Digoxin", "Cyclosporine",
            "Simvastatin", "Phenytoin", "Carbamazepine",
        ],
        "herbs": [
            "Ashwagandha", "Turmeric", "St. John's Wort",
            "Ginkgo", "Garlic", "Ginger",
        ],
    }

entities = fetch_entities()
DRUGS = sorted(entities.get("drugs", []))
HERBS = sorted(entities.get("herbs", []))

if not DRUGS:
    DRUGS = ["Warfarin"]
if not HERBS:
    HERBS = ["St. John's Wort"]

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965906.png", width=60)
    st.markdown("### Model Configuration")
    api_url = st.text_input("Backend Endpoint", value=API_URL, help="URL of the FastAPI Inference Server")
    
    st.markdown("---")
    st.markdown("### Inference Parameters")
    drug = st.selectbox("💊 Select Target Drug", DRUGS, index=0 if "Warfarin" not in DRUGS else DRUGS.index("Warfarin"))
    herb = st.selectbox("🌿 Select Interacting Herb", HERBS, index=0)
    
    st.markdown("---")
    st.markdown("### About This System")
    st.markdown(
        """
        <div style="font-size: 0.9em; color: #a0aec0;">
        Powered by a Reliability-Conditioned Graph Neural Network.
        Validates herb-drug interactions using actual structural graphs from the 
        <b>DrugBank Knowledge Graph</b>.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    predict_btn = st.button("🚀 Analyze Interaction", use_container_width=True)

# ---------- Main Content Area ----------
st.markdown('<p class="hero-text">HDI Graph Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Interaction Discovery & Evidence Reliability Scoring</p>', unsafe_allow_html=True)

# If button is clicked
if predict_btn:
    with st.spinner("Traversing Knowledge Graph & Calculating Attention..."):
        time.sleep(0.5) # Slight delay for dramatic effect in UI
        try:
            response = requests.post(
                f"{api_url}/predict",
                json={
                    "entity1": drug,
                    "entity2": herb,
                    "entity1_type": "drug",
                    "entity2_type": "herb",
                },
                timeout=10,
            )
            result = response.json()
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            result = None
            
    if result:
        st.markdown("---")
        
        # Determine Risk Color
        risk = result.get("risk_level", "unknown").lower()
        badge_class = f"risk-{risk}"
        risk_emoji = {"high": "⚠️", "moderate": "⚡", "low": "✅", "minimal": "🔵"}.get(risk, "⚪")
        
        # Hero Result Card
        st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; color: #e2e8f0;">{drug} ↔ {herb}</h2>
                        <p style="color: #a0aec0; margin-top: 5px;">Link Prediction Analysis Complete</p>
                    </div>
                    <div class="risk-badge {badge_class}">
                        {risk_emoji} {risk.upper()} RISK
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Primary Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <p style="color: #a0aec0; font-weight: 600; margin-bottom: 5px; text-transform: uppercase; font-size: 0.8em;">Interaction Probability</p>
                    <h2 style="margin: 0; font-size: 2.5rem; color: #fff;">{result['interaction_probability']:.1%}</h2>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <p style="color: #a0aec0; font-weight: 600; margin-bottom: 5px; text-transform: uppercase; font-size: 0.8em;">Reliability Score (R)</p>
                    <h2 style="margin: 0; font-size: 2.5rem; color: #fff;">{result['reliability_score']:.2f}</h2>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="glass-card" style="text-align: center;">
                    <p style="color: #a0aec0; font-weight: 600; margin-bottom: 5px; text-transform: uppercase; font-size: 0.8em;">Supporting Evidence</p>
                    <h2 style="margin: 0; font-size: 2.5rem; color: #fff;">{len(result.get("evidence_spans", []))} Spans</h2>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed Breakdown & Evidence Layout
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown("### 🧬 Evidence Reliability Breakdown")
            st.markdown("The R-Score is computed using the attention weights across 5 metadata dimensions from the knowledge graph edges.")
            
            breakdown = result.get("reliability_breakdown", {})
            if breakdown:
                dimensions = {
                    "Corroboration Factor": breakdown.get("corroboration", 0),
                    "Temporal Recency": breakdown.get("temporal_recency", 0),
                    "Biomedical Source Quality": breakdown.get("biomedical_quality", 0),
                    "Molecular Plausibility": breakdown.get("molecular_plausibility", 0),
                    "Source Node Type": breakdown.get("source_type_contribution", 0),
                }
                
                for dim_name, value in dimensions.items():
                    val = min(1.0, max(0.0, value))
                    st.markdown(f"<div style='font-size:0.9em; margin-bottom: 5px;'><b>{dim_name}</b> ({val:.2f})</div>", unsafe_allow_html=True)
                    st.progress(val)
                    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

        with right_col:
            st.markdown("### 📄 Real Graph Evidence")
            st.markdown("Extracted sub-graphs and edge text supporting this link prediction.")
            
            evidence = result.get("evidence_spans", [])
            if evidence:
                for idx, span in enumerate(evidence):
                    with st.expander(f"Evidence Node #{idx+1}", expanded=True):
                        st.write(span)
            else:
                st.info("No direct edge text extracted for this interaction.")
                
            st.markdown("### 💡 Clinical Recommendations")
            for rec in result.get("recommendations", []):
                st.warning(f"• {rec}")
                
            with st.expander("Model Explanation Log"):
                st.write(result.get("explanation", ""))

else:
    # Empty State Visualization
    st.markdown(
        """
        <div style="margin-top: 50px; text-align: center; color: #4facfe;">
            <svg width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="18" cy="5" r="3"></circle>
                <circle cx="6" cy="12" r="3"></circle>
                <circle cx="18" cy="19" r="3"></circle>
                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
            </svg>
            <h3 style="color: #a0aec0; margin-top: 20px;">Awaiting Query</h3>
            <p style="color: #718096; max-width: 500px; margin: 0 auto;">
                Select a drug and herb combination from the sidebar panel to query the cross-attention GNN. 
                The model will return interaction probabilities grounded in real DrugBank edge features.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
