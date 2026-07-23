"""
FastAPI Backend for HDI Prediction Demo

Provides REST API endpoints for:
- Querying herb-drug interaction predictions
- Getting explainable risk scores and evidence
- Health check
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger


# ---------- Pydantic Models ----------

class InteractionQuery(BaseModel):
    """Request model for interaction prediction."""
    entity1: str = Field(..., description="Drug or herb name", example="Warfarin")
    entity2: str = Field(..., description="Drug or herb name", example="St. John's Wort")
    entity1_type: str = Field(default="drug", description="Type: drug or herb")
    entity2_type: str = Field(default="herb", description="Type: drug or herb")


class ReliabilityBreakdown(BaseModel):
    """Reliability score breakdown."""
    corroboration: float = 0.0
    temporal_recency: float = 0.0
    biomedical_quality: float = 0.0
    molecular_plausibility: float = 0.0
    source_type_contribution: float = 0.0


class InteractionResult(BaseModel):
    """Response model for interaction prediction."""
    entity1: str
    entity2: str
    interaction_probability: float
    risk_level: str
    reliability_score: float
    reliability_breakdown: ReliabilityBreakdown
    evidence_spans: list[str] = []
    explanation: str = ""
    recommendations: list[str] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    version: str = "0.1.0"


# ---------- Application ----------

app = FastAPI(
    title="HDI Prediction API",
    description=(
        "Reliability-Conditioned Herb-Drug Interaction Prediction. "
        "Predicts novel interactions via GNN link prediction with "
        "reliability-gated cross-attention fusion."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state (loaded on startup)
_model = None
_graph_data = None
_evidence_surfacer = None


@app.on_event("startup")
async def startup():
    """Load model and data on startup."""
    global _model, _graph_data, _evidence_surfacer
    logger.info("Starting HDI Prediction API...")

    try:
        from src.explainability.evidence_surfacer import EvidenceSurfacer
        _evidence_surfacer = EvidenceSurfacer()
        logger.info("Evidence surfacer loaded")
    except Exception as e:
        logger.warning(f"Could not initialize full pipeline: {e}")
        logger.info("Running in demo mode with synthetic responses")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="0.1.0",
    )


@app.post("/predict", response_model=InteractionResult)
async def predict_interaction(query: InteractionQuery):
    """
    Predict herb-drug interaction with explainable output.

    Returns:
    - Interaction probability
    - Risk level (high/moderate/low/minimal)
    - Reliability score and breakdown (C, T, B, M, S)
    - Evidence spans
    - Natural language explanation
    - Clinical recommendations
    """
    logger.info(
        f"Prediction query: {query.entity1} ({query.entity1_type}) "
        f"↔ {query.entity2} ({query.entity2_type})"
    )

    if _model is not None:
        # Full model inference
        # TODO: Implement full inference pipeline
        pass

    # Demo mode: return synthetic but plausible results
    result = _generate_demo_prediction(query)
    return result


@app.post("/batch_predict", response_model=list[InteractionResult])
async def batch_predict(queries: list[InteractionQuery]):
    """Batch prediction for multiple entity pairs."""
    results = []
    for query in queries:
        result = _generate_demo_prediction(query)
        results.append(result)
    return results


@app.get("/known_entities")
async def list_known_entities():
    """List all known drugs and herbs in the knowledge graph."""
    # Demo data
    return {
        "drugs": [
            "Warfarin", "Metformin", "Digoxin", "Cyclosporine",
            "Simvastatin", "Phenytoin", "Carbamazepine",
            "Omeprazole", "Clopidogrel", "Fluoxetine",
            "Aspirin", "Ibuprofen", "Acetaminophen",
            "Amlodipine", "Lisinopril", "Atorvastatin",
        ],
        "herbs": [
            "Ashwagandha", "Turmeric", "St. John's Wort",
            "Ginkgo", "Garlic", "Ginger", "Neem", "Tulsi",
            "Brahmi", "Shatavari", "Guggul", "Arjuna",
            "Amla", "Guduchi", "Echinacea",
        ],
    }


def _generate_demo_prediction(query: InteractionQuery) -> InteractionResult:
    """Generate a demo prediction with plausible values."""
    # Known high-risk pairs for demo
    high_risk_pairs = {
        ("warfarin", "st. john's wort"): (0.92, 0.88, "high"),
        ("warfarin", "ginkgo"): (0.85, 0.82, "high"),
        ("warfarin", "garlic"): (0.78, 0.75, "moderate"),
        ("digoxin", "st. john's wort"): (0.88, 0.85, "high"),
        ("cyclosporine", "st. john's wort"): (0.91, 0.90, "high"),
        ("metformin", "turmeric"): (0.45, 0.60, "moderate"),
        ("phenytoin", "ginkgo"): (0.55, 0.50, "moderate"),
        ("omeprazole", "turmeric"): (0.30, 0.45, "low"),
    }

    pair = (query.entity1.lower(), query.entity2.lower())
    pair_rev = (query.entity2.lower(), query.entity1.lower())

    if pair in high_risk_pairs:
        prob, rel, risk = high_risk_pairs[pair]
    elif pair_rev in high_risk_pairs:
        prob, rel, risk = high_risk_pairs[pair_rev]
    else:
        import random
        random.seed(hash(pair))
        prob = random.uniform(0.1, 0.5)
        rel = random.uniform(0.3, 0.7)
        risk = "low" if prob < 0.3 else "moderate"

    breakdown = ReliabilityBreakdown(
        corroboration=min(1.0, rel * 1.1),
        temporal_recency=min(1.0, rel * 0.9),
        biomedical_quality=min(1.0, rel * 1.05),
        molecular_plausibility=min(1.0, rel * 0.85),
        source_type_contribution=min(1.0, rel * 0.95),
    )

    evidence = []
    if prob > 0.5:
        evidence.append(
            f"Literature reports suggest {query.entity1} levels may be "
            f"affected by concurrent use of {query.entity2}."
        )
    if prob > 0.7:
        evidence.append(
            f"Multiple case reports document altered drug metabolism "
            f"when {query.entity2} is co-administered with {query.entity1}."
        )

    explanation = (
        f"The model predicts a {prob:.0%} probability of interaction between "
        f"{query.entity1} and {query.entity2}. "
        f"Evidence reliability is {'strong' if rel > 0.7 else 'moderate' if rel > 0.4 else 'limited'} "
        f"(R={rel:.2f})."
    )

    recommendations = []
    if risk in ("high", "moderate"):
        recommendations.append(
            "Consult healthcare provider before combining these substances."
        )
    if query.entity2_type == "herb":
        recommendations.append(
            "Inform your doctor about herbal supplement use."
        )

    return InteractionResult(
        entity1=query.entity1,
        entity2=query.entity2,
        interaction_probability=round(prob, 4),
        risk_level=risk,
        reliability_score=round(rel, 4),
        reliability_breakdown=breakdown,
        evidence_spans=evidence,
        explanation=explanation,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
