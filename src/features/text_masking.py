"""
Outcome Term and Entity Masking Module (Phase 2).

Provides rigorous text masking to isolate study design and methodology from:
1. Outcome conclusions (e.g., 'does not alter', 'inhibition', 'increased AUC')
2. Directional and pharmacokinetic alteration terms
3. Herb and drug entity names (and constituent tokens)
4. Design-only structured extraction
"""

import re
from typing import Dict, List, Optional, Any, Tuple


# Canonical Botanical Entity Synonyms and Constituent Tokens
HERB_TOKENS = [
    r"st\.?\s*johns?(\s*wort)?", r"st\b", r"johns?\b", r"wort\b",
    r"hypericum(\s*perforatum)?", r"hyperforin", r"hypericin",
    r"ginkgo(\s*biloba)?", r"ginkgolides?", r"bilobalide",
    r"ginseng", r"panax", r"ginsenosides?",
    r"garlic", r"allium(\s*sativum)?", r"allicin", r"ajoene",
    r"goldenseal", r"hydrastis(\s*canadensis)?", r"hydrastine", r"berberine",
    r"curcuma(\s*longa)?", r"curcumin(oids?)?", r"turmeric",
    r"green\s*tea", r"camellia(\s*sinensis)?", r"egcg", r"epigallocatechin",
    r"licorice", r"liquorice", r"glycyrrhiz(a|in|ic|inic)", r"glycyrrhetinic",
    r"milk\s*thistle", r"silybum(\s*marianum)?", r"silymarin", r"silybin",
    r"valerian(\s*officinalis)?", r"valerenic",
    r"saw\s*palmetto", r"serenoa(\s*repens)?",
    r"black\s*cohosh", r"actaea(\s*racemosa)?", r"cimicifuga",
    r"kava(\s*kava)?", r"piper\s*methysticum", r"kavalactones?",
    r"echinacea(\s*purpurea)?",
    r"peppermint(\s*oil)?", r"mentha(\s*piperita)?", r"menthol",
    r"cranberry", r"vaccinium",
    r"chamomile", r"matricaria",
    r"grapefruit(\s*juice)?", r"furanocoumarins?", r"bergamottin",
    r"kratom", r"mitragyn(a|ine)",
]

# Canonical Prescription Drug Entity Names and Synonyms
DRUG_TOKENS = [
    r"cyclosporin(e|\s*a)?", r"tacrolimus", r"digoxin", r"warfarin", r"s-warfarin",
    r"midazolam", r"alprazolam", r"theophylline", r"indinavir", r"saquinavir",
    r"ritonavir", r"simvastatin(\s*acid)?", r"atorvastatin", r"rosuvastatin",
    r"dextromethorphan", r"chlorzoxazone", r"felodipine", r"nifedipine",
    r"verapamil", r"nadolol", r"talinolol", r"fexofenadine", r"docetaxel",
    r"paclitaxel", r"irinotecan", r"bortezomib", r"methadone", r"oxycodone",
    r"omeprazole", r"metformin", r"tolbutamide", r"caffeine", r"aspirin",
    r"acetaminophen", r"paracetamol", r"levodopa", r"spironolactone",
    r"finasteride", r"phenelzine", r"hydrochlorothiazide", r"prednisolone",
    r"ethinylestradiol",
]

# Multi-word and phrase-level outcome patterns
OUTCOME_PHRASES = [
    # Negative interaction expressions
    r"does\s+not\s+(?:significantly\s+)?(?:alter|affect|inhibit|induce|modify|change|produce|prolong)",
    r"did\s+not\s+(?:significantly\s+)?(?:alter|affect|inhibit|induce|modify|change|produce|prolong)",
    r"do\s+not\s+(?:significantly\s+)?(?:alter|affect|inhibit|induce|modify|change|produce|prolong)",
    r"fails?\s+to\s+(?:significantly\s+)?(?:alter|affect|inhibit|induce|modify|change|produce)",
    r"failed\s+to\s+(?:significantly\s+)?(?:alter|affect|inhibit|induce|modify|change|produce)",
    r"(?:demonstrat(?:ing|ed)|show(?:ing|ed))\s+(?:no|lack\s+of|minimal)\s+(?:significant\s+)?(?:change|effect|interaction|alteration|difference)",
    r"lack\s+of\s+(?:significant\s+)?(?:effect|interaction|change|alteration|influence|inhibition|induction)",
    r"absence\s+of\s+(?:significant\s+)?(?:effect|interaction|change|alteration)",
    r"no\s+(?:statistically\s+)?significant\s+(?:effect|change|interaction|difference|alteration|influence)",
    r"no\s+effect\s+on",
    r"no\s+change\s+in",
    r"without\s+(?:any\s+)?(?:significant\s+)?(?:effect|change|alteration)",
    r"was\s+not\s+associated\s+with",
    r"well\s+tolerated\s+without",
    r"comparable\s+to\s+placebo",

    # Positive interaction expressions
    r"(?:markedly|significantly|profoundly|greatly|substantially)\s+(?:decreases?|increases?|reduces?|elevates?|inhibits?|induces?|potentiates?)",
    r"leading\s+to\s+(?:acute\s+)?(?:organ\s+rejection|bleeding|toxicity|hematoma|hemorrhage|failure|elevation)",
    r"resulting\s+in\s+(?:a\s+)?(?:\d+\s*(?:percent|%)\s+)?(?:reduction|increase|decrease|drop|elevation)",
    r"profound\s+drop\s+of",
    r"apparent\s+(?:[a-z]+\s+)?toxicity",
    r"potentiation\s+of",
    r"elevat(?:ed|ing|es?)\s+(?:inr|prothrombin|plasma|blood|concentrations?|levels?|auc)",
    r"acute\s+(?:organ\s+)?rejection",
    r"spontaneous\s+(?:spinal\s+)?(?:epidural\s+)?hematoma",
    r"postoperative\s+bleeding",

    # Percentage and metric modifications
    r"\b\d+\s*%\s*(?:reduction|increase|decrease|drop|change)?",
    r"\b\d+\s*percent\s*(?:reduction|increase|decrease|drop|change)?",
    r"\b(?:inr|auc|cmax|tmax|clearance|trough|half-life|bioavailability|exposure)\b",

    # Single outcome/alteration verbs and nouns
    r"\b(?:decreas(?:es|ed|ing|e)|increas(?:es|ed|ing|e)|reduc(?:es|ed|ing|tion|tions|e)|elevat(?:es|ed|ing|ion|e)|inhibit(?:s|ed|ing|ion|or|ors)?|induc(?:es|ed|ing|tion|er|ers|e)|potentiated?|potentiating|antagoniz(?:es|ed|ing|e)|altered?|altering|affects?|affected|affecting)\b",
    r"\b(?:interaction|interactions|rejection|bleeding|hematoma|hemorrhage|toxicity|toxic|subtherapeutic|adverse)\b",
    r"\b(?:significantly|significant|markedly|profoundly|greatly|substantially)\b",
]

# Additional clinical setting words that act as outcome proxies
OUTCOME_PROXIES = [
    r"\b(?:concentrations?|levels?|plasma|blood|serum|systemic)\b",
    r"\b(?:showing|demonstrating|associated|produced|resulting|causing)\b",
    r"\b(?:recipients?|transplant|acute|rejection|bleeding|hematoma)\b",
    r"\b(?:healthy|volunteers?|randomized|placebo|controlled)\b",
]

_RE_HERBS = re.compile(r"\b(" + "|".join(HERB_TOKENS) + r")\b", re.IGNORECASE)
_RE_DRUGS = re.compile(r"\b(" + "|".join(DRUG_TOKENS) + r")\b", re.IGNORECASE)
_RE_OUTCOMES = [re.compile(p, re.IGNORECASE) for p in OUTCOME_PHRASES]
_RE_PROXIES = [re.compile(p, re.IGNORECASE) for p in OUTCOME_PROXIES]


def mask_outcome_language(text: str) -> str:
    """
    Mask all explicit outcome, interaction, and directional alteration terms.
    Preserves entity mentions and methodological context.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text
    for pat in _RE_OUTCOMES:
        t = pat.sub(" [MASK_OUTCOME] ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def mask_entities(text: str) -> str:
    """
    Mask botanical and prescription drug entity mentions with generic placeholders.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    t = _RE_HERBS.sub(" [HERB] ", text)
    t = _RE_DRUGS.sub(" [DRUG] ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def mask_outcome_and_entities(text: str, mask_proxies: bool = False) -> str:
    """
    Full masking function stripping both outcome conclusions and entity names.
    Optionally masks clinical setting proxy words if mask_proxies is True.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text
    # 1. Mask outcome terms
    for pat in _RE_OUTCOMES:
        t = pat.sub(" ", t)
    # 2. Mask entities
    t = _RE_HERBS.sub(" [HERB] ", t)
    t = _RE_DRUGS.sub(" [DRUG] ", t)
    # 3. Optional proxy masking
    if mask_proxies:
        for pat in _RE_PROXIES:
            t = pat.sub(" ", t)
    # 4. Clean formatting
    t = re.sub(r"[^\w\s\[\]]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_study_design_text(text: str) -> str:
    """
    Extract strictly methodological and study-design tokens from abstract/text:
    Species, clinical study phase/format, route, administration duration, sample size.
    Excludes all outcome findings.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    design_tokens = []
    lower = text.lower()

    # Study type / Assay
    if "randomized" in lower or "rct" in lower:
        design_tokens.append("randomized_trial")
    elif "clinical" in lower or "subjects" in lower or "volunteers" in lower:
        design_tokens.append("clinical_study")
    elif "case report" in lower or "recipient" in lower or "patient" in lower:
        design_tokens.append("case_observation")
    elif "in vitro" in lower or "microsome" in lower:
        design_tokens.append("in_vitro_assay")
    elif "animal" in lower or "rat" in lower or "mouse" in lower:
        design_tokens.append("preclinical_in_vivo")

    # Route
    if "oral" in lower or "ingestion" in lower or "intake" in lower:
        design_tokens.append("oral_administration")
    elif "intravenous" in lower or "iv" in lower:
        design_tokens.append("intravenous_administration")

    # Formulation / Preparation
    if "extract" in lower or "standardized" in lower:
        design_tokens.append("standardized_extract")
    elif "juice" in lower:
        design_tokens.append("juice_preparation")
    elif "oil" in lower:
        design_tokens.append("oil_formulation")
    elif "tea" in lower:
        design_tokens.append("herbal_tea")

    return " ".join(design_tokens) if design_tokens else "methodology_unspecified"
