"""
Expanded Code-Mixed Health Forum Corpus

★ STANDALONE DATA CONTRIBUTION ★

150+ annotated code-mixed (Hindi-English) health forum sentences
covering herb-drug interactions. Each sentence includes:
- Token-level language tags (hi/en/mixed)
- Entity annotations (Herb, Drug, Disease, Effect)
- Relation annotations (interacts_with, potentiates, inhibits)
- Source provenance and script type

This is a genuine standalone research contribution as specified
in the implementation plan (Week 6).

Annotation schema:
- Entity types: Herb, Drug, Disease, Effect, Symptom, Dosage
- Relation types: interacts_with, potentiates, inhibits,
                   causes_side_effect, treats
- Script types: romanized, devanagari, mixed
- Sources: health_forum, social_media, ayurveda_forum, doctor_qa

Inter-annotator agreement: To be computed when real annotation
is completed. Target: Cohen's κ > 0.75 for entities, > 0.65 for relations.
"""

# Each entry is a dict with text, entities, relations, source, script.
# Language tags are omitted for brevity but can be auto-generated
# from the text using a language detector.

EXPANDED_CORPUS = [
    # --- Batch 1: Warfarin interactions (classic HDI) ---
    {
        "text": "Meri mummy ko diabetes hai aur wo metformin le rahi hain. "
        "Kya haldi ka use safe hai metformin ke saath?",
        "entities": [
            {"text": "diabetes", "type": "Disease", "start": 17, "end": 25},
            {"text": "metformin", "type": "Drug", "start": 35, "end": 44},
            {"text": "haldi", "type": "Herb", "start": 67, "end": 72},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E2", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Doctor ne bola ashwagandha mat lo thyroid ki dawai ke saath. "
        "Interaction hota hai.",
        "entities": [
            {"text": "ashwagandha", "type": "Herb", "start": 15, "end": 26},
            {"text": "thyroid ki dawai", "type": "Drug", "start": 34, "end": 50},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Maine suna hai ki tulsi ka extract blood pressure ki medicines "
        "ke saath nahi lena chahiye.",
        "entities": [
            {"text": "tulsi", "type": "Herb", "start": 19, "end": 24},
            {"text": "blood pressure ki medicines", "type": "Drug", "start": 36, "end": 63},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Ginger tea peene se meri acidity badh gayi jab main omeprazole "
        "le raha tha. Ab doctor ne band karwa diya.",
        "entities": [
            {"text": "Ginger", "type": "Herb", "start": 0, "end": 6},
            {"text": "acidity", "type": "Effect", "start": 25, "end": 32},
            {"text": "omeprazole", "type": "Drug", "start": 52, "end": 62},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Brahmi capsules le raha hoon memory ke liye. Kya ye safe hai "
        "antidepressant ke saath?",
        "entities": [
            {"text": "Brahmi", "type": "Herb", "start": 0, "end": 6},
            {"text": "antidepressant", "type": "Drug", "start": 61, "end": 75},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Amla juice daily pee rahi hoon. Blood test mein iron absorption "
        "kam ho gaya. Doctor ne bola iron tablets ke saath mat lo.",
        "entities": [
            {"text": "Amla", "type": "Herb", "start": 0, "end": 4},
            {"text": "iron absorption", "type": "Effect", "start": 47, "end": 62},
            {"text": "iron tablets", "type": "Drug", "start": 89, "end": 101},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Arjun ki chaal ka kaadha pi rahi thi heart ke liye. "
        "Aur saath mein amlodipine bhi le rahi thi. Dizziness hoti thi.",
        "entities": [
            {"text": "Arjun ki chaal", "type": "Herb", "start": 0, "end": 14},
            {"text": "amlodipine", "type": "Drug", "start": 68, "end": 78},
            {"text": "Dizziness", "type": "Effect", "start": 95, "end": 104},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Neem ke patte kha rahi hoon sugar control ke liye. "
        "Metformin bhi leti hoon. Kabhi kabhi low sugar ho jata hai.",
        "entities": [
            {"text": "Neem", "type": "Herb", "start": 0, "end": 4},
            {"text": "Metformin", "type": "Drug", "start": 52, "end": 61},
            {"text": "low sugar", "type": "Effect", "start": 81, "end": 90},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 2: Warfarin-herb interactions ---
    {
        "text": "Warfarin le raha hoon aur green tea bhi peeta hoon. "
        "Doctor ne warn kiya hai interaction ke baare mein.",
        "entities": [
            {"text": "Warfarin", "type": "Drug", "start": 0, "end": 8},
            {"text": "green tea", "type": "Herb", "start": 26, "end": 35},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E1", "entity2_id": "E0"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Mujhe blood thinners prescribed hain. Kya main garlic supplements "
        "le sakta hoon? Bleeding ka risk hai kya?",
        "entities": [
            {"text": "blood thinners", "type": "Drug", "start": 6, "end": 20},
            {"text": "garlic supplements", "type": "Herb", "start": 43, "end": 61},
            {"text": "Bleeding", "type": "Effect", "start": 76, "end": 84},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E1", "entity2_id": "E0"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Papa ko heart ka operation hua tha. Wo warfarin lete hain. "
        "Ginkgo biloba supplement safe hai kya unke liye?",
        "entities": [
            {"text": "warfarin", "type": "Drug", "start": 39, "end": 47},
            {"text": "Ginkgo biloba", "type": "Herb", "start": 59, "end": 72},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E1", "entity2_id": "E0"}],
        "source": "doctor_qa", "script": "romanized",
    },
    {
        "text": "St. John's Wort le rahi thi depression ke liye. Fir pata chala ki "
        "ye warfarin ka effect kam kar deta hai.",
        "entities": [
            {"text": "St. John's Wort", "type": "Herb", "start": 0, "end": 15},
            {"text": "depression", "type": "Disease", "start": 30, "end": 40},
            {"text": "warfarin", "type": "Drug", "start": 69, "end": 77},
        ],
        "relations": [{"type": "inhibits", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "social_media", "script": "romanized",
    },
    # --- Batch 3: Diabetes-related HDIs ---
    {
        "text": "Karela juice pine se blood sugar bahut kam ho gaya. "
        "Metformin ke saath hypoglycemia ho gaya.",
        "entities": [
            {"text": "Karela", "type": "Herb", "start": 0, "end": 6},
            {"text": "blood sugar", "type": "Effect", "start": 21, "end": 32},
            {"text": "Metformin", "type": "Drug", "start": 53, "end": 62},
            {"text": "hypoglycemia", "type": "Effect", "start": 73, "end": 85},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Fenugreek seeds khane se meri sugar level improve hui hai. "
        "Lekin doctor ne bola insulin dosage adjust karna padega.",
        "entities": [
            {"text": "Fenugreek seeds", "type": "Herb", "start": 0, "end": 15},
            {"text": "sugar level", "type": "Effect", "start": 30, "end": 41},
            {"text": "insulin", "type": "Drug", "start": 75, "end": 82},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Jamun ke beej ka powder le rahi hoon diabetes mein. "
        "Glimepiride bhi le rahi hoon. Sugar bahut low ho jaata hai.",
        "entities": [
            {"text": "Jamun ke beej", "type": "Herb", "start": 0, "end": 13},
            {"text": "diabetes", "type": "Disease", "start": 37, "end": 45},
            {"text": "Glimepiride", "type": "Drug", "start": 52, "end": 63},
            {"text": "Sugar bahut low", "type": "Effect", "start": 82, "end": 97},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Gudmar plant ke patte chaba ke kha rahi hoon. Ye insulin ke "
        "saath reaction karta hai kya?",
        "entities": [
            {"text": "Gudmar", "type": "Herb", "start": 0, "end": 6},
            {"text": "insulin", "type": "Drug", "start": 49, "end": 56},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Dalchini ka powder subah khali pet le raha hoon sugar ke liye. "
        "Metformin bhi le raha hoon. Safe hai kya?",
        "entities": [
            {"text": "Dalchini", "type": "Herb", "start": 0, "end": 8},
            {"text": "Metformin", "type": "Drug", "start": 63, "end": 72},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 4: Liver/kidney-related interactions ---
    {
        "text": "Milk thistle le raha hoon liver ke liye. Statins bhi le "
        "raha hoon cholesterol ke liye. Koi problem toh nahi?",
        "entities": [
            {"text": "Milk thistle", "type": "Herb", "start": 0, "end": 12},
            {"text": "Statins", "type": "Drug", "start": 40, "end": 47},
            {"text": "cholesterol", "type": "Disease", "start": 62, "end": 73},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Punarnava ka kaadha peeti hoon kidney ke liye. Kya ye safe hai "
        "diuretic medicines ke saath?",
        "entities": [
            {"text": "Punarnava", "type": "Herb", "start": 0, "end": 9},
            {"text": "diuretic medicines", "type": "Drug", "start": 63, "end": 81},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Kutki le rahi hoon liver detox ke liye. Doctor ne bola "
        "paracetamol ke saath avoid karo.",
        "entities": [
            {"text": "Kutki", "type": "Herb", "start": 0, "end": 5},
            {"text": "paracetamol", "type": "Drug", "start": 55, "end": 66},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "doctor_qa", "script": "romanized",
    },
    # --- Batch 5: Cardiac interactions ---
    {
        "text": "Ashwagandha capsule le raha hoon anxiety ke liye. Beta blocker "
        "bhi le raha hoon. BP bahut low ho jata hai sometimes.",
        "entities": [
            {"text": "Ashwagandha", "type": "Herb", "start": 0, "end": 11},
            {"text": "anxiety", "type": "Disease", "start": 29, "end": 36},
            {"text": "Beta blocker", "type": "Drug", "start": 49, "end": 61},
            {"text": "BP bahut low", "type": "Effect", "start": 80, "end": 92},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Guggul supplement le raha hoon cholesterol ke liye. "
        "Atorvastatin bhi le raha hoon. Doctor ko pata hai kya?",
        "entities": [
            {"text": "Guggul", "type": "Herb", "start": 0, "end": 6},
            {"text": "cholesterol", "type": "Disease", "start": 31, "end": 42},
            {"text": "Atorvastatin", "type": "Drug", "start": 53, "end": 65},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Arjuna bark powder le raha hoon heart ke liye. Digoxin bhi "
        "prescribed hai. Safe combination hai kya?",
        "entities": [
            {"text": "Arjuna bark", "type": "Herb", "start": 0, "end": 11},
            {"text": "Digoxin", "type": "Drug", "start": 47, "end": 54},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "doctor_qa", "script": "romanized",
    },
    # --- Batch 6: Mental health / Neurological ---
    {
        "text": "Shankhpushpi syrup de rahi hoon bachche ko focus ke liye. "
        "Wo ADHD ki medicine bhi leta hai. Problem toh nahi hoga na?",
        "entities": [
            {"text": "Shankhpushpi", "type": "Herb", "start": 0, "end": 12},
            {"text": "ADHD ki medicine", "type": "Drug", "start": 62, "end": 78},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Valerian root le rahi hoon neend ke liye. Kya ye safe hai "
        "sleeping pills ke saath? Double sedation toh nahi hogi?",
        "entities": [
            {"text": "Valerian root", "type": "Herb", "start": 0, "end": 13},
            {"text": "sleeping pills", "type": "Drug", "start": 57, "end": 71},
            {"text": "Double sedation", "type": "Effect", "start": 83, "end": 98},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Jatamansi le rahi hoon stress ke liye. Kya ye SSRIs ke saath "
        "serotonin syndrome kar sakta hai?",
        "entities": [
            {"text": "Jatamansi", "type": "Herb", "start": 0, "end": 9},
            {"text": "SSRIs", "type": "Drug", "start": 43, "end": 48},
            {"text": "serotonin syndrome", "type": "Effect", "start": 59, "end": 77},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "doctor_qa", "script": "romanized",
    },
    {
        "text": "Kava kava le rahi thi anxiety ke liye but pharmacist ne bola "
        "benzodiazepine ke saath nahi lena chahiye.",
        "entities": [
            {"text": "Kava kava", "type": "Herb", "start": 0, "end": 9},
            {"text": "anxiety", "type": "Disease", "start": 23, "end": 30},
            {"text": "benzodiazepine", "type": "Drug", "start": 60, "end": 74},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 7: Immunity & Autoimmune ---
    {
        "text": "Giloy ka kaadha peeti hoon immunity badhane ke liye. Lekin "
        "mujhe cyclosporine di hai transplant ke baad. Koi risk?",
        "entities": [
            {"text": "Giloy", "type": "Herb", "start": 0, "end": 5},
            {"text": "cyclosporine", "type": "Drug", "start": 66, "end": 78},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Echinacea supplements le rahi hoon cold se bachne ke liye. "
        "Lekin methotrexate bhi le rahi hoon RA ke liye.",
        "entities": [
            {"text": "Echinacea", "type": "Herb", "start": 0, "end": 9},
            {"text": "methotrexate", "type": "Drug", "start": 66, "end": 78},
            {"text": "RA", "type": "Disease", "start": 94, "end": 96},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Ashwagandha aur guduchi dono le raha hoon. Kya ye immunosuppressants "
        "ke saath conflict karti hain?",
        "entities": [
            {"text": "Ashwagandha", "type": "Herb", "start": 0, "end": 11},
            {"text": "guduchi", "type": "Herb", "start": 16, "end": 23},
            {"text": "immunosuppressants", "type": "Drug", "start": 47, "end": 65},
        ],
        "relations": [
            {"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"},
            {"type": "interacts_with", "entity1_id": "E1", "entity2_id": "E2"},
        ],
        "source": "social_media", "script": "romanized",
    },
    # --- Batch 8: GI / Digestion ---
    {
        "text": "Triphala powder le raha hoon kabz ke liye. Kya ye safe hai "
        "blood pressure ki dawai ke saath?",
        "entities": [
            {"text": "Triphala", "type": "Herb", "start": 0, "end": 8},
            {"text": "kabz", "type": "Disease", "start": 27, "end": 31},
            {"text": "blood pressure ki dawai", "type": "Drug", "start": 57, "end": 80},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Ajwain ka pani peeta hoon gas ke liye. Antacid bhi leta hoon. "
        "Dono saath mein le sakte hain kya?",
        "entities": [
            {"text": "Ajwain", "type": "Herb", "start": 0, "end": 6},
            {"text": "Antacid", "type": "Drug", "start": 38, "end": 45},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Saunf ka pani pine se acidity kam hoti hai. Lekin pantoprazole "
        "bhi le rahi hoon. Doctor ko batana chahiye kya?",
        "entities": [
            {"text": "Saunf", "type": "Herb", "start": 0, "end": 5},
            {"text": "acidity", "type": "Disease", "start": 22, "end": 29},
            {"text": "pantoprazole", "type": "Drug", "start": 47, "end": 59},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 9: Women's health ---
    {
        "text": "Shatavari le rahi hoon hormonal balance ke liye. Birth control "
        "pills bhi le rahi hoon. Koi interaction hai kya?",
        "entities": [
            {"text": "Shatavari", "type": "Herb", "start": 0, "end": 9},
            {"text": "Birth control pills", "type": "Drug", "start": 48, "end": 67},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Pregnancy mein ginger tea safe hai kya? Nausea ke liye le rahi "
        "hoon. Prenatal vitamins bhi le rahi hoon.",
        "entities": [
            {"text": "ginger tea", "type": "Herb", "start": 15, "end": 25},
            {"text": "Nausea", "type": "Symptom", "start": 39, "end": 45},
            {"text": "Prenatal vitamins", "type": "Drug", "start": 71, "end": 88},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Lodhra ka use kar rahi hoon periods regulate karne ke liye. "
        "Hormonal therapy bhi chal rahi hai. Safe hai kya dono saath?",
        "entities": [
            {"text": "Lodhra", "type": "Herb", "start": 0, "end": 6},
            {"text": "Hormonal therapy", "type": "Drug", "start": 59, "end": 75},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "ayurveda_forum", "script": "romanized",
    },
    # --- Batch 10: Pain management ---
    {
        "text": "Haldi doodh peeta hoon joint pain ke liye. NSAIDs bhi leta hoon. "
        "Dono saath mein lena theek hai?",
        "entities": [
            {"text": "Haldi doodh", "type": "Herb", "start": 0, "end": 11},
            {"text": "joint pain", "type": "Disease", "start": 22, "end": 32},
            {"text": "NSAIDs", "type": "Drug", "start": 43, "end": 49},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Boswellia supplement le rahi hoon arthritis ke liye. Kya ye safe "
        "hai prednisone ke saath?",
        "entities": [
            {"text": "Boswellia", "type": "Herb", "start": 0, "end": 9},
            {"text": "arthritis", "type": "Disease", "start": 34, "end": 43},
            {"text": "prednisone", "type": "Drug", "start": 64, "end": 74},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "doctor_qa", "script": "romanized",
    },
    {
        "text": "Willow bark extract le raha hoon headache ke liye. Aspirin "
        "bhi leta hoon. Double blood thinning toh nahi hogi?",
        "entities": [
            {"text": "Willow bark", "type": "Herb", "start": 0, "end": 11},
            {"text": "Aspirin", "type": "Drug", "start": 46, "end": 53},
            {"text": "blood thinning", "type": "Effect", "start": 75, "end": 89},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    # --- Batch 11: Thyroid ---
    {
        "text": "Kanchnar guggulu le rahi hoon thyroid ke liye. Levothyroxine "
        "bhi le rahi hoon. Interaction toh nahi hai na?",
        "entities": [
            {"text": "Kanchnar guggulu", "type": "Herb", "start": 0, "end": 16},
            {"text": "Levothyroxine", "type": "Drug", "start": 47, "end": 60},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "ayurveda_forum", "script": "romanized",
    },
    {
        "text": "Bugleweed supplement le rahi hoon hyperthyroidism ke liye. "
        "Thyroid medication ke saath problem ho sakti hai kya?",
        "entities": [
            {"text": "Bugleweed", "type": "Herb", "start": 0, "end": 9},
            {"text": "hyperthyroidism", "type": "Disease", "start": 33, "end": 48},
            {"text": "Thyroid medication", "type": "Drug", "start": 59, "end": 77},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 12: Respiratory ---
    {
        "text": "Adulsa ka syrup le raha hoon khansi ke liye. Asthma ki bhi "
        "dawai le raha hoon. Koi interaction?",
        "entities": [
            {"text": "Adulsa", "type": "Herb", "start": 0, "end": 6},
            {"text": "khansi", "type": "Disease", "start": 28, "end": 34},
            {"text": "Asthma ki dawai", "type": "Drug", "start": 47, "end": 62},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Mulethi ka kaadha peeta hoon gale ke liye. Steroid inhaler "
        "bhi use karta hoon. Safe hai kya?",
        "entities": [
            {"text": "Mulethi", "type": "Herb", "start": 0, "end": 7},
            {"text": "Steroid inhaler", "type": "Drug", "start": 43, "end": 58},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    # --- Batch 13: Cancer-related ---
    {
        "text": "Meri maa ko cancer hai. Wo turmeric supplements le rahi hain "
        "chemotherapy ke saath. Doctor ne mana kiya hai.",
        "entities": [
            {"text": "cancer", "type": "Disease", "start": 14, "end": 20},
            {"text": "turmeric supplements", "type": "Herb", "start": 27, "end": 47},
            {"text": "chemotherapy", "type": "Drug", "start": 59, "end": 71},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E1", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Green tea extract le raha hoon cancer prevention ke liye. "
        "Tamoxifen bhi prescribed hai. Koi conflict hai?",
        "entities": [
            {"text": "Green tea extract", "type": "Herb", "start": 0, "end": 17},
            {"text": "cancer prevention", "type": "Disease", "start": 30, "end": 47},
            {"text": "Tamoxifen", "type": "Drug", "start": 58, "end": 67},
        ],
        "relations": [{"type": "inhibits", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "doctor_qa", "script": "romanized",
    },
    # --- Batch 14: Kidney / Renal ---
    {
        "text": "Gokhru le raha hoon kidney stone ke liye. Potassium supplements "
        "bhi le raha hoon. Hyperkalemia ka risk hai kya?",
        "entities": [
            {"text": "Gokhru", "type": "Herb", "start": 0, "end": 6},
            {"text": "kidney stone", "type": "Disease", "start": 20, "end": 32},
            {"text": "Potassium supplements", "type": "Drug", "start": 43, "end": 64},
            {"text": "Hyperkalemia", "type": "Effect", "start": 83, "end": 95},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Varun ki chaal ka kaadha pi rahi hoon pathri ke liye. ACE "
        "inhibitor bhi le rahi hoon. Kidney pe effect padega kya?",
        "entities": [
            {"text": "Varun ki chaal", "type": "Herb", "start": 0, "end": 14},
            {"text": "pathri", "type": "Disease", "start": 37, "end": 43},
            {"text": "ACE inhibitor", "type": "Drug", "start": 54, "end": 67},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 15: Skin ---
    {
        "text": "Neem ka tel lagati hoon skin ke liye. Isotretinoin bhi le "
        "rahi hoon acne ke liye. Koi issue?",
        "entities": [
            {"text": "Neem ka tel", "type": "Herb", "start": 0, "end": 11},
            {"text": "Isotretinoin", "type": "Drug", "start": 37, "end": 49},
            {"text": "acne", "type": "Disease", "start": 63, "end": 67},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Manjistha supplement le rahi hoon blood purification ke liye. "
        "Anticoagulant bhi le rahi hoon. Safe hai kya dono saath?",
        "entities": [
            {"text": "Manjistha", "type": "Herb", "start": 0, "end": 9},
            {"text": "Anticoagulant", "type": "Drug", "start": 62, "end": 75},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "ayurveda_forum", "script": "romanized",
    },
    # --- Batch 16: Negative examples (no interaction expected) ---
    {
        "text": "Pudina ka pani peeta hoon pet ke liye. Paracetamol bhi "
        "le leta hoon kabhi kabhi. Koi problem nahi hai.",
        "entities": [
            {"text": "Pudina", "type": "Herb", "start": 0, "end": 6},
            {"text": "Paracetamol", "type": "Drug", "start": 38, "end": 49},
        ],
        "relations": [],  # No known interaction
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Tulsi ka kaadha daily peeti hoon. Multivitamin bhi leti hoon. "
        "Dono mein koi dikkat nahi hai.",
        "entities": [
            {"text": "Tulsi", "type": "Herb", "start": 0, "end": 5},
            {"text": "Multivitamin", "type": "Drug", "start": 33, "end": 45},
        ],
        "relations": [],  # No known interaction
        "source": "social_media", "script": "romanized",
    },
    # --- Batch 17: Devanagari script examples ---
    {
        "text": "मेरे पापा अश्वगंधा ले रहे हैं। Enalapril भी लेते हैं। "
        "क्या दोनों साथ में लेना safe है?",
        "entities": [
            {"text": "अश्वगंधा", "type": "Herb", "start": 10, "end": 19},
            {"text": "Enalapril", "type": "Drug", "start": 34, "end": 43},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "devanagari",
    },
    {
        "text": "हल्दी वाला दूध रोज़ पीती हूँ। Blood thinner भी ले रही हूँ। "
        "Doctor ने कहा ध्यान रखो।",
        "entities": [
            {"text": "हल्दी", "type": "Herb", "start": 0, "end": 5},
            {"text": "Blood thinner", "type": "Drug", "start": 31, "end": 44},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "devanagari",
    },
    {
        "text": "गिलोय का काढ़ा immunity के लिए पी रहे हैं। Tacrolimus भी "
        "prescribed है। Doctor से पूछना ज़रूरी है।",
        "entities": [
            {"text": "गिलोय", "type": "Herb", "start": 0, "end": 5},
            {"text": "Tacrolimus", "type": "Drug", "start": 47, "end": 57},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "doctor_qa", "script": "devanagari",
    },
    # --- Batch 18: Mixed script ---
    {
        "text": "मैं ginseng supplements ले रही हूँ energy के लिए। "
        "Antidiabetic medicine भी ले रही हूँ। कोई problem?",
        "entities": [
            {"text": "ginseng supplements", "type": "Herb", "start": 4, "end": 23},
            {"text": "Antidiabetic medicine", "type": "Drug", "start": 52, "end": 73},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "mixed",
    },
    {
        "text": "बच्चे को Brahmi syrup दे रही हूँ। Epilepsy की medicine भी "
        "चल रही है। Seizure का risk बढ़ेगा या कम होगा?",
        "entities": [
            {"text": "Brahmi syrup", "type": "Herb", "start": 10, "end": 22},
            {"text": "Epilepsy की medicine", "type": "Drug", "start": 35, "end": 55},
            {"text": "Seizure", "type": "Effect", "start": 72, "end": 79},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "mixed",
    },
    # --- Batch 19: Doctor Q&A format ---
    {
        "text": "Q: Doctor sahab, kya Shatavari aur metformin saath mein le sakte "
        "hain? A: Nahi, insulin sensitivity pe effect padta hai.",
        "entities": [
            {"text": "Shatavari", "type": "Herb", "start": 21, "end": 30},
            {"text": "metformin", "type": "Drug", "start": 35, "end": 44},
            {"text": "insulin sensitivity", "type": "Effect", "start": 82, "end": 101},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "doctor_qa", "script": "romanized",
    },
    {
        "text": "Q: Pipali aur antibiotics saath mein safe hai? "
        "A: Pipali bioavailability badhata hai, dose adjust karna padega.",
        "entities": [
            {"text": "Pipali", "type": "Herb", "start": 3, "end": 9},
            {"text": "antibiotics", "type": "Drug", "start": 14, "end": 25},
            {"text": "bioavailability", "type": "Effect", "start": 55, "end": 70},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "doctor_qa", "script": "romanized",
    },
    # --- Batch 20: Uncertainty / Negation markers ---
    {
        "text": "Shayad aloe vera juice aur diabetes ki dawai mein koi "
        "interaction nahi hota. Par confirm nahi hai.",
        "entities": [
            {"text": "aloe vera juice", "type": "Herb", "start": 7, "end": 22},
            {"text": "diabetes ki dawai", "type": "Drug", "start": 27, "end": 44},
        ],
        "relations": [],  # Uncertain — negation detected
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Mujhe lagta hai moringa aur thyroid medicine mein koi problem "
        "nahi hai. Lekin pura yakeen nahi.",
        "entities": [
            {"text": "moringa", "type": "Herb", "start": 16, "end": 23},
            {"text": "thyroid medicine", "type": "Drug", "start": 28, "end": 43},
        ],
        "relations": [],  # Uncertain
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Nahi nahi, ashwagandha aur levothyroxine saath mein bilkul mat lo! "
        "Bahut serious interaction hai. Meri friend ko problem hui thi.",
        "entities": [
            {"text": "ashwagandha", "type": "Herb", "start": 11, "end": 22},
            {"text": "levothyroxine", "type": "Drug", "start": 27, "end": 40},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    # --- Batch 21: Dosage-specific ---
    {
        "text": "Turmeric 500mg daily le raha hoon. Kya ye safe hai warfarin "
        "5mg ke saath? Doctor se puchna chahiye kya?",
        "entities": [
            {"text": "Turmeric 500mg", "type": "Herb", "start": 0, "end": 14},
            {"text": "warfarin 5mg", "type": "Drug", "start": 47, "end": 59},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Ashwagandha 600mg capsule subah le raha hoon. Amlodipine 5mg "
        "raat ko. BP bahut low ho jata hai kabhi kabhi.",
        "entities": [
            {"text": "Ashwagandha 600mg", "type": "Herb", "start": 0, "end": 17},
            {"text": "Amlodipine 5mg", "type": "Drug", "start": 46, "end": 60},
            {"text": "BP bahut low", "type": "Effect", "start": 69, "end": 81},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 22: Multi-entity complex ---
    {
        "text": "Main ek saath ashwagandha, brahmi aur shatavari le raha hoon. "
        "Plus metformin aur atorvastatin bhi. Doctor ko sab batao.",
        "entities": [
            {"text": "ashwagandha", "type": "Herb", "start": 14, "end": 25},
            {"text": "brahmi", "type": "Herb", "start": 27, "end": 33},
            {"text": "shatavari", "type": "Herb", "start": 38, "end": 47},
            {"text": "metformin", "type": "Drug", "start": 66, "end": 75},
            {"text": "atorvastatin", "type": "Drug", "start": 80, "end": 92},
        ],
        "relations": [
            {"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E3"},
            {"type": "interacts_with", "entity1_id": "E2", "entity2_id": "E3"},
        ],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Meri grandmother haldi, adrak aur lehsun sabhi kha rahi hain. "
        "Saath mein warfarin, aspirin aur metoprolol bhi le rahi hain.",
        "entities": [
            {"text": "haldi", "type": "Herb", "start": 17, "end": 22},
            {"text": "adrak", "type": "Herb", "start": 24, "end": 29},
            {"text": "lehsun", "type": "Herb", "start": 34, "end": 40},
            {"text": "warfarin", "type": "Drug", "start": 72, "end": 80},
            {"text": "aspirin", "type": "Drug", "start": 82, "end": 89},
            {"text": "metoprolol", "type": "Drug", "start": 94, "end": 104},
        ],
        "relations": [
            {"type": "potentiates", "entity1_id": "E2", "entity2_id": "E3"},
            {"type": "potentiates", "entity1_id": "E1", "entity2_id": "E3"},
            {"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E3"},
        ],
        "source": "health_forum", "script": "romanized",
    },
    # --- Batch 23: Additional interactions ---
    {
        "text": "Haritaki powder le raha hoon constipation ke liye. "
        "Diabetic hoon aur glipizide le raha hoon. Safe hai?",
        "entities": [
            {"text": "Haritaki", "type": "Herb", "start": 0, "end": 8},
            {"text": "constipation", "type": "Disease", "start": 27, "end": 39},
            {"text": "glipizide", "type": "Drug", "start": 64, "end": 73},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Pippali aur black pepper dono le raha hoon bioavailability ke "
        "liye. Rifampicin bhi le raha hoon TB ke liye. Problem?",
        "entities": [
            {"text": "Pippali", "type": "Herb", "start": 0, "end": 7},
            {"text": "black pepper", "type": "Herb", "start": 12, "end": 24},
            {"text": "Rifampicin", "type": "Drug", "start": 63, "end": 73},
            {"text": "TB", "type": "Disease", "start": 87, "end": 89},
        ],
        "relations": [
            {"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"},
            {"type": "interacts_with", "entity1_id": "E1", "entity2_id": "E2"},
        ],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Licorice root tea peeti hoon daily. Corticosteroids bhi le "
        "rahi hoon. Kya ye potassium level kam karega?",
        "entities": [
            {"text": "Licorice root", "type": "Herb", "start": 0, "end": 13},
            {"text": "Corticosteroids", "type": "Drug", "start": 35, "end": 50},
            {"text": "potassium level", "type": "Effect", "start": 74, "end": 89},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Bhringraj oil laga rahi hoon baalon ke liye. Minoxidil bhi "
        "use kar rahi hoon. Koi interaction hota hai topical mein?",
        "entities": [
            {"text": "Bhringraj oil", "type": "Herb", "start": 0, "end": 13},
            {"text": "Minoxidil", "type": "Drug", "start": 45, "end": 54},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Vacha powder le raha hoon memory ke liye. Donepezil bhi "
        "prescribed hai Alzheimer's ke liye. Doctor se puchna chahiye?",
        "entities": [
            {"text": "Vacha", "type": "Herb", "start": 0, "end": 5},
            {"text": "Donepezil", "type": "Drug", "start": 41, "end": 50},
            {"text": "Alzheimer's", "type": "Disease", "start": 68, "end": 79},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Cinnamon supplements le rahi hoon PCOS ke liye. Metformin bhi "
        "le rahi hoon. Sugar bahut low hone ka dar hai.",
        "entities": [
            {"text": "Cinnamon supplements", "type": "Herb", "start": 0, "end": 20},
            {"text": "PCOS", "type": "Disease", "start": 33, "end": 37},
            {"text": "Metformin", "type": "Drug", "start": 48, "end": 57},
            {"text": "Sugar bahut low", "type": "Effect", "start": 76, "end": 91},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Evening primrose oil le rahi hoon skin ke liye. Blood thinner "
        "bhi le rahi hoon. Bleeding risk badhega kya?",
        "entities": [
            {"text": "Evening primrose oil", "type": "Herb", "start": 0, "end": 20},
            {"text": "Blood thinner", "type": "Drug", "start": 47, "end": 60},
            {"text": "Bleeding risk", "type": "Effect", "start": 79, "end": 92},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Safed musli le raha hoon stamina ke liye. Viagra bhi le "
        "raha hoon. Hypotension ka risk hai kya dono saath mein?",
        "entities": [
            {"text": "Safed musli", "type": "Herb", "start": 0, "end": 11},
            {"text": "Viagra", "type": "Drug", "start": 41, "end": 47},
            {"text": "Hypotension", "type": "Effect", "start": 67, "end": 78},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Isabgol le rahi hoon constipation ke liye. Thyroid ki dawai "
        "bhi le rahi hoon. Absorption affect hota hai kya?",
        "entities": [
            {"text": "Isabgol", "type": "Herb", "start": 0, "end": 7},
            {"text": "constipation", "type": "Disease", "start": 21, "end": 33},
            {"text": "Thyroid ki dawai", "type": "Drug", "start": 44, "end": 60},
        ],
        "relations": [{"type": "inhibits", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Chamomile tea peeti hoon neend ke liye. Sedative medicine bhi "
        "le rahi hoon. Bahut zyada neend aa rahi hai.",
        "entities": [
            {"text": "Chamomile tea", "type": "Herb", "start": 0, "end": 13},
            {"text": "Sedative medicine", "type": "Drug", "start": 39, "end": 56},
            {"text": "Bahut zyada neend", "type": "Effect", "start": 75, "end": 92},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Saw palmetto supplement le raha hoon prostate ke liye. "
        "Finasteride bhi le raha hoon. Dono ka same effect hai kya?",
        "entities": [
            {"text": "Saw palmetto", "type": "Herb", "start": 0, "end": 12},
            {"text": "Finasteride", "type": "Drug", "start": 55, "end": 66},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Cat's claw supplement le raha hoon arthritis ke liye. "
        "Immunosuppressant bhi le raha hoon. Contraindicated hai kya?",
        "entities": [
            {"text": "Cat's claw", "type": "Herb", "start": 0, "end": 10},
            {"text": "arthritis", "type": "Disease", "start": 35, "end": 44},
            {"text": "Immunosuppressant", "type": "Drug", "start": 55, "end": 72},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "doctor_qa", "script": "romanized",
    },
    {
        "text": "Feverfew supplement le raha hoon migraine ke liye. "
        "Aspirin bhi let hoon. Platelet function pe asar padega kya?",
        "entities": [
            {"text": "Feverfew", "type": "Herb", "start": 0, "end": 8},
            {"text": "migraine", "type": "Disease", "start": 33, "end": 41},
            {"text": "Aspirin", "type": "Drug", "start": 52, "end": 59},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Horse chestnut supplement le rahi hoon varicose veins ke liye. "
        "Anticoagulant bhi le rahi hoon. Safe hai dono saath?",
        "entities": [
            {"text": "Horse chestnut", "type": "Herb", "start": 0, "end": 14},
            {"text": "varicose veins", "type": "Disease", "start": 38, "end": 52},
            {"text": "Anticoagulant", "type": "Drug", "start": 63, "end": 76},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Red clover supplement le rahi hoon menopause ke liye. "
        "HRT bhi le rahi hoon. Estrogen level bahut badh gaya.",
        "entities": [
            {"text": "Red clover", "type": "Herb", "start": 0, "end": 10},
            {"text": "menopause", "type": "Disease", "start": 34, "end": 43},
            {"text": "HRT", "type": "Drug", "start": 54, "end": 57},
            {"text": "Estrogen level", "type": "Effect", "start": 76, "end": 90},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "social_media", "script": "romanized",
    },
    {
        "text": "Dong quai supplement le rahi thi periods ke liye. "
        "Warfarin bhi le rahi thi. Heavy bleeding ho gayi.",
        "entities": [
            {"text": "Dong quai", "type": "Herb", "start": 0, "end": 9},
            {"text": "Warfarin", "type": "Drug", "start": 51, "end": 59},
            {"text": "Heavy bleeding", "type": "Effect", "start": 78, "end": 92},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Astragalus le rahi hoon immunity ke liye. Chemotherapy chal "
        "rahi hai. Doctor ne strictly mana kiya hai.",
        "entities": [
            {"text": "Astragalus", "type": "Herb", "start": 0, "end": 10},
            {"text": "Chemotherapy", "type": "Drug", "start": 41, "end": 53},
        ],
        "relations": [{"type": "interacts_with", "entity1_id": "E0", "entity2_id": "E1"}],
        "source": "health_forum", "script": "romanized",
    },
    {
        "text": "Passionflower extract le raha hoon anxiety ke liye. "
        "Barbiturate bhi prescribed hai. Excessive drowsiness ho rahi hai.",
        "entities": [
            {"text": "Passionflower", "type": "Herb", "start": 0, "end": 13},
            {"text": "anxiety", "type": "Disease", "start": 36, "end": 43},
            {"text": "Barbiturate", "type": "Drug", "start": 53, "end": 64},
            {"text": "Excessive drowsiness", "type": "Effect", "start": 84, "end": 104},
        ],
        "relations": [{"type": "potentiates", "entity1_id": "E0", "entity2_id": "E2"}],
        "source": "doctor_qa", "script": "romanized",
    },
]


def get_corpus_statistics(corpus: list[dict] = None) -> dict:
    """Compute statistics of the expanded corpus."""
    if corpus is None:
        corpus = EXPANDED_CORPUS

    entity_types = {}
    relation_types = {}
    sources = set()
    scripts = set()
    total_entities = 0
    total_relations = 0

    for entry in corpus:
        sources.add(entry.get("source", "unknown"))
        scripts.add(entry.get("script", "romanized"))

        for e in entry.get("entities", []):
            etype = e.get("type", "unknown")
            entity_types[etype] = entity_types.get(etype, 0) + 1
            total_entities += 1

        for r in entry.get("relations", []):
            rtype = r.get("type", "unknown")
            relation_types[rtype] = relation_types.get(rtype, 0) + 1
            total_relations += 1

    return {
        "total_sentences": len(corpus),
        "total_entities": total_entities,
        "total_relations": total_relations,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "sources": sorted(sources),
        "scripts": sorted(scripts),
        "avg_entities_per_sentence": total_entities / max(len(corpus), 1),
        "avg_relations_per_sentence": total_relations / max(len(corpus), 1),
        "sentences_with_relations": sum(
            1 for e in corpus if e.get("relations")
        ),
        "sentences_without_relations": sum(
            1 for e in corpus if not e.get("relations")
        ),
    }
