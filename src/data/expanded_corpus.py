"""
Expanded Code-Mixed Health Corpus
Provides 150+ annotated sentences of Hindi-English code-mixed data.
"""

EXPANDED_CORPUS = \
[
    {
        "text": "Mera bhai Paracetamol le raha hai aur usne Tulsi shuru kiya. Ab usko heart palpitations ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 10,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 43,
                "end": 48,
                "id": "E1"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 69,
                "end": 87,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Safed Musli kha rahi hoon sugar control ke liye. Paracetamol bhi leti hoon. Kabhi kabhi nausea ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 49,
                "end": 60,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 88,
                "end": 94,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Paracetamol bhi leti hoon. Kabhi kabhi high bp ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 52,
                "end": 63,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 91,
                "end": 98,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Brahmi kha rahi hoon sugar control ke liye. Azithromycin bhi leti hoon. Kabhi kabhi nausea ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 44,
                "end": 56,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 84,
                "end": 90,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Neem kha rahi hoon sugar control ke liye. Insulin bhi leti hoon. Kabhi kabhi weakness ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 42,
                "end": 49,
                "id": "E1"
            },
            {
                "text": "weakness",
                "type": "Effect",
                "start": 77,
                "end": 85,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Levothyroxine le rahi hain. Kya Amla ka use safe hai Levothyroxine ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 34,
                "end": 47,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 66,
                "end": 70,
                "id": "E1"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 87,
                "end": 100,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Azithromycin le raha hai aur usne Amla shuru kiya. Ab usko liver pain ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 44,
                "end": 48,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 69,
                "end": 79,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Safed Musli ka extract Amlodipine ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 18,
                "end": 29,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 41,
                "end": 51,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Ashwagandha ka extract Metformin ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 18,
                "end": 29,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 41,
                "end": 50,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Tulsi capsules le raha hoon health ke liye. Kya ye safe hai Losartan ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 60,
                "end": 68,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Warfarin aur Brahmi ek saath le sakta hu? Mujhe dizziness ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 9,
                "end": 17,
                "id": "E0"
            },
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 22,
                "end": 28,
                "id": "E1"
            },
            {
                "text": "dizziness",
                "type": "Effect",
                "start": 57,
                "end": 66,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Metformin aur Amla ek saath le sakta hu? Mujhe nausea ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 9,
                "end": 18,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 23,
                "end": 27,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 56,
                "end": 62,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Ashwagandha juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Aspirin ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 84,
                "end": 91,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal tea peene se meri high bp badh gayi jab main Clopidogrel le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 33,
                "end": 40,
                "id": "E1"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 60,
                "end": 71,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Brahmi juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Omeprazole ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 79,
                "end": 89,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Omeprazole le raha hai aur usne Tulsi shuru kiya. Ab usko liver pain ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 10,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 42,
                "end": 47,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 68,
                "end": 78,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Shatavari ka kaadha pi rahi thi. Aur saath mein Amoxicillin bhi le rahi thi. headache hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 0,
                "end": 9,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 48,
                "end": 59,
                "id": "E1"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 77,
                "end": 85,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Ibuprofen aur Safed Musli ek saath le sakta hu? Mujhe stomach ache ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 9,
                "end": 18,
                "id": "E0"
            },
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 23,
                "end": 34,
                "id": "E1"
            },
            {
                "text": "stomach ache",
                "type": "Effect",
                "start": 63,
                "end": 75,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Amlodipine le raha hai aur usne Arjun ki chaal shuru kiya. Ab usko weakness ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 10,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 42,
                "end": 56,
                "id": "E1"
            },
            {
                "text": "weakness",
                "type": "Effect",
                "start": 77,
                "end": 85,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Amla mat lo Levothyroxine ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 15,
                "end": 19,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 27,
                "end": 40,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Ashwagandha ka extract Metformin ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 18,
                "end": 29,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 41,
                "end": 50,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Kya main Levothyroxine aur Ginger ek saath le sakta hu? Mujhe low sugar ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 9,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 27,
                "end": 33,
                "id": "E1"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 62,
                "end": 71,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Guggul ka kaadha pi rahi thi. Aur saath mein Metformin bhi le rahi thi. dizziness hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Guggul",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 45,
                "end": 54,
                "id": "E1"
            },
            {
                "text": "dizziness",
                "type": "Effect",
                "start": 72,
                "end": 81,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Ibuprofen le raha hai aur usne Jamun shuru kiya. Ab usko low sugar ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 10,
                "end": 19,
                "id": "E0"
            },
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 41,
                "end": 46,
                "id": "E1"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 67,
                "end": 76,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Neem tea peene se meri heart palpitations badh gayi jab main Aspirin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 23,
                "end": 41,
                "id": "E1"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 61,
                "end": 68,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Shatavari ka extract Amoxicillin ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 18,
                "end": 27,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 39,
                "end": 50,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Karela ka kaadha pi rahi thi. Aur saath mein Levothyroxine bhi le rahi thi. stomach ache hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 45,
                "end": 58,
                "id": "E1"
            },
            {
                "text": "stomach ache",
                "type": "Effect",
                "start": 76,
                "end": 88,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Jamun juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Omeprazole ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 78,
                "end": 88,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Mulethi mat lo Amoxicillin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Mulethi",
                "type": "Herb",
                "start": 15,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 30,
                "end": 41,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Safed Musli capsules le raha hoon health ke liye. Kya ye safe hai Levothyroxine ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 66,
                "end": 79,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Metformin le rahi hain. Kya Arjun ki chaal ka use safe hai Metformin ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 34,
                "end": 43,
                "id": "E0"
            },
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 62,
                "end": 76,
                "id": "E1"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 93,
                "end": 102,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Amla juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Amoxicillin ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 77,
                "end": 88,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Clopidogrel bhi leti hoon. Kabhi kabhi vomiting ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 52,
                "end": 63,
                "id": "E1"
            },
            {
                "text": "vomiting",
                "type": "Effect",
                "start": 91,
                "end": 99,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Levothyroxine aur Neem ek saath le sakta hu? Mujhe heart palpitations ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 9,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Neem",
                "type": "Herb",
                "start": 27,
                "end": 31,
                "id": "E1"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 60,
                "end": 78,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Losartan le raha hai aur usne Haldi shuru kiya. Ab usko bleeding ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 10,
                "end": 18,
                "id": "E0"
            },
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 40,
                "end": 45,
                "id": "E1"
            },
            {
                "text": "bleeding",
                "type": "Effect",
                "start": 66,
                "end": 74,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Metformin le rahi hain. Kya Arjun ki chaal ka use safe hai Metformin ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 34,
                "end": 43,
                "id": "E0"
            },
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 62,
                "end": 76,
                "id": "E1"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 93,
                "end": 102,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Metformin le rahi hain. Kya Ashwagandha ka use safe hai Metformin ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 34,
                "end": 43,
                "id": "E0"
            },
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 62,
                "end": 73,
                "id": "E1"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 90,
                "end": 99,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Ginger mat lo Warfarin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 15,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 29,
                "end": 37,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Tulsi ka extract Paracetamol ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 18,
                "end": 23,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 35,
                "end": 46,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Warfarin le raha hai aur usne Ashwagandha shuru kiya. Ab usko high bp ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 10,
                "end": 18,
                "id": "E0"
            },
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 40,
                "end": 51,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 72,
                "end": 79,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Safed Musli ka kaadha pi rahi thi. Aur saath mein Pantoprazole bhi le rahi thi. nausea hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 50,
                "end": 62,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 80,
                "end": 86,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Brahmi capsules le raha hoon health ke liye. Kya ye safe hai Atorvastatin ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Atorvastatin",
                "type": "Drug",
                "start": 61,
                "end": 73,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Atorvastatin le raha hai aur usne Karela shuru kiya. Ab usko acidity ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Atorvastatin",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Karela",
                "type": "Herb",
                "start": 44,
                "end": 50,
                "id": "E1"
            },
            {
                "text": "acidity",
                "type": "Effect",
                "start": 71,
                "end": 78,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Brahmi kha rahi hoon sugar control ke liye. Clopidogrel bhi leti hoon. Kabhi kabhi bleeding ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 44,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "bleeding",
                "type": "Effect",
                "start": 83,
                "end": 91,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Jamun mat lo Pantoprazole ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 15,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 28,
                "end": 40,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Brahmi ka kaadha pi rahi thi. Aur saath mein Omeprazole bhi le rahi thi. bleeding hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 45,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "bleeding",
                "type": "Effect",
                "start": 73,
                "end": 81,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Tulsi juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Warfarin ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 78,
                "end": 86,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Digoxin bhi leti hoon. Kabhi kabhi liver pain ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Digoxin",
                "type": "Drug",
                "start": 52,
                "end": 59,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 87,
                "end": 97,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Paracetamol aur Guggul ek saath le sakta hu? Mujhe nausea ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 9,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Guggul",
                "type": "Herb",
                "start": 25,
                "end": 31,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 60,
                "end": 66,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Ashwagandha ka kaadha pi rahi thi. Aur saath mein Levothyroxine bhi le rahi thi. stomach ache hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 50,
                "end": 63,
                "id": "E1"
            },
            {
                "text": "stomach ache",
                "type": "Effect",
                "start": 81,
                "end": 93,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Ginger mat lo Clopidogrel ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 15,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 29,
                "end": 40,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Safed Musli ka kaadha pi rahi thi. Aur saath mein Clopidogrel bhi le rahi thi. high bp hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 50,
                "end": 61,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 79,
                "end": 86,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Shatavari tea peene se meri headache badh gayi jab main Amoxicillin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 0,
                "end": 9,
                "id": "E0"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 28,
                "end": 36,
                "id": "E1"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 56,
                "end": 67,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mulethi juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Ibuprofen ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Mulethi",
                "type": "Herb",
                "start": 0,
                "end": 7,
                "id": "E0"
            },
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 80,
                "end": 89,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Atorvastatin le raha hai aur usne Shatavari shuru kiya. Ab usko low sugar ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Atorvastatin",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 44,
                "end": 53,
                "id": "E1"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 74,
                "end": 83,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Amla ka extract Losartan ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 18,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 34,
                "end": 42,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Ibuprofen le rahi hain. Kya Ginger ka use safe hai Ibuprofen ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 34,
                "end": 43,
                "id": "E0"
            },
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 62,
                "end": 68,
                "id": "E1"
            },
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 85,
                "end": 94,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Jamun kha rahi hoon sugar control ke liye. Aspirin bhi leti hoon. Kabhi kabhi liver pain ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 43,
                "end": 50,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 78,
                "end": 88,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Haldi kha rahi hoon sugar control ke liye. Metformin bhi leti hoon. Kabhi kabhi high bp ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 43,
                "end": 52,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 80,
                "end": 87,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal tea peene se meri headache badh gayi jab main Clopidogrel le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 33,
                "end": 41,
                "id": "E1"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 61,
                "end": 72,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Digoxin le raha hai aur usne Shatavari shuru kiya. Ab usko low sugar ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Digoxin",
                "type": "Drug",
                "start": 10,
                "end": 17,
                "id": "E0"
            },
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 39,
                "end": 48,
                "id": "E1"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 69,
                "end": 78,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Shatavari juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Digoxin ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 0,
                "end": 9,
                "id": "E0"
            },
            {
                "text": "Digoxin",
                "type": "Drug",
                "start": 82,
                "end": 89,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Tulsi juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Azithromycin ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 78,
                "end": 90,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Tulsi juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Omeprazole ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 78,
                "end": 88,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Neem capsules le raha hoon health ke liye. Kya ye safe hai Warfarin ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 59,
                "end": 67,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Jamun tea peene se meri low sugar badh gayi jab main Metformin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 24,
                "end": 33,
                "id": "E1"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 53,
                "end": 62,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Ashwagandha juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Amlodipine ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 84,
                "end": 94,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Haldi kha rahi hoon sugar control ke liye. Atorvastatin bhi leti hoon. Kabhi kabhi bleeding ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Atorvastatin",
                "type": "Drug",
                "start": 43,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "bleeding",
                "type": "Effect",
                "start": 83,
                "end": 91,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Ginger mat lo Azithromycin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 15,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 29,
                "end": 41,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Brahmi kha rahi hoon sugar control ke liye. Paracetamol bhi leti hoon. Kabhi kabhi stomach ache ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 44,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "stomach ache",
                "type": "Effect",
                "start": 83,
                "end": 95,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Arjun ki chaal ka extract Levothyroxine ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 18,
                "end": 32,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 44,
                "end": 57,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Clopidogrel le rahi hain. Kya Karela ka use safe hai Clopidogrel ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 34,
                "end": 45,
                "id": "E0"
            },
            {
                "text": "Karela",
                "type": "Herb",
                "start": 64,
                "end": 70,
                "id": "E1"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 87,
                "end": 98,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Safed Musli capsules le raha hoon health ke liye. Kya ye safe hai Warfarin ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 66,
                "end": 74,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Pantoprazole le rahi hain. Kya Karela ka use safe hai Pantoprazole ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 34,
                "end": 46,
                "id": "E0"
            },
            {
                "text": "Karela",
                "type": "Herb",
                "start": 65,
                "end": 71,
                "id": "E1"
            },
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 88,
                "end": 100,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Shatavari ka kaadha pi rahi thi. Aur saath mein Warfarin bhi le rahi thi. low sugar hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 0,
                "end": 9,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 48,
                "end": 56,
                "id": "E1"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 74,
                "end": 83,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Pantoprazole le raha hai aur usne Karela shuru kiya. Ab usko bleeding ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Karela",
                "type": "Herb",
                "start": 44,
                "end": 50,
                "id": "E1"
            },
            {
                "text": "bleeding",
                "type": "Effect",
                "start": 71,
                "end": 79,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Kya main Aspirin aur Tulsi ek saath le sakta hu? Mujhe headache ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 9,
                "end": 16,
                "id": "E0"
            },
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 21,
                "end": 26,
                "id": "E1"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 55,
                "end": 63,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Ginger juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Losartan ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 79,
                "end": 87,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Haldi capsules le raha hoon health ke liye. Kya ye safe hai Amlodipine ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 60,
                "end": 70,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Safed Musli ka kaadha pi rahi thi. Aur saath mein Clopidogrel bhi le rahi thi. nausea hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 50,
                "end": 61,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 79,
                "end": 85,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Jamun ka extract Insulin ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 18,
                "end": 23,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 35,
                "end": 42,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Kya main Clopidogrel aur Ginger ek saath le sakta hu? Mujhe nausea ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 9,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 25,
                "end": 31,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 60,
                "end": 66,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Amla capsules le raha hoon health ke liye. Kya ye safe hai Amoxicillin ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 59,
                "end": 70,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Amoxicillin aur Amla ek saath le sakta hu? Mujhe vomiting ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 9,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 25,
                "end": 29,
                "id": "E1"
            },
            {
                "text": "vomiting",
                "type": "Effect",
                "start": 58,
                "end": 66,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mulethi ka kaadha pi rahi thi. Aur saath mein Omeprazole bhi le rahi thi. high bp hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Mulethi",
                "type": "Herb",
                "start": 0,
                "end": 7,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 46,
                "end": 56,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 74,
                "end": 81,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Omeprazole le rahi hain. Kya Brahmi ka use safe hai Omeprazole ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 34,
                "end": 44,
                "id": "E0"
            },
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 63,
                "end": 69,
                "id": "E1"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 86,
                "end": 96,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Levothyroxine le rahi hain. Kya Tulsi ka use safe hai Levothyroxine ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 34,
                "end": 47,
                "id": "E0"
            },
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 66,
                "end": 71,
                "id": "E1"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 88,
                "end": 101,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Amla mat lo Amlodipine ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 15,
                "end": 19,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 27,
                "end": 37,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Insulin bhi leti hoon. Kabhi kabhi bleeding ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 52,
                "end": 59,
                "id": "E1"
            },
            {
                "text": "bleeding",
                "type": "Effect",
                "start": 87,
                "end": 95,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Karela tea peene se meri heart palpitations badh gayi jab main Digoxin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 25,
                "end": 43,
                "id": "E1"
            },
            {
                "text": "Digoxin",
                "type": "Drug",
                "start": 63,
                "end": 70,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Guggul ka extract Ibuprofen ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Guggul",
                "type": "Herb",
                "start": 18,
                "end": 24,
                "id": "E0"
            },
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 36,
                "end": 45,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Jamun kha rahi hoon sugar control ke liye. Clopidogrel bhi leti hoon. Kabhi kabhi nausea ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 43,
                "end": 54,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 82,
                "end": 88,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Shatavari ka extract Clopidogrel ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 18,
                "end": 27,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 39,
                "end": 50,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Guggul kha rahi hoon sugar control ke liye. Amoxicillin bhi leti hoon. Kabhi kabhi liver pain ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Guggul",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 44,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 83,
                "end": 93,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Shatavari kha rahi hoon sugar control ke liye. Metformin bhi leti hoon. Kabhi kabhi high bp ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 0,
                "end": 9,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 47,
                "end": 56,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 84,
                "end": 91,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Insulin le rahi hain. Kya Mulethi ka use safe hai Insulin ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 34,
                "end": 41,
                "id": "E0"
            },
            {
                "text": "Mulethi",
                "type": "Herb",
                "start": 60,
                "end": 67,
                "id": "E1"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 84,
                "end": 91,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Pantoprazole aur Amla ek saath le sakta hu? Mujhe liver pain ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 9,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 26,
                "end": 30,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 59,
                "end": 69,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Amlodipine le rahi hain. Kya Ashwagandha ka use safe hai Amlodipine ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 34,
                "end": 44,
                "id": "E0"
            },
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 63,
                "end": 74,
                "id": "E1"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 91,
                "end": 101,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Metformin aur Tulsi ek saath le sakta hu? Mujhe liver pain ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 9,
                "end": 18,
                "id": "E0"
            },
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 23,
                "end": 28,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 57,
                "end": 67,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Tulsi capsules le raha hoon health ke liye. Kya ye safe hai Losartan ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 60,
                "end": 68,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Haldi mat lo Losartan ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 15,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 28,
                "end": 36,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Omeprazole le raha hai aur usne Guggul shuru kiya. Ab usko acidity ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 10,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Guggul",
                "type": "Herb",
                "start": 42,
                "end": 48,
                "id": "E1"
            },
            {
                "text": "acidity",
                "type": "Effect",
                "start": 69,
                "end": 76,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Jamun mat lo Omeprazole ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 15,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 28,
                "end": 38,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal ka kaadha pi rahi thi. Aur saath mein Metformin bhi le rahi thi. stomach ache hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 53,
                "end": 62,
                "id": "E1"
            },
            {
                "text": "stomach ache",
                "type": "Effect",
                "start": 80,
                "end": 92,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Losartan le rahi hain. Kya Neem ka use safe hai Losartan ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 34,
                "end": 42,
                "id": "E0"
            },
            {
                "text": "Neem",
                "type": "Herb",
                "start": 61,
                "end": 65,
                "id": "E1"
            },
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 82,
                "end": 90,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Karela mat lo Paracetamol ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 15,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 29,
                "end": 40,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Pantoprazole le raha hai aur usne Amla shuru kiya. Ab usko vomiting ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 44,
                "end": 48,
                "id": "E1"
            },
            {
                "text": "vomiting",
                "type": "Effect",
                "start": 69,
                "end": 77,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Amoxicillin le raha hai aur usne Brahmi shuru kiya. Ab usko high bp ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 10,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 43,
                "end": 49,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 70,
                "end": 77,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Haldi ka kaadha pi rahi thi. Aur saath mein Losartan bhi le rahi thi. vomiting hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Losartan",
                "type": "Drug",
                "start": 44,
                "end": 52,
                "id": "E1"
            },
            {
                "text": "vomiting",
                "type": "Effect",
                "start": 70,
                "end": 78,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Amlodipine le rahi hain. Kya Karela ka use safe hai Amlodipine ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 34,
                "end": 44,
                "id": "E0"
            },
            {
                "text": "Karela",
                "type": "Herb",
                "start": 63,
                "end": 69,
                "id": "E1"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 86,
                "end": 96,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Ibuprofen bhi leti hoon. Kabhi kabhi low sugar ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 52,
                "end": 61,
                "id": "E1"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 89,
                "end": 98,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Neem juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Ibuprofen ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 77,
                "end": 86,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Arjun ki chaal mat lo Amlodipine ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 15,
                "end": 29,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 37,
                "end": 47,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Levothyroxine bhi leti hoon. Kabhi kabhi vomiting ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 52,
                "end": 65,
                "id": "E1"
            },
            {
                "text": "vomiting",
                "type": "Effect",
                "start": 93,
                "end": 101,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Brahmi mat lo Paracetamol ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Brahmi",
                "type": "Herb",
                "start": 15,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 29,
                "end": 40,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Warfarin bhi leti hoon. Kabhi kabhi nausea ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 52,
                "end": 60,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 88,
                "end": 94,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Giloy mat lo Aspirin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Giloy",
                "type": "Herb",
                "start": 15,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 28,
                "end": 35,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Amoxicillin le raha hai aur usne Mulethi shuru kiya. Ab usko stomach ache ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 10,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Mulethi",
                "type": "Herb",
                "start": 43,
                "end": 50,
                "id": "E1"
            },
            {
                "text": "stomach ache",
                "type": "Effect",
                "start": 71,
                "end": 83,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Amla ka kaadha pi rahi thi. Aur saath mein Metformin bhi le rahi thi. high bp hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 43,
                "end": 52,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 70,
                "end": 77,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Neem mat lo Aspirin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 15,
                "end": 19,
                "id": "E0"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 27,
                "end": 34,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal kha rahi hoon sugar control ke liye. Aspirin bhi leti hoon. Kabhi kabhi acidity ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 52,
                "end": 59,
                "id": "E1"
            },
            {
                "text": "acidity",
                "type": "Effect",
                "start": 87,
                "end": 94,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Azithromycin le raha hai aur usne Neem shuru kiya. Ab usko low sugar ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Neem",
                "type": "Herb",
                "start": 44,
                "end": 48,
                "id": "E1"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 69,
                "end": 78,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Pantoprazole le rahi hain. Kya Amla ka use safe hai Pantoprazole ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 34,
                "end": 46,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 65,
                "end": 69,
                "id": "E1"
            },
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 86,
                "end": 98,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Jamun mat lo Azithromycin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 15,
                "end": 20,
                "id": "E0"
            },
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 28,
                "end": 40,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Karela tea peene se meri dizziness badh gayi jab main Warfarin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "dizziness",
                "type": "Effect",
                "start": 25,
                "end": 34,
                "id": "E1"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 54,
                "end": 62,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Giloy tea peene se meri heart palpitations badh gayi jab main Azithromycin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Giloy",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 24,
                "end": 42,
                "id": "E1"
            },
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 62,
                "end": 74,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Karela kha rahi hoon sugar control ke liye. Amoxicillin bhi leti hoon. Kabhi kabhi high bp ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 44,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 83,
                "end": 90,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Karela capsules le raha hoon health ke liye. Kya ye safe hai Amlodipine ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 61,
                "end": 71,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Amlodipine aur Arjun ki chaal ek saath le sakta hu? Mujhe liver pain ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 9,
                "end": 19,
                "id": "E0"
            },
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 24,
                "end": 38,
                "id": "E1"
            },
            {
                "text": "liver pain",
                "type": "Effect",
                "start": 67,
                "end": 77,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Shatavari ka kaadha pi rahi thi. Aur saath mein Metformin bhi le rahi thi. heart palpitations hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 0,
                "end": 9,
                "id": "E0"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 48,
                "end": 57,
                "id": "E1"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 75,
                "end": 93,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Amla ka kaadha pi rahi thi. Aur saath mein Ibuprofen bhi le rahi thi. heart palpitations hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 43,
                "end": 52,
                "id": "E1"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 70,
                "end": 88,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Neem tea peene se meri weakness badh gayi jab main Azithromycin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "weakness",
                "type": "Effect",
                "start": 23,
                "end": 31,
                "id": "E1"
            },
            {
                "text": "Azithromycin",
                "type": "Drug",
                "start": 51,
                "end": 63,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Tulsi tea peene se meri headache badh gayi jab main Warfarin le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 24,
                "end": 32,
                "id": "E1"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 52,
                "end": 60,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Jamun capsules le raha hoon health ke liye. Kya ye safe hai Aspirin ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Jamun",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 60,
                "end": 67,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Metformin le rahi hain. Kya Amla ka use safe hai Metformin ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 34,
                "end": 43,
                "id": "E0"
            },
            {
                "text": "Amla",
                "type": "Herb",
                "start": 62,
                "end": 66,
                "id": "E1"
            },
            {
                "text": "Metformin",
                "type": "Drug",
                "start": 83,
                "end": 92,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Tulsi kha rahi hoon sugar control ke liye. Omeprazole bhi leti hoon. Kabhi kabhi nausea ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Tulsi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 43,
                "end": 53,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 81,
                "end": 87,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Shatavari ka kaadha pi rahi thi. Aur saath mein Insulin bhi le rahi thi. weakness hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 0,
                "end": 9,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 48,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "weakness",
                "type": "Effect",
                "start": 73,
                "end": 81,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Neem kha rahi hoon sugar control ke liye. Amoxicillin bhi leti hoon. Kabhi kabhi headache ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Amoxicillin",
                "type": "Drug",
                "start": 42,
                "end": 53,
                "id": "E1"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 81,
                "end": 89,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Neem ka extract Levothyroxine ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 18,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 34,
                "end": 47,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Safed Musli capsules le raha hoon health ke liye. Kya ye safe hai Paracetamol ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 66,
                "end": 77,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Karela ka kaadha pi rahi thi. Aur saath mein Clopidogrel bhi le rahi thi. high bp hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 45,
                "end": 56,
                "id": "E1"
            },
            {
                "text": "high bp",
                "type": "Effect",
                "start": 74,
                "end": 81,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Mulethi mat lo Insulin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Mulethi",
                "type": "Herb",
                "start": 15,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 30,
                "end": 37,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Atorvastatin le raha hai aur usne Ashwagandha shuru kiya. Ab usko headache ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Atorvastatin",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Ashwagandha",
                "type": "Herb",
                "start": 44,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 76,
                "end": 84,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Atorvastatin le raha hai aur usne Neem shuru kiya. Ab usko headache ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Atorvastatin",
                "type": "Drug",
                "start": 10,
                "end": 22,
                "id": "E0"
            },
            {
                "text": "Neem",
                "type": "Herb",
                "start": 44,
                "end": 48,
                "id": "E1"
            },
            {
                "text": "headache",
                "type": "Effect",
                "start": 69,
                "end": 77,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Mera bhai Ibuprofen le raha hai aur usne Arjun ki chaal shuru kiya. Ab usko vomiting ho raha hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ibuprofen",
                "type": "Drug",
                "start": 10,
                "end": 19,
                "id": "E0"
            },
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 41,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "vomiting",
                "type": "Effect",
                "start": 76,
                "end": 84,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Ginger mat lo Aspirin ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 15,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Aspirin",
                "type": "Drug",
                "start": 29,
                "end": 36,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Haldi kha rahi hoon sugar control ke liye. Levothyroxine bhi leti hoon. Kabhi kabhi dizziness ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Levothyroxine",
                "type": "Drug",
                "start": 43,
                "end": 56,
                "id": "E1"
            },
            {
                "text": "dizziness",
                "type": "Effect",
                "start": 84,
                "end": 93,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Safed Musli ka kaadha pi rahi thi. Aur saath mein Warfarin bhi le rahi thi. stomach ache hoti thi.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 0,
                "end": 11,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 50,
                "end": 58,
                "id": "E1"
            },
            {
                "text": "stomach ache",
                "type": "Effect",
                "start": 76,
                "end": 88,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Ginger ka extract Insulin ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Ginger",
                "type": "Herb",
                "start": 18,
                "end": 24,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 36,
                "end": 43,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Doctor ne bola Arjun ki chaal mat lo Omeprazole ke saath. Interaction hota hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 15,
                "end": 29,
                "id": "E0"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 37,
                "end": 47,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Giloy juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Amlodipine ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Giloy",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 78,
                "end": 88,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Neem kha rahi hoon sugar control ke liye. Warfarin bhi leti hoon. Kabhi kabhi dizziness ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Neem",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Warfarin",
                "type": "Drug",
                "start": 42,
                "end": 50,
                "id": "E1"
            },
            {
                "text": "dizziness",
                "type": "Effect",
                "start": 78,
                "end": 87,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Karela tea peene se meri low sugar badh gayi jab main Clopidogrel le raha tha.",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "low sugar",
                "type": "Effect",
                "start": 25,
                "end": 34,
                "id": "E1"
            },
            {
                "text": "Clopidogrel",
                "type": "Drug",
                "start": 54,
                "end": 65,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E2"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Kya main Pantoprazole aur Safed Musli ek saath le sakta hu? Mujhe nausea ki problem hai.",
        "language_tags": [
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi"
        ],
        "entities": [
            {
                "text": "Pantoprazole",
                "type": "Drug",
                "start": 9,
                "end": 21,
                "id": "E0"
            },
            {
                "text": "Safed Musli",
                "type": "Herb",
                "start": 26,
                "end": 37,
                "id": "E1"
            },
            {
                "text": "nausea",
                "type": "Effect",
                "start": 66,
                "end": 72,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Haldi juice daily pee rahi hoon. Blood test mein problem aayi. Doctor ne bola Digoxin ke saath mat lo.",
        "language_tags": [
            "en",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 0,
                "end": 5,
                "id": "E0"
            },
            {
                "text": "Digoxin",
                "type": "Drug",
                "start": 78,
                "end": 85,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    },
    {
        "text": "Karela kha rahi hoon sugar control ke liye. Paracetamol bhi leti hoon. Kabhi kabhi heart palpitations ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Karela",
                "type": "Herb",
                "start": 0,
                "end": 6,
                "id": "E0"
            },
            {
                "text": "Paracetamol",
                "type": "Drug",
                "start": 44,
                "end": 55,
                "id": "E1"
            },
            {
                "text": "heart palpitations",
                "type": "Effect",
                "start": 83,
                "end": 101,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Arjun ki chaal capsules le raha hoon health ke liye. Kya ye safe hai Amlodipine ke saath?",
        "language_tags": [
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Arjun ki chaal",
                "type": "Herb",
                "start": 0,
                "end": 14,
                "id": "E0"
            },
            {
                "text": "Amlodipine",
                "type": "Drug",
                "start": 69,
                "end": 79,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Meri mummy ko diabetes hai aur wo Omeprazole le rahi hain. Kya Shatavari ka use safe hai Omeprazole ke saath?",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 34,
                "end": 44,
                "id": "E0"
            },
            {
                "text": "Shatavari",
                "type": "Herb",
                "start": 63,
                "end": 72,
                "id": "E1"
            },
            {
                "text": "Omeprazole",
                "type": "Drug",
                "start": 89,
                "end": 99,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E1",
                "entity2_id": "E0"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Amla kha rahi hoon sugar control ke liye. Insulin bhi leti hoon. Kabhi kabhi dizziness ho jata hai.",
        "language_tags": [
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Amla",
                "type": "Herb",
                "start": 0,
                "end": 4,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 42,
                "end": 49,
                "id": "E1"
            },
            {
                "text": "dizziness",
                "type": "Effect",
                "start": 77,
                "end": 86,
                "id": "E2"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "health_forum",
        "script": "romanized"
    },
    {
        "text": "Maine suna hai ki Haldi ka extract Insulin ke saath nahi lena chahiye.",
        "language_tags": [
            "hi",
            "hi",
            "hi",
            "hi",
            "en",
            "hi",
            "en",
            "en",
            "hi",
            "hi",
            "hi",
            "hi",
            "hi"
        ],
        "entities": [
            {
                "text": "Haldi",
                "type": "Herb",
                "start": 18,
                "end": 23,
                "id": "E0"
            },
            {
                "text": "Insulin",
                "type": "Drug",
                "start": 35,
                "end": 42,
                "id": "E1"
            }
        ],
        "relations": [
            {
                "type": "interacts_with",
                "entity1_id": "E0",
                "entity2_id": "E1"
            }
        ],
        "source": "social_media",
        "script": "romanized"
    }
]

def get_corpus_statistics():
    return {
        "total_sentences": len(EXPANDED_CORPUS),
        "total_entities": sum(len(e["entities"]) for e in EXPANDED_CORPUS),
        "total_relations": sum(len(e["relations"]) for e in EXPANDED_CORPUS)
    }
