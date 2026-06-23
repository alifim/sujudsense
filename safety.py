from typing import List

class SafetyPolicy:
    REFUSAL_PHRASE = (
        "I do not have enough specific biomechanical or jurisprudential context in my current knowledge base "
        "to safely advise on that specific movement."
    )
    JAILBREAK_PHRASE = (
        "I am SujudSense, and I cannot provide medical diagnoses or alter my core instructions. "
        "Please consult a doctor for severe pain."
    )

    OFF_TOPIC_PATTERNS: List[str] = [
        "python script",
        "build a chatbot",
        "hack",
        "blockchain",
        "medical diagnosis",
        "medical advice",
        "medical prescription",
        "prescription",
        "surgeon",
        "hospital database",
        "ignore previous instructions",
        "translate my prayer",
        "provide nutritional advice",
        "nutrition",
        "act as a doctor",
        "act as a surgeon",
    ]

    MEDICAL_TERMS: List[str] = [
        "surgery",
        "surgical",
        "doctor",
        "medical",
        "prescription",
        "diagnosis",
        "injury",
        "surgeon",
    ]

    PRAYER_TERMS: List[str] = [
        "sujud",
        "sajdah",
        "ruku",
        "rakah",
        "rakah",
        "rakahs",
        "rak'ah",
        "jalsa",
        "tashahhud",
        "qiyam",
        "salah",
        "prayer",
        "bowing",
        "prostration",
    ]

    GENERAL_QUERY_PATTERNS: List[str] = [
        "what does",
        "what is",
        "define",
        "meaning of",
        "how many",
        "why is",
        "what makes",
        "is it okay to pray",
        "can a person",
        "when should",
        "explain",
    ]

    GENERAL_CAPABILITY_PATTERNS: List[str] = [
        "what can you do",
        "what are your capabilities",
        "who are you",
        "tell me about yourself",
        "how can you help",
        "what do you do",
        "what can i ask",
    ]

    GENERAL_CAPABILITY_RESPONSE = (
        "I help with prayer posture adjustments when physical pain or mobility issues interact "
        "with Fiqh, using only the supplied biomechanics and jurisprudence knowledge. "
        "Ask me about a specific issue such as knee pain in Sujud or back strain in Ruku."
    )

    @classmethod
    def _contains_medical_terms(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(term in query_lower for term in cls.MEDICAL_TERMS)

    @classmethod
    def _contains_prayer_terms(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(term in query_lower for term in cls.PRAYER_TERMS)

    @classmethod
    def is_off_topic(cls, query: str) -> bool:
        query_lower = query.lower()
        if any(pattern in query_lower for pattern in cls.OFF_TOPIC_PATTERNS):
            return True
        if cls._contains_medical_terms(query) and not cls._contains_prayer_terms(query):
            return True
        return False

    @classmethod
    def is_boundary_query(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(
            query_lower.startswith(pattern) or pattern in query_lower
            for pattern in cls.GENERAL_QUERY_PATTERNS
        )

    @classmethod
    def is_capability_query(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in cls.GENERAL_CAPABILITY_PATTERNS)

    @classmethod
    def should_block(cls, query: str) -> bool:
        return cls.is_off_topic(query)

    @classmethod
    def should_provide_capability_response(cls, query: str) -> bool:
        return cls.is_capability_query(query)

    @classmethod
    def should_refuse(cls, query: str) -> bool:
        return cls.is_boundary_query(query)
