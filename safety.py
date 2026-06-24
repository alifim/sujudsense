from typing import List
from pydantic import BaseModel, Field

class QueryIntent(BaseModel):
    """Schema for the LLM intent classifier to enforce domain boundaries."""
    is_prayer_related: bool = Field(
        description="True if the query is about Islamic prayer positions like Sujud, Ruku, or Salah."
    )
    has_medical_or_mobility_context: bool = Field(
        description=(
            "True ONLY if the user explicitly mentions pain, injury, surgery, physical limitations, "
            "or joint protection (e.g., 'hurts', 'aches', 'herniation', 'protect my hips'). "
            "MUST BE FALSE for general posture goals, form checks, or 'how-to' tutorials "
            "(e.g., 'keep my spine flat', 'proper way to bend', 'where do my elbows go') "
            "UNLESS they explicitly state they are doing it to accommodate a specific pain or injury."
        )
    )

class SafetyPolicy:
    REFUSAL_PHRASE = (
        "I focus specifically on adapting prayer postures for physical pain, injuries, "
        "or mobility limitations. To help you safely, could you please share if you are "
        "experiencing any specific discomfort or injury?"
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

    MEDICAL_NOTICE = (
        "If you have severe or worsening pain, please consult a medical professional before trying any physical adjustments."
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
    def is_capability_query(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in cls.GENERAL_CAPABILITY_PATTERNS)

    @classmethod
    def should_block(cls, query: str) -> bool:
        return cls.is_off_topic(query)

    @classmethod
    def should_provide_capability_response(cls, query: str) -> bool:
        return cls.is_capability_query(query)
