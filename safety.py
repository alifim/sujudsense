from typing import List
from pydantic import BaseModel, Field

class QueryIntent(BaseModel):
    """Schema for the LLM intent classifier to enforce domain boundaries."""
    reasoning: str = Field(
        description=(
            "Briefly analyze the user's intent. Determine if the query is a genuine request "
            "for physical movement adaptation, or if it is a disallowed category like general "
            "medical advice, coding, or roleplay."
        )
    )
    is_prayer_related: bool = Field(
        description="True if the query is about Islamic prayer positions like Sujud, Ruku, or Salah."
    )
    is_valid_mobility_adaptation_request: bool = Field(
        description=(
            "True if the user is asking about prayer posture adjustments due to physical pain, injury, "
            "surgery, mobility limitation, or any bodily constraint that affects prayer movements. "
            "This includes: direct descriptions of pain ('knee hurts', 'back pain'), questions about "
            "when adaptations are permitted ('when should I sit on a chair'), and requests for guidance "
            "on modified postures for physical conditions. "
            "MUST BE FALSE for general medical advice unrelated to prayer (e.g., 'how to heal a torn ACL', "
            "'foods for inflammation'), coding tasks, or AI roleplay. "
            "MUST BE FALSE for general religious knowledge without physical limitation (e.g., 'how many rakahs in Fajr')."
        )
    )

class SafetyPolicy:
    ERROR_PHRASE = (
        "I'm experiencing a temporary technical issue "
        "and cannot safely analyze your posture request right now. "
        "Please try asking again in a moment."
    )

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
        "your capabilities",
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
    def should_block(cls, query: str) -> bool:
        query_lower = query.lower()
        if any(pattern in query_lower for pattern in cls.OFF_TOPIC_PATTERNS):
            return True
        if cls._contains_medical_terms(query) and not cls._contains_prayer_terms(query):
            return True
        return False

    @classmethod
    def should_provide_capability_response(cls, query: str) -> bool:
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in cls.GENERAL_CAPABILITY_PATTERNS)
