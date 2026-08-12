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
            "True ONLY if the user is describing a physical bodily limitation, joint constraint, or biomechanical pain "
            "that specifically affects their physical movement or mechanics (e.g., 'knees hurt when bending', 'back surgery recovery', 'hip immobility'). "
            "MUST BE FALSE for general medical advice, dietary questions, or treatment plans (e.g., 'how to heal a torn ACL', 'foods for inflammation'). "
            "MUST BE FALSE for unrelated tasks, coding, or AI roleplay that happen to mention pain (e.g., 'write a Python script for knee pain', 'act as a doctor'). "
            "MUST BE FALSE for general posture goals or form checks without an underlying physical limitation (e.g., 'keep my spine flat', 'where do my elbows go')."
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

    @classmethod
    def is_obvious_mobility_adaptation(cls, query: str) -> bool:
        """Hardcoded bypass for clear mobility+prayer queries that strict classifier might miss."""
        query_lower = query.lower()
        
        # Pattern 1: chair/sitting adaptation + prayer term
        if ("chair" in query_lower or "sit" in query_lower) and cls._contains_prayer_terms(query):
            return True
        
        # Pattern 2: explicit body part + pain/injury + prayer
        body_parts = ["knee", "back", "shoulder", "hip", "wrist", "elbow", "neck", "ankle", "leg", "arm"]
        pain_terms = ["hurt", "pain", "surgery", "injury", "cannot", "unable", "stiffness", "limited"]
        has_body_pain = any(b in query_lower for b in body_parts) and any(p in query_lower for p in pain_terms)
        if has_body_pain and cls._contains_prayer_terms(query):
            return True
        
        return False
