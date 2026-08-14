from typing import List
from pydantic import BaseModel, Field

class QueryIntent(BaseModel):
    """Schema for the LLM intent classifier to enforce domain boundaries."""
    reasoning: str
    is_prayer_related: bool
    is_valid_mobility_adaptation_request: bool

# Minimal schema for API call — no descriptions
_MINIMAL_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "is_prayer_related": {"type": "boolean"},
        "is_valid_mobility_adaptation_request": {"type": "boolean"}
    },
    "required": ["reasoning", "is_prayer_related", "is_valid_mobility_adaptation_request"],
    "additionalProperties": False
}

# Put descriptions in SYSTEM PROMPT instead of schema
_INTENT_SYSTEM_PROMPT = """You are an intent classifier for SujudSense, an Islamic prayer guidance app.

Your task: analyze the query and return JSON with:
- reasoning: brief analysis
- is_prayer_related: true if about Islamic prayer positions (Ruku, Sujud, Salah, etc.)
- is_valid_mobility_adaptation_request: true ONLY if the user DESCRIBES a physical bodily limitation, joint constraint, or biomechanical pain affecting movement

CRITICAL DISTINCTION:
- DESCRIBING symptoms/limitations = True (even without prayer words): "knees hurt when bending", "hard time bending due to stiffness", "back surgery recovery", "hip immobility"
- ASKING for treatment/healing/cure = False: "how to heal a torn ACL", "what should I do to fix this", "best foods for inflammation", "what surgery do I need"

The user may not mention prayer explicitly. If they describe a genuine movement limitation, it IS a valid mobility adaptation request.

EXAMPLES:
- "My knees hurt when I bend" -> is_valid_mobility_adaptation_request: true
- "I had back surgery and can't bow" -> is_valid_mobility_adaptation_request: true  
- "Hard time bending my knees due to stiffness" -> is_valid_mobility_adaptation_request: true
- "How to heal a torn ACL" -> is_valid_mobility_adaptation_request: false
- "What foods reduce inflammation" -> is_valid_mobility_adaptation_request: false
- "Where should I place my elbows in Sujud" -> is_valid_mobility_adaptation_request: false

Respond with valid JSON only. No explanations outside JSON."""

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
        "If you have severe or worsening pain, please consult a medical professional before trying any physical adjustments.\n\n"
        "**Sources:**\n"
        "- [Biomechanics: Nazish & Kalra (2018)](https://www.ijhsr.org/IJHSR_Vol.8_Issue.7_July2018/43.pdf)\n"
        "- [Fiqh: GAIAE (2015)](https://islamiceducationinuae.wordpress.com/wp-content/uploads/2018/12/Salatul-Mareed_Prayer-of-the-Sick.pdf)"
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
        if ("chair" in query_lower or "sitting" in query_lower) and cls._contains_prayer_terms(query):
            return True
        
        # Pattern 2: explicit body part + pain/injury + prayer
        body_parts = ["knee", "back", "shoulder", "hip", "wrist", "elbow", "neck", "ankle", "leg", "arm"]
        pain_terms = ["hurt", "pain", "surgery", "injury", "cannot", "unable", "stiffness", "limited"]
        has_body_pain = any(b in query_lower for b in body_parts) and any(p in query_lower for p in pain_terms)
        if has_body_pain and cls._contains_prayer_terms(query):
            return True
        
        return False
