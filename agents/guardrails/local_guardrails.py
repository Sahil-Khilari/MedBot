import re
from langchain_core.messages import AIMessage

_BLOCKED_RESPONSE = (
    "I'm a medical assistant and can only help with health and medical questions. "
    "Please ask me something related to health, medicine, symptoms, treatments, or medical conditions."
)

# Prompt injection / jailbreak — always block regardless of medical content
_INJECTION_PATTERNS = [
    r'(?i)(ignore (previous|all) instructions|you are now|jailbreak|pretend you are|act as if you)',
    r'<script|javascript:|on\w+\s*=',
    r'(?i)\bhow to (make|build|synthesize) (bomb|weapon|explosive|poison|meth)\b',
]

# Medical keywords — query must contain at least one of these to be allowed
_MEDICAL_KEYWORDS = re.compile(
    r'\b(symptom|disease|diagnosis|diagnose|treatment|treat|medication|medicine|therapy|surgery|'
    r'cancer|tumor|tumour|diabetes|hypertension|infection|vaccine|anatomy|pathology|'
    r'prescription|dose|dosage|side.?effect|physician|doctor|hospital|clinic|patient|'
    r'health|medical|illness|disorder|syndrome|chronic|acute|emergency|first.?aid|'
    r'pain|fever|cough|rash|allerg|blood.?pressure|blood.?sugar|blood.?test|'
    r'heart|lung|kidney|liver|immune|virus|bacteria|fungal|parasite|'
    r'wound|fracture|bone|muscle|nerve|brain|spine|skin|lesion|'
    r'biopsy|radiology|x.?ray|mri|ct.?scan|ultrasound|ecg|ekg|'
    r'stroke|asthma|arthritis|cholesterol|obesity|mental.?health|depression|anxiety|'
    r'nausea|vomit|diarrhea|diarrhoea|headache|migraine|fatigue|insomnia|dizziness|'
    r'swelling|inflammation|infection|abscess|ulcer|bleeding|bruise|'
    r'nutrition|diet|calorie|vitamin|supplement|weight.?loss|weight.?gain|'
    r'pregnant|pregnancy|childbirth|pediatric|geriatric|oncology|cardiology|'
    r'neurology|orthopedic|ophthalmology|dermatology|psychiatry|'
    r'antibiotic|antiviral|antifungal|painkiller|inhaler|insulin|chemotherapy|'
    r'cure|prevention|prognosis|relapse|remission|chronic|terminal|benign|malignant)\b',
    re.IGNORECASE
)

# Short greetings / meta-questions — allow through without requiring medical keywords
_GREETING_PATTERNS = re.compile(
    r'^(hi|hello|hey|good\s+(morning|afternoon|evening|day)|'
    r'thanks|thank\s+you|okay|ok|sure|got\s+it|'
    r'bye|goodbye|see\s+you|'
    r'yes|no|yep|nope|'
    r'what\s+can\s+you\s+(do|help)|'
    r'tell\s+me\s+about\s+yourself|who\s+are\s+you|what\s+are\s+you)[^\w]*$',
    re.IGNORECASE
)


class LocalGuardrails:
    """Strict keyword-based guardrails.

    Policy: a query is allowed ONLY if it contains at least one medical keyword
    OR is a short greeting/meta-question. Everything else is blocked.
    """

    def __init__(self, llm):
        self.llm = llm  # kept for interface compatibility

    def check_input(self, user_input: str) -> tuple[bool, str]:
        text = user_input.strip()

        # Always block prompt injection / jailbreak attempts first
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, AIMessage(content=_BLOCKED_RESPONSE)

        # Allow simple greetings and meta-questions
        if _GREETING_PATTERNS.match(text):
            return True, user_input

        # STRICT RULE: must contain at least one medical keyword
        if _MEDICAL_KEYWORDS.search(text):
            return True, user_input

        # Everything else is blocked — no medical term found
        print(f"[Guardrails] Blocked non-medical query: '{text[:80]}'")
        return False, AIMessage(content=_BLOCKED_RESPONSE)

    def check_output(self, output: str, user_input: str = "") -> str:
        if not output:
            return output
        return output if isinstance(output, str) else output.content