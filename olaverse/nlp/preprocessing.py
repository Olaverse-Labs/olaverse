import re

# Pidgin particles/words list to preserve during preprocessing/stopword-filtering
PIDGIN_PARTICLES = {
    "sha", "sef", "abeg", "na", "dey", "wetin", "oo", "ooo", "o", "go", "no",
    "comot", "chook", "yanga", "wahala", "awuf", "jara", "mumu", "kolo", "waka",
    "gbagbe", "shey", "abi", "joor", "kwanu", "nna", "shuu", "abeg", "shakara",
    "lamba", "gist"
}

# Regex compilation for performance
# Matches +2348031234567, 2348031234567, 08031234567, 0803 123 4567, 0803-123-4567
PHONE_REGEX = re.compile(
    r'\b(?:\+?234|0)[789][01]\d(?:[ \-]?\d){7}\b'
)

EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Match 11-digit numbers with BVN/NIN context
BVN_CONTEXT_REGEX = re.compile(
    r'(?i)\b(?:bvn|bank\s+verification\s+number)\b\s*(?:is)?\s*(?::)?\s*(\b\d{11}\b)'
)

NIN_CONTEXT_REGEX = re.compile(
    r'(?i)\b(?:nin|national\s+identification\s+number|national\s+id)\b\s*(?:is)?\s*(?::)?\s*(\b\d{11}\b)'
)

# Standard 11 digit fallback if not matched by phone number or context
GENERIC_ID_REGEX = re.compile(
    r'\b\d{11}\b'
)

def mask_pii(text):
    """
    Mask Nigerian PII patterns in text.
    Replaces phone numbers with [PHONE], emails with [EMAIL], BVN with [BVN], and NIN with [NIN].
    """
    if not text or not isinstance(text, str):
        return text

    # Helper to mask a specific span
    # We do replacements in steps, taking care not to overwrite already masked tokens.
    
    # 1. Mask Email
    text = EMAIL_REGEX.sub("[EMAIL]", text)
    
    # 2. Mask BVN context-based
    def replace_bvn(match):
        full_match = match.group(0)
        num = match.group(1)
        return full_match.replace(num, "[BVN]")
    text = BVN_CONTEXT_REGEX.sub(replace_bvn, text)
    
    # 3. Mask NIN context-based
    def replace_nin(match):
        full_match = match.group(0)
        num = match.group(1)
        return full_match.replace(num, "[NIN]")
    text = NIN_CONTEXT_REGEX.sub(replace_nin, text)
    
    # 4. Mask Phone numbers (must be done carefully to avoid matching BVN/NIN numbers, but phone numbers start with specific prefixes)
    # We find all matching phone numbers and replace them.
    # Note: phone numbers are 11 digits starting with 070, 080, 090, 081, 091, 070 etc., or international equivalents (+234...).
    text = PHONE_REGEX.sub("[PHONE]", text)
    
    # 5. Mask any remaining 11 digit numbers that weren't masked by phone or specific BVN/NIN context
    # We default them to [ID] or if "bvn" is in the whole text we can guess [BVN], otherwise we call it [ID].
    def replace_generic_id(match):
        num = match.group(0)
        # Check if the surrounding context (entire text) mentions BVN or NIN
        lower_text = text.lower()
        if "bvn" in lower_text:
            return "[BVN]"
        elif "nin" in lower_text:
            return "[NIN]"
        return "[ID]"
        
    text = GENERIC_ID_REGEX.sub(replace_generic_id, text)

    return text

def is_pidgin_particle(word):
    """
    Check if a word is a common Nigerian Pidgin particle/word.
    """
    if not word or not isinstance(word, str):
        return False
    return word.strip().lower() in PIDGIN_PARTICLES
