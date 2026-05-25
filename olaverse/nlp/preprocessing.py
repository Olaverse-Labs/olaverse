import re

# Regex compilation for performance
EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Standard E.164 and international phone number formats
PHONE_REGEX = re.compile(
    r'\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b'
)

# Standard Credit Card matching (13-19 digits, optionally separated by spaces or dashes)
CREDIT_CARD_REGEX = re.compile(
    r'\b(?:\d[ -]*?){13,19}\b'
)

# Standard US Social Security Number (SSN) format
SSN_REGEX = re.compile(
    r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'
)

# URL matching
URL_REGEX = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
)

# HTML tags matching
HTML_REGEX = re.compile(r'<[^>]*>')


def mask_pii(text: str) -> str:
    """
    Mask general Personally Identifiable Information (PII) globally.
    Replaces Emails, Credit Cards, Social Security Numbers (SSN), and Phone numbers.
    """
    if not text or not isinstance(text, str):
        return text

    # Apply masks sequentially
    # It's generally best to do longer sequences first
    text = EMAIL_REGEX.sub("[EMAIL]", text)
    
    # Simple check for SSN vs CC (if overlapping, CC usually longer)
    # However, the generic nature of these regexes means we just replace whatever hits
    text = SSN_REGEX.sub("[SSN]", text)
    text = CREDIT_CARD_REGEX.sub("[CREDIT_CARD]", text)
    text = PHONE_REGEX.sub("[PHONE]", text)

    return text

def clean_text(text: str, remove_urls: bool = True, remove_html: bool = True) -> str:
    """
    General text cleaning utility.
    Strips extra whitespaces, and optionally removes URLs and HTML tags.
    """
    if not text or not isinstance(text, str):
        return text
        
    if remove_html:
        text = HTML_REGEX.sub(" ", text)
        
    if remove_urls:
        text = URL_REGEX.sub("", text)
        
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
