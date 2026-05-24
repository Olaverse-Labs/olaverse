# Constants and utilities for Nigerian context

STATES = {
    "Abia": "Umuahia",
    "Adamawa": "Yola",
    "Akwa Ibom": "Uyo",
    "Anambra": "Awka",
    "Bauchi": "Bauchi",
    "Bayelsa": "Yenagoa",
    "Benue": "Makurdi",
    "Borno": "Maiduguri",
    "Cross River": "Calabar",
    "Delta": "Asaba",
    "Ebonyi": "Abakaliki",
    "Edo": "Benin City",
    "Ekiti": "Ado Ekiti",
    "Enugu": "Enugu",
    "Gombe": "Gombe",
    "Imo": "Owerri",
    "Jigawa": "Dutse",
    "Kaduna": "Kaduna",
    "Kano": "Kano",
    "Katsina": "Katsina",
    "Kebbi": "Birnin Kebbi",
    "Kogi": "Lokoja",
    "Kwara": "Ilorin",
    "Lagos": "Ikeja",
    "Nasarawa": "Lafia",
    "Niger": "Minna",
    "Ogun": "Abeokuta",
    "Ondo": "Akure",
    "Osun": "Oshogbo",
    "Oyo": "Ibadan",
    "Plateau": "Jos",
    "Rivers": "Port Harcourt",
    "Sokoto": "Sokoto",
    "Taraba": "Jalingo",
    "Yobe": "Damaturu",
    "Zamfara": "Gusau",
    "FCT": "Abuja",
    "Federal Capital Territory": "Abuja"
}

BANKS = {
    "Access Bank": "044",
    "Citibank": "023",
    "Ecobank": "050",
    "Fidelity Bank": "070",
    "First Bank of Nigeria": "011",
    "First City Monument Bank": "214",
    "Globus Bank": "00103",
    "Guaranty Trust Bank": "058",
    "Heritage Bank": "030",
    "Keystone Bank": "082",
    "Optimus Bank": "00107",
    "Parallex Bank": "00030",
    "PremiumTrust Bank": "00105",
    "Providus Bank": "101",
    "Signature Bank": "00108",
    "Stanbic IBTC Bank": "221",
    "Standard Chartered Bank": "068",
    "Sterling Bank": "232",
    "SunTrust Bank": "100",
    "Titan Trust Bank": "00102",
    "Union Bank of Nigeria": "032",
    "United Bank for Africa": "033",
    "Unity Bank": "215",
    "Wema Bank": "035",
    "Zenith Bank": "057"
}

def format_naira(amount):
    """
    Format a number or string as Naira (₦).
    e.g. 1500000 -> ₦1,500,000.00
    """
    try:
        val = float(amount)
        return f"₦{val:,.2f}"
    except (ValueError, TypeError):
        return f"₦{amount}"

# Prefix groupings based on NCC allocations
_MTN_PREFIXES = {"0803", "0806", "0810", "0813", "0814", "0816", "0903", "0906", "0703", "0706", "0913", "0916", "0704"}
_AIRTEL_PREFIXES = {"0802", "0808", "0812", "0701", "0708", "0902", "0907", "0901", "0912", "0904"}
_GLO_PREFIXES = {"0805", "0807", "0811", "0815", "0705", "0905", "0915"}
_9MOBILE_PREFIXES = {"0809", "0817", "0818", "0909", "0908"}

def get_telco(phone):
    """
    Given a phone number (e.g. +2348031234567, 08031234567, 8031234567),
    returns the operator name (MTN, Airtel, Glo, 9mobile) or None.
    """
    if not phone or not isinstance(phone, str):
        return None
        
    # Remove non-digits
    digits = "".join(c for c in phone if c.isdigit())
    
    # Normalize to local 0-prefix 11-digit structure
    if digits.startswith("234") and len(digits) > 3:
        digits = "0" + digits[3:]
    elif not digits.startswith("0") and len(digits) == 10:
        digits = "0" + digits
        
    if len(digits) < 4:
        return None
        
    prefix = digits[:4]
    
    if prefix in _MTN_PREFIXES:
        return "MTN"
    elif prefix in _AIRTEL_PREFIXES:
        return "Airtel"
    elif prefix in _GLO_PREFIXES:
        return "Glo"
    elif prefix in _9MOBILE_PREFIXES:
        return "9mobile"
        
    return None
