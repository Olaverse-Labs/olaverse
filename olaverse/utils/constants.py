# Global Constants and Utilities

CURRENCIES = {
    "USD": "$",    # US Dollar
    "EUR": "€",    # Euro
    "GBP": "£",    # British Pound
    "JPY": "¥",    # Japanese Yen
    "CNY": "¥",    # Chinese Yuan
    "NGN": "₦",    # Nigerian Naira
    "ZAR": "R",    # South African Rand
    "INR": "₹",    # Indian Rupee
    "CAD": "$",    # Canadian Dollar
    "AUD": "$",    # Australian Dollar
    "BRL": "R$",   # Brazilian Real
}

CONTINENTS = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
    "AN": "Antarctica"
}

def format_currency(amount, symbol: str = "$") -> str:
    """
    Format a number or string as a generic currency.
    e.g. format_currency(1500, "$") -> $1,500.00
    """
    try:
        val = float(amount)
        return f"{symbol}{val:,.2f}"
    except (ValueError, TypeError):
        return f"{symbol}{amount}"
