# Utilities

The `olaverse.utils` module provides global constants, currency formatting, and audio I/O helpers used throughout the library and available for your own use.

```python
from olaverse.utils import format_currency, CURRENCIES, CONTINENTS, save_audio, load_audio
```

---

## Currency Formatting

### `format_currency`

Format any numeric value as a currency string with the correct symbol.

```python
from olaverse.utils import format_currency

format_currency(1500, "₦")       # → '₦1,500.00'
format_currency(3_750_000, "₦")  # → '₦3,750,000.00'
format_currency(99.9, "$")        # → '$99.90'
format_currency("invalid", "£")  # → '£invalid'
```

::: olaverse.utils.format_currency

---

### `CURRENCIES`

A dictionary mapping ISO 4217 currency codes to their symbols. Includes the Nigerian Naira and major world currencies.

```python
from olaverse.utils import CURRENCIES

CURRENCIES["NGN"]   # → '₦'
CURRENCIES["USD"]   # → '$'
CURRENCIES["GBP"]   # → '£'
CURRENCIES["EUR"]   # → '€'

# Use with format_currency
amount = 5_000
symbol = CURRENCIES["NGN"]
format_currency(amount, symbol)  # → '₦5,000.00'
```

| Code | Currency | Symbol |
|---|---|---|
| NGN | Nigerian Naira | ₦ |
| USD | US Dollar | $ |
| GBP | British Pound | £ |
| EUR | Euro | € |
| JPY | Japanese Yen | ¥ |
| ZAR | South African Rand | R |
| INR | Indian Rupee | ₹ |
| BRL | Brazilian Real | R$ |

---

### `CONTINENTS`

ISO continent codes mapped to full names.

```python
from olaverse.utils import CONTINENTS

CONTINENTS["AF"]   # → 'Africa'
CONTINENTS["EU"]   # → 'Europe'
CONTINENTS["NA"]   # → 'North America'
```

---

## Audio I/O

Standard read/write helpers for `.wav` files. Useful when working with the speech pipeline or external TTS models.

```bash
pip install scipy   # required for audio I/O
```

### `save_audio`

```python
import numpy as np
from olaverse.utils import save_audio

# Save a generated waveform (e.g. from a vocoder)
waveform = np.zeros(22050, dtype=np.float32)  # 1 second of silence
save_audio(waveform, sample_rate=22050, output_path="output/test.wav")
```

::: olaverse.utils.save_audio

---

### `load_audio`

```python
from olaverse.utils import load_audio

sample_rate, waveform = load_audio("path/to/audio.wav")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(waveform) / sample_rate:.2f}s")
print(f"Shape: {waveform.shape}")
```

::: olaverse.utils.load_audio
