"""
example_contract_analysis.py

Demonstrates using Legal-Peace for contract clause analysis.
Requires: pip install olaverse[legal]  (GPU + unsloth)
"""

from olaverse.llm import LegalPeace

# --------------------------------------------------------------------------- #
# 1. Initialize and load the model
# --------------------------------------------------------------------------- #
model = LegalPeace()
model.load()   # downloads olaverse/legal-peace-v1.0 from HF on first run

# --------------------------------------------------------------------------- #
# 2. Contract clause analysis
# --------------------------------------------------------------------------- #
clauses = [
    "The parties agree that all disputes shall be resolved through binding "
    "arbitration in Delaware under the rules of the American Arbitration Association.",

    "Either party may terminate this agreement with 30 days written notice. "
    "Upon termination, all outstanding invoices become immediately due and payable.",

    "The Licensor grants the Licensee a non-exclusive, non-transferable, "
    "revocable license to use the Software solely for internal business purposes.",
]

for i, clause in enumerate(clauses, 1):
    print(f"\n{'='*60}")
    print(f"Clause {i}:")
    print(clause)
    print(f"{'-'*60}")

    prompt = (
        f"Analyze this contract clause:\n'{clause}'\n\n"
        "Identify:\n"
        "1. Key obligations for each party\n"
        "2. Potential risks or ambiguities\n"
        "3. Recommended modifications"
    )

    response = model.generate(prompt, max_new_tokens=400, temperature=0.5)
    print("Analysis:")
    print(response)
