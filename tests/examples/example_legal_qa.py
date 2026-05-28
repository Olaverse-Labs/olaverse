"""
example_legal_qa.py

Demonstrates using Legal-Peace for legal Q&A and case reasoning.
Requires: pip install olaverse[legal]  (GPU + unsloth)
"""

from olaverse.llm import LegalPeace

model = LegalPeace()
model.load()

# --------------------------------------------------------------------------- #
# Legal Q&A examples
# --------------------------------------------------------------------------- #
questions = [
    {
        "question": "What constitutes a breach of contract under U.S. law?",
        "context": "General contract law",
    },
    {
        "question": "Is a verbal agreement legally binding?",
        "context": "U.S. contract formation",
    },
    {
        "question": "What are the elements required to prove negligence?",
        "context": "U.S. tort law",
    },
    {
        "question": "What is the difference between indemnification and limitation of liability clauses?",
        "context": "Commercial contracts",
    },
]

for qa in questions:
    print(f"\n{'='*60}")
    print(f"Question: {qa['question']}")
    print(f"Context:  {qa['context']}")
    print(f"{'-'*60}")

    prompt = (
        f"Question: {qa['question']}\n"
        f"Context: {qa['context']}\n"
        "Provide a clear, structured legal answer:"
    )

    response = model.generate(prompt, max_new_tokens=350, temperature=0.4)
    print("Answer:")
    print(response)
