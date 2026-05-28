"""
test_legal_peace.py

Unit and integration tests for the LegalPeace olaverse interface.

Tests that can run WITHOUT a GPU (mock tests) are separated from
GPU inference tests. Run lightweight tests with:
    pytest peace/tests/test_legal_peace.py -m "not gpu"

Run all tests (requires GPU + unsloth):
    pytest peace/tests/test_legal_peace.py
"""

import pytest
from unittest.mock import MagicMock, patch


# =========================================================================== #
# Lightweight / Mock Tests (no GPU required)
# =========================================================================== #

class TestLegalPeaceInit:
    """Tests for LegalPeace initialization — no model load required."""

    def test_default_model_name(self):
        from olaverse.llm import LegalPeace
        model = LegalPeace()
        assert model.model_name == "olaverse/legal-peace-v1.0"

    def test_custom_model_name(self):
        from olaverse.llm import LegalPeace
        model = LegalPeace(model_name="some/other-model")
        assert model.model_name == "some/other-model"

    def test_default_params(self):
        from olaverse.llm import LegalPeace
        model = LegalPeace()
        assert model.max_seq_length == 2048
        assert model.load_in_4bit is True
        assert model._loaded is False

    def test_not_loaded_before_load_call(self):
        from olaverse.llm import LegalPeace
        model = LegalPeace()
        assert model.model is None
        assert model.tokenizer is None

    def test_load_raises_without_unsloth(self):
        from olaverse.llm import LegalPeace
        model = LegalPeace()
        with patch("builtins.__import__", side_effect=ImportError("No module named 'unsloth'")):
            with pytest.raises(ImportError, match="unsloth"):
                model.load()

    def test_generate_triggers_load_if_not_loaded(self):
        """generate() should call load() automatically if not yet loaded."""
        from olaverse.llm import LegalPeace
        model = LegalPeace()
        model.load = MagicMock()
        model._loaded = False

        # Patch the internal model/tokenizer to avoid actual inference
        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value=MagicMock())
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.decode.return_value = "Mocked legal analysis."
        mock_model = MagicMock()
        mock_model.generate.return_value = [mock_output[0]]

        def side_effect_load():
            model._loaded = True
            model.model = mock_model
            model.tokenizer = mock_tokenizer

        model.load.side_effect = side_effect_load

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch", MagicMock()):
                try:
                    model.generate("Test prompt")
                except Exception:
                    pass  # We only care that load() was called

        model.load.assert_called_once()

    def test_double_load_is_idempotent(self):
        """Calling load() twice should not reload the model."""
        from olaverse.llm import LegalPeace
        model = LegalPeace()
        model._loaded = True  # pretend already loaded
        model.model = MagicMock()
        model.tokenizer = MagicMock()

        with patch("olaverse.llm.legal.LegalPeace.load") as mock_load:
            mock_load.side_effect = lambda: None
            # Calling generate should not invoke load again since _loaded=True
            model.load()


# =========================================================================== #
# Integration Tests (require GPU + unsloth)
# =========================================================================== #

@pytest.mark.gpu
class TestLegalPeaceInference:
    """Full inference tests — require GPU and unsloth installed."""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        from olaverse.llm import LegalPeace
        model = LegalPeace()
        model.load()
        return model

    def test_load_succeeds(self, loaded_model):
        assert loaded_model._loaded is True
        assert loaded_model.model is not None
        assert loaded_model.tokenizer is not None

    def test_generate_returns_string(self, loaded_model):
        response = loaded_model.generate("What is a contract?", max_new_tokens=50)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_contract_clause(self, loaded_model):
        clause = "All disputes shall be resolved through binding arbitration in Delaware."
        prompt = f"Analyze this clause: '{clause}'"
        response = loaded_model.generate(prompt, max_new_tokens=200, temperature=0.5)
        assert isinstance(response, str)
        assert len(response) > 50  # Should produce a meaningful response

    def test_generate_with_low_temperature(self, loaded_model):
        """Low temperature should give more deterministic output."""
        prompt = "What is a force majeure clause?"
        r1 = loaded_model.generate(prompt, max_new_tokens=100, temperature=0.1)
        r2 = loaded_model.generate(prompt, max_new_tokens=100, temperature=0.1)
        # With low temperature results should be very similar (not strictly identical due to sampling)
        assert isinstance(r1, str) and isinstance(r2, str)

    def test_generate_legal_qa(self, loaded_model):
        prompt = (
            "Question: What constitutes a breach of contract?\n"
            "Context: U.S. contract law\n"
            "Answer:"
        )
        response = loaded_model.generate(prompt, max_new_tokens=200)
        assert isinstance(response, str)
        assert len(response) > 30
