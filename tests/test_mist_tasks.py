"""
test_mist_tasks.py

Tests for the task-specific MIST wrappers: MISTTitleGenerator (mist-tg-0.3b)
and MISTQuestionGenerator (mist-qg-1.5b).

The default tests are lightweight — no network, no GPU, no model downloads.
The real-checkpoint tests at the bottom are marked ``slow`` (excluded by
default; run with ``pytest -m slow``).
"""

import pytest
from unittest.mock import patch

from olaverse import MISTTitleGenerator, MISTQuestionGenerator, QG_LANGUAGES
from olaverse.llm.mist_tasks import _QG_USER_TEMPLATE

slow = pytest.mark.slow


# =========================================================================== #
# Registry / construction
# =========================================================================== #

def test_title_generator_default_repo():
    assert MISTTitleGenerator().model_name == "olaverse/mist-tg-0.3b"


def test_question_generator_default_repo():
    assert MISTQuestionGenerator().model_name == "olaverse/mist-qg-1.5b"


def test_unknown_size_passes_through_as_hf_id():
    # Lets callers point at a fork or a future checkpoint without an SDK release.
    assert MISTTitleGenerator(size="some-org/custom-tg").model_name == "some-org/custom-tg"
    assert MISTQuestionGenerator(size="some-org/custom-qg").model_name == "some-org/custom-qg"


def test_constructors_do_not_load():
    # Construction must stay cheap — no download until .load()/.generate().
    assert MISTTitleGenerator()._loaded is False
    assert MISTQuestionGenerator()._loaded is False


def test_import_error_without_transformers():
    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(ImportError) as exc:
            MISTTitleGenerator().load()
        assert "olaverse[deeplearning]" in str(exc.value)


# =========================================================================== #
# QG — language resolution
# =========================================================================== #

def test_qg_language_count():
    assert len(QG_LANGUAGES) == 25


@pytest.mark.parametrize("value,expected", [
    ("yo", "Yoruba"),
    ("Yoruba", "Yoruba"),
    ("yoruba", "Yoruba"),
    ("  FR  ", "French"),
    ("en", "English"),
])
def test_qg_resolve_language(value, expected):
    assert MISTQuestionGenerator._resolve_language(value) == expected


def test_qg_rejects_unsupported_language():
    with pytest.raises(ValueError) as exc:
        MISTQuestionGenerator._resolve_language("klingon")
    assert "klingon" in str(exc.value)


def test_qg_prompt_interpolates_resolved_name():
    prompt = _QG_USER_TEMPLATE.format(n=3, language="Yoruba", passage="A passage.")
    assert "Write 3 questions" in prompt
    assert "Write the questions in Yoruba." in prompt
    assert "A passage." in prompt
    # The JSON contract must survive .format() — braces are doubled in the template.
    assert '{"questions": ["...", "...", "..."]}' in prompt


# =========================================================================== #
# QG — output parsing
# =========================================================================== #

def test_qg_parses_clean_json():
    text = '{"questions": ["What causes tides?", "Does the sun matter?"]}'
    assert MISTQuestionGenerator._parse_questions(text, 3) == [
        "What causes tides?", "Does the sun matter?",
    ]


def test_qg_parses_json_with_surrounding_text():
    text = 'Sure, here you go:\n{"questions": ["Q one?"]}\nHope that helps!'
    assert MISTQuestionGenerator._parse_questions(text, 3) == ["Q one?"]


def test_qg_truncates_to_n():
    text = '{"questions": ["a?", "b?", "c?", "d?"]}'
    assert len(MISTQuestionGenerator._parse_questions(text, 2)) == 2


def test_qg_drops_empty_entries():
    text = '{"questions": ["Real question?", "", "   "]}'
    assert MISTQuestionGenerator._parse_questions(text, 3) == ["Real question?"]


def test_qg_recovers_from_truncated_json():
    # max_new_tokens cut the generation off before the closing brace.
    text = '{"questions": ["What causes ocean tides?", "Does the sun affect tides?"'
    recovered = MISTQuestionGenerator._parse_questions(text, 3)
    assert "What causes ocean tides?" in recovered
    assert "questions" not in recovered


def test_qg_returns_empty_on_garbage():
    assert MISTQuestionGenerator._parse_questions("total nonsense", 3) == []


# =========================================================================== #
# Real checkpoints — opt in with: pytest -m slow
# =========================================================================== #

@slow
def test_title_generator_real():
    titler = MISTTitleGenerator()
    title = titler.generate(
        "My laptop keeps freezing every time I open more than five browser tabs, any idea why?"
    )
    assert isinstance(title, str) and title.strip()


@slow
def test_title_generator_batch_matches_single_real():
    titler = MISTTitleGenerator()
    messages = ["How do I center a div in CSS?", "What makes Yoruba a tonal language?"]
    assert titler.generate_batch(messages) == [titler.generate(m) for m in messages]


@slow
def test_question_generator_real():
    qg = MISTQuestionGenerator()
    passage = ("Tides are caused by the gravitational pull of the moon and, to a "
               "lesser extent, the sun, acting on Earth's oceans.")
    questions = qg.generate(passage, n=3)
    assert 1 <= len(questions) <= 3
    assert all(isinstance(q, str) and q.strip() for q in questions)


@slow
def test_question_generator_respects_high_resource_language_real():
    qg = MISTQuestionGenerator()
    passage = ("Tides are caused by the gravitational pull of the moon and, to a "
               "lesser extent, the sun, acting on Earth's oceans.")
    questions = qg.generate(passage, n=2, language="French")
    assert questions
    # Not a language classifier — just assert it stopped writing English.
    assert any(tok in " ".join(questions).lower() for tok in ("les ", "la ", "est", "qu"))
