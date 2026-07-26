from olaverse.llm.legal import LegalPeace
from olaverse.llm.detector import LIDNeural5  # re-exported from olaverse.nlp for backward compat
from olaverse.llm.mist import MIST
from olaverse.llm.mist_tasks import MISTTitleGenerator, MISTQuestionGenerator, QG_LANGUAGES

__all__ = [
    "LegalPeace",
    "LIDNeural5",
    "MIST",
    "MISTTitleGenerator",
    "MISTQuestionGenerator",
    "QG_LANGUAGES",
]
