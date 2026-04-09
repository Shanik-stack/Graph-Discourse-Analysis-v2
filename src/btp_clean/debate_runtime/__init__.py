from .api import (
    add_speaker_statement,
    addalot_of_statements,
    get_debate_output,
    start_debate,
)
from .core import Debate, Fallacy_Checker

__all__ = [
    "Debate",
    "Fallacy_Checker",
    "start_debate",
    "add_speaker_statement",
    "addalot_of_statements",
    "get_debate_output",
]
