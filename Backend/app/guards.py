"""Guardrails AI validation.

Validators are custom classes built on the real Guardrails API (Validator,
register_validator, PassResult/FailResult) and run through `Guard`. No Guardrails
Hub validators are used, so no Hub token is needed.
"""
import re
from typing import Any, Dict

from guardrails import Guard, OnFailAction
from guardrails.validators import FailResult, PassResult, Validator, register_validator

from .config import MAX_ANSWER_CHARS, MAX_QUESTION_CHARS

_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)"
    r"|(reveal|show|print|repeat)\s+(me\s+)?(your\s+|the\s+)?(system\s+prompt|hidden\s+instructions|api[\s_-]?key)"
    r"|you\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)",
    re.IGNORECASE)
_SECRET = re.compile(r"gsk_[A-Za-z0-9]{10,}|GROQ_API_KEY|system prompt", re.IGNORECASE)
_FILENAME = re.compile(r"[\w\-. ]+\.(?:pdf|txt)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z]{5,}")


@register_validator(name="rag/safe_question", data_type="string")
class SafeQuestion(Validator):
    """Input check: non-empty, not too long, no obvious prompt-injection phrasing."""

    def validate(self, value: Any, metadata: Dict) -> Any:
        if not isinstance(value, str) or not value.strip():
            return FailResult(error_message="The question is empty.")
        if len(value) > MAX_QUESTION_CHARS:
            return FailResult(error_message=f"The question is too long (max {MAX_QUESTION_CHARS} characters).")
        if _INJECTION.search(value):
            return FailResult(error_message="The question contains instructions that cannot be followed.")
        return PassResult()


@register_validator(name="rag/grounded_answer", data_type="string")
class GroundedAnswer(Validator):
    """Output check: well-formed, no leaked secrets, no invented citations, mostly supported by context."""

    def validate(self, value: Any, metadata: Dict) -> Any:
        metadata = metadata or {}
        if not isinstance(value, str) or not value.strip():
            return FailResult(error_message="The answer is empty.")
        if len(value) > MAX_ANSWER_CHARS:
            return FailResult(error_message="The answer is too long.")
        if _SECRET.search(value):
            return FailResult(error_message="The answer exposes restricted content.")
        known = {s.lower() for s in metadata.get("sources", [])}
        for cited in (m.group(0) for m in _FILENAME.finditer(value)):
            if cited.strip().lower() not in known:
                return FailResult(error_message=f"The answer cites an unknown source: {cited.strip()}")
        support = (metadata.get("context", "") + " " + metadata.get("question", "")).lower()
        words = set(_WORD.findall(value.lower()))
        if len(words) >= 4:
            supported = sum(1 for w in words if w in support)
            if supported / len(words) < 0.4:
                return FailResult(error_message="The answer is not supported by the document context.")
        return PassResult()


_input_guard = Guard(name="rag_input").use(SafeQuestion(on_fail=OnFailAction.NOOP))
_output_guard = Guard(name="rag_output").use(GroundedAnswer(on_fail=OnFailAction.NOOP))


def _reason(outcome) -> str:
    for summary in getattr(outcome, "validation_summaries", None) or []:
        if getattr(summary, "failure_reason", None):
            return summary.failure_reason
    return "Validation failed."


def validate_question(question: Any) -> tuple[bool, str]:
    outcome = _input_guard.validate(question if isinstance(question, str) else "")
    return bool(outcome.validation_passed), "" if outcome.validation_passed else _reason(outcome)


def validate_answer(answer: str, context: str, question: str, sources: list[str]) -> tuple[bool, str]:
    outcome = _output_guard.validate(
        answer, metadata={"context": context, "question": question, "sources": sources})
    return bool(outcome.validation_passed), "" if outcome.validation_passed else _reason(outcome)
