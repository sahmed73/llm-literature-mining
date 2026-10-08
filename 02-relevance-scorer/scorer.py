"""
LLM-based relevance scorer using a local model served by Ollama.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

import requests

import config


def _call_ollama(title: str, abstract: str,
                 logger: logging.Logger) -> Tuple[Optional[int], Optional[str]]:
    """
    Call Ollama to score a paper's relevance.
    Returns (score, reason) or (None, None) on failure.
    """
    # Truncate abstract
    if abstract and len(abstract) > config.MAX_ABSTRACT_CHARS:
        abstract = abstract[:config.MAX_ABSTRACT_CHARS] + "…"

    user_msg = config.USER_PROMPT_TEMPLATE.format(
        title=title or "(no title)",
        abstract=abstract or "(no abstract available — score based on title only)",
    )

    payload = {
        "model": config.MODEL_NAME,
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,       # low temp for consistent scoring
            "num_predict": 150,       # short response expected
        },
    }
    if config.NUM_CTX:
        payload["options"]["num_ctx"] = config.NUM_CTX
    if config.THINK is not None:
        payload["think"] = config.THINK
    if config.FORCE_JSON:
        payload["format"] = config.SCORE_SCHEMA

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data.get("message", {}).get("content", "").strip()
            score, reason = _parse_response(content)

            if score is not None:
                return score, reason
            else:
                logger.debug("  Parse failed (attempt %d): raw='%s'", attempt, content[:200])

        except requests.exceptions.Timeout:
            logger.warning("  Ollama timeout (attempt %d/%d)", attempt, config.MAX_RETRIES)
        except requests.exceptions.ConnectionError:
            logger.error("  Ollama connection error (attempt %d/%d)", attempt, config.MAX_RETRIES)
            time.sleep(5)
        except Exception as exc:
            logger.warning("  Ollama error (attempt %d/%d): %s", attempt, config.MAX_RETRIES, exc)

        time.sleep(1)

    return None, None


def _parse_response(content: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse the LLM response to extract score and reason.
    Handles various response formats robustly.
    """
    # Strategy 1: Try direct JSON parse
    try:
        # Find JSON in the response (may be surrounded by text)
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            obj = json.loads(json_match.group())
            score = obj.get("score")
            reason = obj.get("reason", "")
            if score is not None:
                score = int(float(score))
                score = max(0, min(10, score))  # clamp to 0-10
                return score, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Strategy 2: Look for a bare number
    number_match = re.search(r'\b(\d{1,2})\b', content)
    if number_match:
        score = int(number_match.group(1))
        if 0 <= score <= 10:
            return score, content[:200]

    return None, None


def score_paper(paper: Dict[str, Any],
                logger: logging.Logger) -> Dict[str, Any]:
    """
    Score a single paper and return it with added score fields.

    Adds:
        relevance_score: int (0-10) or None if scoring failed
        relevance_reason: str
        score_confidence: str ("high" if abstract present, "low" if title-only)
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    score, reason = _call_ollama(title, abstract, logger)

    result = dict(paper)
    result["relevance_score"] = score
    result["relevance_reason"] = reason or ""
    result["score_confidence"] = "high" if abstract else "low"

    return result
