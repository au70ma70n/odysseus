"""
Streaming repetition-collapse detector.

Catches the Gemma 4 token-degeneration pattern (issue #622) where the model
starts doubling a word/phrase and then collapses into a single repeated token
filling the rest of the generation budget.

The detector is fed incremental text deltas and maintains a lightweight
sliding window.  It reports a collapse when:

1. A short token/word (≤ ``MAX_REPEAT_TOKEN_LEN`` chars) has appeared
   ``REPEAT_THRESHOLD`` times consecutively (whitespace-separated), OR
2. A short phrase (≤ ``MAX_REPEAT_PHRASE_WORDS`` words) has appeared
   ``PHRASE_REPEAT_THRESHOLD`` times in the last ``WINDOW_CHARS`` characters.

Both heuristics are intentionally conservative — they fire well after a
human would notice the output is garbage, so false-positives on legitimate
text (e.g. "ha ha ha ha") are extremely unlikely at the default thresholds.
"""

from __future__ import annotations

import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

WINDOW_CHARS = 1200
MAX_REPEAT_TOKEN_LEN = 30
REPEAT_THRESHOLD = 15
MAX_REPEAT_PHRASE_WORDS = 4
PHRASE_REPEAT_THRESHOLD = 10


class RepetitionCollapseDetector:
    """Feed text deltas; call ``check()`` periodically to test for collapse."""

    __slots__ = ("_buf", "_collapsed", "_trigger_phrase", "_check_interval",
                 "_chars_since_check")

    def __init__(self, *, check_every_chars: int = 120):
        self._buf: str = ""
        self._collapsed: bool = False
        self._trigger_phrase: str | None = None
        self._check_interval = check_every_chars
        self._chars_since_check = 0

    # ------------------------------------------------------------------
    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def trigger_phrase(self) -> str | None:
        return self._trigger_phrase

    # ------------------------------------------------------------------
    def feed(self, delta: str) -> bool:
        """Append *delta* and return True if collapse is now detected."""
        if self._collapsed:
            return True
        self._buf += delta
        self._chars_since_check += len(delta)
        if self._chars_since_check >= self._check_interval:
            self._chars_since_check = 0
            return self.check()
        return False

    def check(self) -> bool:
        """Run the heuristics against the current buffer."""
        if self._collapsed:
            return True
        tail = self._buf[-WINDOW_CHARS:] if len(self._buf) > WINDOW_CHARS else self._buf
        if len(tail) < 60:
            return False

        # --- Heuristic 1: single-token runaway ---
        # Split on whitespace and look for long consecutive runs of the same token.
        tokens = tail.split()
        if len(tokens) >= REPEAT_THRESHOLD:
            run_len = 1
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i - 1] and len(tokens[i]) <= MAX_REPEAT_TOKEN_LEN:
                    run_len += 1
                    if run_len >= REPEAT_THRESHOLD:
                        self._collapsed = True
                        self._trigger_phrase = tokens[i]
                        logger.warning(
                            "Repetition collapse detected: token %r repeated %d+ times",
                            self._trigger_phrase, run_len,
                        )
                        return True
                else:
                    run_len = 1

        # --- Heuristic 2: short-phrase repetition ---
        # Slide a window of N words and count how often each phrase appears.
        # Only fires when the repeated phrase is a substantial fraction of the
        # window — avoids false-positives on structured text like numbered
        # lists where a common word ("Item", "Step") recurs structurally.
        for n in range(2, MAX_REPEAT_PHRASE_WORDS + 1):
            if len(tokens) < n + PHRASE_REPEAT_THRESHOLD:
                continue
            phrases: Counter[str] = Counter()
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i + n])
                if len(phrase) > MAX_REPEAT_TOKEN_LEN * n:
                    continue
                phrases[phrase] += 1
            for phrase, count in phrases.most_common(3):
                if count >= PHRASE_REPEAT_THRESHOLD:
                    density = (count * n) / len(tokens)
                    if density < 0.50:
                        continue
                    self._collapsed = True
                    self._trigger_phrase = phrase
                    logger.warning(
                        "Repetition collapse detected: phrase %r repeated %d times "
                        "(density %.0f%%) in window",
                        phrase, count, density * 100,
                    )
                    return True

        return False

    def reset(self) -> None:
        self._buf = ""
        self._collapsed = False
        self._trigger_phrase = None
        self._chars_since_check = 0
