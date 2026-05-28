"""Normalize text before TTS so symbols/markdown aren't read aloud.

Natural sentence punctuation (. , ! ? ; : - " ' ( )) is kept — TTS engines use
it for prosody and do not speak it. Markdown markers and symbols that get
verbalized (e.g. '#' -> "hash", '@' -> "at") or that corrupt neighbouring words
(e.g. '*', '_') are removed.
"""

import re

# ![alt](url) -> alt   ;   [text](url) -> text
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# leading markdown tokens per line: headings, blockquotes, list bullets
_MD_LEADING = re.compile(r"(?m)^[ \t]*(?:#{1,6}\s+|>\s+|[-*+]\s+|\d+[.)]\s+)")
# emphasis / inline-code markers
_EMPHASIS = re.compile(r"[*_`~]+")
# symbols that are spoken aloud or break words -> replaced with a space
_SPOKEN_SYMBOLS = re.compile(r"[#@^|\\<>{}\[\]=+]+")
# emoji / pictographs / flags / variation selectors
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff"
    "\U00002b00-\U00002bff"
    "️‍"
    "]+"
)
_NEWLINES = re.compile(r"\s*[\r\n]+\s*")
_SPACES = re.compile(r"[ \t ]+")
# leftover empty brackets / spaced-out punctuation, e.g. "(  )" or " , ,"
_EMPTY_PARENS = re.compile(r"\(\s*\)")


def clean_tts_text(text: str) -> str:
    if not text:
        return text
    t = _MD_IMAGE.sub(r"\1", text)
    t = _MD_LINK.sub(r"\1", t)
    t = _MD_LEADING.sub("", t)
    t = _EMPHASIS.sub(" ", t)
    t = _EMOJI.sub(" ", t)
    t = _SPOKEN_SYMBOLS.sub(" ", t)
    t = _NEWLINES.sub(". ", t)   # paragraph breaks become sentence pauses
    t = _EMPTY_PARENS.sub(" ", t)

    # Periods *inside* a token are spoken as "dot" (domains, emails,
    # abbreviations). "word.Word" -> sentence break; other in-token dots -> space.
    # Decimals (digit.digit) are left so they read naturally as "point".
    t = re.sub(r"(?<=[A-Za-z])\.(?=[A-Z])", ". ", t)
    t = re.sub(r"(?<=[A-Za-z])\.(?=[a-z])", " ", t)
    t = re.sub(r"(?<=[A-Za-z])\.(?=\d)", " ", t)
    t = re.sub(r"(?<=\d)\.(?=[A-Za-z])", " ", t)

    t = _SPACES.sub(" ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)  # drop space before punctuation
    t = re.sub(r"\.{2,}", ".", t)           # ellipsis / joined periods -> "."
    return t.strip()
