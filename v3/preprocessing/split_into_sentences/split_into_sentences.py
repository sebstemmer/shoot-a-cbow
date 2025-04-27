import re


def split_into_sentences(text: str) -> list[str]:
    pattern = r"[.!?]\s+"
    return re.split(pattern, text)
