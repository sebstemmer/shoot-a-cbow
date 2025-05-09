import re


def remove_punctuation(sentences: list[str]) -> list[str]:
    pattern = r"[^\w\s]"
    return [re.sub(pattern, "", sentence) for sentence in sentences]
