class Vocab:
    def __init__(
            self,
            vocab_size: int,
            word_to_idx_vocab: dict[str, int],
            idx_to_word_vocab: dict[int, str],
            word_to_idx_count_vocab: dict[str, tuple[int, int]],
            idx_to_vocab_freq: dict[int, float]
    ):
        self.vocab_size: int = vocab_size
        self.word_to_idx_vocab: dict[str, int] = word_to_idx_vocab
        self.idx_to_word_vocab: dict[int, str] = idx_to_word_vocab
        self.word_to_idx_count_vocab: dict[
            str,
            tuple[int, int]
        ] = word_to_idx_count_vocab
        self.idx_to_vocab_freq: dict[int, float] = idx_to_vocab_freq
