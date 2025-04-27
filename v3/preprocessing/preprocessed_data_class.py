class PreprocessedData:
    def __init__(
            self,
            context_window_size: int,
            training_data: list[tuple[int, list[int]]]
    ):
        self.context_window_size: int = context_window_size
        self.training_data: list[tuple[int, list[int]]] = training_data
