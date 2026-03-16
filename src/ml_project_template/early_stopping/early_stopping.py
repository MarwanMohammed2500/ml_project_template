class EarlyStopping:
    """
    Early Stopping class

    Args:
        patience: int=5:
            The number of epochs to wait before deciding that the model is not learning anymore

        delta: float=0.0:
            Minimum change in monitored metric (loss in this implementation) that is considered improvement
    """

    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.best_model_state = None
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.counter = 0

    def __call__(self, loss, model):
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        if self.best_model_state is None:
            raise ValueError(
                "Calling this method before running early stopping is not permitted"
            )
        model.load_state_dict(self.best_model_state)
    def reset(self):
        """Resets the early stopping state for a new training phase."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
