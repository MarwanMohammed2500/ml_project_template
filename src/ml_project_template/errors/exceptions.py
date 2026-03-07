class InvalidEnvVarError(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidModelPathError(Exception):
    def __init__(self, message):
        super().__init__(message)
