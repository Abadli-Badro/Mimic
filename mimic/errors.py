"""Expected failures with machine-readable codes and processing context."""
from functools import wraps


class MimicError(ValueError):
    def __init__(self, code, message, stage=None):
        super().__init__(message)
        self.code = code
        self.stage = stage

    def __str__(self):
        prefix = f'{self.stage}: ' if self.stage else ''
        return f'{prefix}[{self.code}] {super().__str__()}'


def stage(name):
    """Label expected failures while preserving the original exception cause."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except MimicError as exc:
                if exc.stage is None:
                    exc.stage = name
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                code = 'io_error' if isinstance(exc, OSError) else 'invalid_data'
                raise MimicError(code, str(exc), name) from exc
        return wrapped
    return decorate
