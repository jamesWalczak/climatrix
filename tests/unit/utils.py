from functools import wraps

import pytest


def skip_on_error(*err_types):
    """
    Decorator to skip a test if any of the specified error types are raised.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except err_types as e:
                pytest.skip(f"Skipped due to error: {e}")

        return wrapper

    return decorator
