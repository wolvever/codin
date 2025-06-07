"""ID generation utilities for the codin framework.

This module provides functions for generating unique identifiers
with customizable prefixes and lengths.
"""


def new_id(prefix: str, length: int = 8, uuid: bool = False) -> str:
    """Create an ID with the given prefix.

    Args:
        length: Length of the random segment if ``uuid`` is ``False``.
        uuid: Generate a UUID4 segment when ``True``. Defaults to ``False``.

    Returns:
        A string identifier prefixed by ``prefix``.
    """
    import random
    import string
    import uuid as _uuid

    if uuid:
        suffix = str(_uuid.uuid4())
    else:
        suffix = "".join(
            random.choices(string.ascii_letters + string.digits, k=length)
        )

    return f"{prefix}-{suffix}"
