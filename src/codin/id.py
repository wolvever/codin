"""ID generation utilities for the codin framework.

This module provides functions for generating unique identifiers
with customizable prefixes and lengths.
"""


def new_id(prefix: str, length: int = 8) -> str:
    """Create a random prefix of specified length using alphanumeric characters.

    Args:
        length: Length of the prefix to generate. Defaults to 8.

    Returns:
        A random string of alphanumeric characters of specified length.
    """
    import random
    import string

    return f'{prefix}-{"".join(random.choices(string.ascii_letters + string.digits, k=length))}'
