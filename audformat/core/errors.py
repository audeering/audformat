from collections.abc import Sequence


class BadIdError(ValueError):
    r"""Raised when a field identifier is unknown.

    Args:
        name: name of the field
        invalid_id: identifier causing the error
        dictionary: dictionary with valid identifiers

    """

    def __init__(self, name: str, invalid_id: str, dictionary: dict):
        if not dictionary:
            message = f"Bad {name} ID '{invalid_id}', " f"no {name} objects defined yet"
        else:
            message = (
                f"Bad {name} ID '{invalid_id}', " f"expected one of {list(dictionary)}"
            )
        super().__init__(message)


class BadKeyError(KeyError):
    """Raised when a key is not found.

    Args:
        invalid_key: value causing the error
        valid_keys: list of valid strings

    """

    def __init__(self, invalid_key: str, valid_keys: Sequence[str]):
        message = f"Bad key '{invalid_key}', " f"expected one of {list(valid_keys)}"
        super().__init__(message)


class BadTypeError(TypeError):
    r"""Raised when a value has an unexpected type.

    Args:
        invalid_value: value causing the error
        expected_type: expected value type

    """

    def __init__(self, invalid_value: object, expected_type: type):
        message = f"Bad type '{type(invalid_value)}', " f"expected '{expected_type}'"
        super().__init__(message)


class BadValueError(ValueError):
    """Raised when a value is not in a list of pre-defined strings.

    Args:
        invalid_value: value causing the error
        valid_values: list of valid strings

    """

    def __init__(self, invalid_value: str, valid_values: Sequence[str]):
        message = (
            f"Bad value '{invalid_value}', " f"expected one of {list(valid_values)}"
        )
        super().__init__(message)


class TableExistsError(KeyError):
    r"""If a table of another type with same ID exists already.

    Args:
        table_type: table type
        table_id: table identifier

    """

    def __init__(self, table_type: str, table_id: str):
        KeyError(
            f"There is already a " f"{table_type} " f"table with ID " f"'{table_id}'."
        )
