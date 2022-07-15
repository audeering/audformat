import typing

import pandas as pd


def compatible_index(
        objs: typing.Sequence[typing.Union[pd.Index, pd.Series, pd.DataFrame]],
) -> bool:
    r"""Check if index objects are compatible.

    Two index objects are compatible
    if they have the same number of levels
    and share the same level names and dtypes.

    Args:
        objs: index objects

    Returns:
        ``True`` if index objects are compatible, otherwise ``False``

    Examples:
        >>> idx1 = pd.Index([1, 2, 3], name='l')
        >>> idx2 = pd.MultiIndex.from_arrays([[10, 20]], names=['l'])
        >>> compatible_index([idx1, idx2])
        True
        >>> compatible_index([idx1, pd.Series(['a', 'b'], index=idx2)])
        True
        >>> idx3 = idx2.set_names(['L'])
        >>> compatible_index([idx2, idx3])
        False
        >>> idx4 = idx2.set_levels([['10', '20']])
        >>> compatible_index([idx2, idx4])
        False
        >>> idx5 = pd.MultiIndex.from_arrays([[1], ['a']], names=['l1', 'l2'])
        >>> compatible_index([idx2, idx5])
        False
        >>> idx6 = pd.MultiIndex.from_arrays([['a'], [1]], names=['l2', 'l1'])
        >>> compatible_index([idx5, idx6])
        False

    """
    objs = [obj if isinstance(obj, pd.Index) else obj.index for obj in objs]

    # check names
    levels = set([obj.names for obj in objs])
    if len(levels) > 1:
        return False

    # check dtypes
    dtypes = set()
    for obj in objs:
        if isinstance(obj, pd.MultiIndex):
            dtypes.add(tuple(obj.dtypes))
        else:
            dtypes.add(tuple([obj.dtype]))
    if len(dtypes) > 1:
        return False

    return True
