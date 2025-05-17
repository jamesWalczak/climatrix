def ensure_list_or_slice(obj):
    """
    Ensure the input is a list or slice.
    """
    if isinstance(obj, slice):
        return obj
    elif isinstance(obj, list):
        return obj
    return [obj]
