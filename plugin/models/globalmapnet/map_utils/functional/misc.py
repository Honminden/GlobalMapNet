import datetime


def unified_timestamp():
    """
    Returns the current timestamp in milliseconds.

    Returns:
        int: The current timestamp in milliseconds.
    """
    return int(datetime.datetime.now().timestamp() * 1000)


def init_map_element_from(old_map_element, category=None, coords=None, details=None):
    """
    Initialize a map element from an old map element. If fields are not specified, they will be copied from the old map element.

    Args:
        old_map_element (dict): The old map element.
        category (str): The category of the new map element.
        coords (list of coords): The coords of the new map element.
        details (dict): The details of the new map element.

    Returns:
        dict: The new map element.
    """
    new_map_element = dict()
    if category is None:
        category = old_map_element['category']
    if coords is None:
        coords = old_map_element['coords']
    if details is None and 'details' in old_map_element.keys():
        details = old_map_element['details']

    new_map_element['category'] = category
    new_map_element['coords'] = coords
    if details is not None:
        new_map_element['details'] = details

    return new_map_element