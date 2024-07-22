# NOT REQUIRED. IS REPLACED BY NUMPY SEARCHSORTED

def locate_index(xx, x):
    """
    The `locate_index` function uses binary search to find the index at which a given value `x` should be inserted into a sorted list `xx`.

    :param xx: The parameter "xx" is a list of numbers
    :param x: The parameter `x` is the value that we are trying to locate in the list `xx`
    :return: the lower index of the element x in the list xx.

    Example Usage:

    xx = [1.0, 2.0, 3.0, 4.0, 5.0]: Initializes a sorted list xx.
    x_value = 3.5: Sets a test value to search for.
    result = locate(xx, x_value): Calls the locate function with the provided values.
    Returns the result, which is 3 indicating the index where x_value would be inserted in the sorted list.
    """

    n = len(xx)
    lower_index = 0
    upper_index = n + 1

    while upper_index - lower_index > 1:
        mid_index = (upper_index + lower_index) // 2

        if (xx[n - 1] > xx[0]) == (x > xx[mid_index - 1]):
            lower_index = mid_index
        else:
            upper_index = mid_index

    return lower_index
