def multiply(x, y):
    """
    Function used to multiply two values with the required signature.

    Parameters:
    x (float) : Value 1.
    y (float) : Value 2.

    Returns:
    float : x * y
    """
    return x * y


def subtract(x, y):
    """
    Function used to subtract two values with the required signature.

    Parameters:
    x (float) : Value 1.
    y (float) : Value 2.

    Returns:
    float : x - y
    """
    return x - y


def addition(x, y):
    """
    Function used to add two values with the required signature.

    Parameters:
    x (float) : Value 1.
    y (float) : Value 2.

    Returns:
    float : x + y
    """
    return x + y


def safe_divide(x, y):
    """
    Function used to safely divide two values with the required signature.
    Should there be a division by 0, the maximum of the two values is returned.

    Parameters:
    x (float) : Value 1.
    y (float) : Value 2.

    Returns:
    float : x / y OR max(x, y) should y == 0.
    """
    try:
        return x / y
    except FloatingPointError:
        return max(x, y)
    except ZeroDivisionError:
        return max(x, y)


def euclidian(x, y):
    """
    Function used to determine the Euclidian distance between two vector values.

    Parameters:
    x (Iterable) : An iterable containing the first vector values.
    y (Iterable) : An iterable containing the second vector values.

    Returns:
    float : The Euclidian distance between x and y.
    """
    if len(x) != len(y):
        raise RuntimeError('Mismatched Shapes')
    s = 0
    for i, j in zip(x, y):
        s += (i - j) ** 2
    return s ** 0.5


def safe_divide(x, y):
    """
    Function used to safely divide two values with the required signature.
    Should there be a division by 0, the maximum of the two values is returned.

    Parameters:
    x (float) : Value 1.
    y (float) : Value 2.

    Returns:
    float : x / y OR max(x, y) should y == 0.
    """
    try:
        return x / y
    except FloatingPointError:
        return max(x, y)
    except ZeroDivisionError:
        return max(x, y)

def less_than(x, y):
    """
    Function used to compare two values with the required signature.

    Parameters:
    x (float) : Value 1.
    y (float) : Valie 2.

    Returns:
    float : 1 if x < y, else 0.
    """
    return x < y


default_operators = ['multiply', 'subtract', 'addition', 'safe_divide', 'less_than']