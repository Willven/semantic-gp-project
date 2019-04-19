def multiply(x, y):
    return x * y

def subtract(x, y):
    return x - y

def addition(x, y):
    return x + y

def safe_divide(x, y):
    try:
        return x / y
    except FloatingPointError:
        return max(x, y)
    except ZeroDivisionError:
        return max(x, y)
            
def euclidian(x, y):
    if len(x) != len(y):
        raise RuntimeError('Mismatched Shapes')
    s = 0
    for i, j in zip(x, y):
        s += (i-j)**2
    return s**0.5

def less_than(x, y):
    return int(x < y)
    
default_operators = ['multiply', 'subtract', 'addition', 'safe_divide', 'less_than']