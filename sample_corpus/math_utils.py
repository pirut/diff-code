def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    result = 0
    for _ in range(b):
        result += a
    return result


def divide(a, b):
    if b == 0:
        raise ValueError("division by zero")
    return a / b


def factorial(n):
    total = 1
    for value in range(2, n + 1):
        total *= value
    return total

