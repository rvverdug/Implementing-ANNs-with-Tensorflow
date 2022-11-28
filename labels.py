"""This module holds utility functions for generating labels."""
from twin_networks import Subtask


def get_label_sum_greater_5(a: int, b: int) -> bool:
    return a + b >= 5


def get_label_a_minus_b(a: int, b: int):
    return a - b


def get_label_function(subtask: Subtask):
    """Returns the correct label-function for a specific subtask."""
    if Subtask.DIFFERENCE:
        return get_label_a_minus_b
    if Subtask.LARGER_FIVE:
        return get_label_sum_greater_5
