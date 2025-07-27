import numpy as np


def sum_memory_usage(fields):
    """
    fields: a list of fields
    """

    memory_usage = 0
    for field in fields:
        f_np = field.to_numpy()
        memory_usage += f_np.nbytes

    return memory_usage
