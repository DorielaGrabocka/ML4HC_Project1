import numpy as np


def preprocess_x_pytorch(x: np.ndarray):
    """
    pytorch expects (N,C,L) input, whereas now it's (N,L,C)
    should also be float32
    """
    return np.transpose(x, (0, 2, 1)).astype(np.float32)


def preprocess_y_pytorch(y: np.ndarray):
    """
    pytorch expected format
    """
    return y.astype(np.int64)

#function needed to reshape input(tree algorithms)
def convert3Dto2D(x):
  return x.reshape(-1, x.shape[1])