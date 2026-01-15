import numpy as np


def sigmoid(eta):
    out = np.zeros_like(eta)

    pos = eta >= 0
    out[pos] = 1 / (1 + np.exp(-eta[pos]))

    neg = ~pos
    out[neg] = np.exp(eta[neg]) / (1 + np.exp(eta[neg]))

    return out