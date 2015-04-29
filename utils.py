import numpy as np

def evenly_sample_ts(ts):
    return np.linspace(ts.min(), ts.max(), len(ts))
