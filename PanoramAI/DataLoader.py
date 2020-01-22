import numpy as np

def load_data(base_path = None, filename = None):
    if not base_path:
        base_path = "/Users/tmcclintock/Github/PanoramAI/data/"
    if not filename:
        M = 32
        N = 256
        Npics = 10000
        channels = "rgb"
        filename = f"panoramas_{M}x{N}_Npics{Npics}_{channels}.npy"
    return np.load(base_path + filename)
