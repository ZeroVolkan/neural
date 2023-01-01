import gzip
import struct
import numpy as np


def images_read(file: str) -> list[list[float]]:
    """Чтение ахрива с числами"""
    with gzip.open(file, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        rows, columns = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = np.reshape(data, (-1, 28, 28))
    return data


def labels_read(file: str) -> (list[float], list[int]):
    """Чтение ахрива с проверками"""
    with gzip.open(file, "rb") as f:
        magic, number = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
    return data
