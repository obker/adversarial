import numpy as np
import gzip

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)

    return labels

def load_data():
    x_train = extract_data("datas/train-images-idx3-ubyte.gz", 60000)
    y_train = extract_labels("datas/train-labels-idx1-ubyte.gz", 60000)
    x_test = extract_data("datas/t10k-images-idx3-ubyte.gz", 10000)
    y_test = extract_labels("datas/t10k-labels-idx1-ubyte.gz", 10000)

    return (x_train, y_train),(x_test, y_test)


