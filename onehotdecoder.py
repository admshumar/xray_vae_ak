import numpy as np

class OneHotDecoder:
    """
    Takes in a list of one-hot encoded vectors and return a class vector.
    """

    def __init__(self, labels):
        labels = np.asarray(labels)
        self.labels = labels
        print('num labels is ', len(labels))
        self.number_of_samples = len(labels)

    def decode_to_multiclass(self):
        return np.asarray([np.where(self.labels[i] == 1)[0][0]
                           for i in range(self.number_of_samples)])
