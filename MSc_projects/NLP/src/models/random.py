from models.model_configurations import RandomConfig
import numpy as np

class Model:
    def __init__(self):
        pass

    def __call__(self, dataset) -> np.array:
        return np.random.randint(RandomConfig.num_classes, size=len([dataset]))


