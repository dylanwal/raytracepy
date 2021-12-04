import core
import numpy as np


help(core)
help(core.normalise)

a = np.array([1,2,3], dtype="float64")
print(core.normalise(a))
