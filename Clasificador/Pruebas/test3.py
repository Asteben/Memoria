import pandas as pd
import numpy as np

x = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]])
print(x.shape)
def cero(temp):
    result = []
    for aux in temp:
        if aux[0] == 0:
            result.append(0)
        else:
            result.append(1)
    return result

print(sum(cero(x)))