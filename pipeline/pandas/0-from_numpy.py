#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
    col= [chr(65 + i) for i in range(array.shape[1])]
    df = pd.DataFrame(array, columns=col)
    return df
print ('aaaaaa')