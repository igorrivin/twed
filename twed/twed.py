import numpy as np
import pandas as pd
from .utils import twedcore

class tweddist(object):
    def __init__(self, nu, lam):
        self.nu = nu
        self.lam = lam

    def __call__(self, dfA, dfB, matrix=False):
        df1 = dfA.astype('float64')
        df2 = dfB.astype('float64')
        fullres = twedcore(df1.values, np.array(df1.index, dtype='float64'), 
                df2.values, np.array(df2.index, dtype='float64'), 
                self.nu, self.lam)
        if matrix is False:
            return fullres[0]
        else:
            return fullres
    