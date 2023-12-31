import sys
import numpy as np
import torch

########################################################################################################
#####  Helper function to get the list of preference vectors k for a given sparsity coefficient k0 #####
########################################################################################################

def get_pref_vects(k0):
    pref_vects = {"0":[[0.0, 0.1, 0.9], [0.0, 0.9, 0.1],
       [0.0, 0.2, 0.8], [0.0, 0.8, 0.2],
       [0.0, 0.3, 0.7], [0.0, 0.7, 0.3],
       [0.0, 0.4, 0.6], [0.0, 0.6, 0.4],
       [0.0, 0.5, 0.5], [0.0, 0.55, 0.45], [0.0, 0.45, 0.55],
       [0.0, 0.65, 0.35], [0.0, 0.35, 0.65],
       [0.0, 0.75, 0.25], [0.0, 0.25, 0.75],
       [0.0, 0.85, 0.15], [0.0, 0.15, 0.85],
       [0.0, 0.95, 0.05], [0.0, 0.05, 0.95]],
       "0.1":[[0.1, 0.1, 0.8], [0.1, 0.8, 0.1],
       [0.1, 0.2, 0.7], [0.1, 0.7, 0.2],
       [0.1, 0.3, 0.6], [0.1, 0.6, 0.3],
       [0.1, 0.4, 0.5], [0.1, 0.5, 0.4], [0.1, 0.45, 0.45],
       [0.1, 0.55, 0.35], [0.1, 0.35, 0.55],
       [0.1, 0.65, 0.25], [0.1, 0.25, 0.65],
       [0.1, 0.75, 0.15], [0.1, 0.15, 0.75],
       [0.1, 0.85, 0.05], [0.1, 0.05, 0.85]],
       "0.01":[[1e-2, 0.09, 0.9], [1e-2, 0.9, 0.09],
       [1e-2, 0.19, 0.8], [1e-2, 0.8, 0.19],
       [1e-2, 0.29, 0.7], [1e-2, 0.7, 0.29],
       [1e-2, 0.39, 0.6], [1e-2, 0.6, 0.39],
       [1e-2, 0.49, 0.5], [1e-2, 0.5, 0.49], [1e-2, 0.59, 0.4], [1e-2, 0.4, 0.59],
       [1e-2, 0.69, 0.3], [1e-2, 0.3, 0.69],
       [1e-2, 0.79, 0.2], [1e-2, 0.2, 0.79],
       [1e-2, 0.89, 0.1], [1e-2, 0.1, 0.89]],
       "0.001":[[1e-3, 0.099, 0.9], [1e-3, 0.9, 0.099],
       [1e-3, 0.199, 0.8], [1e-3, 0.8, 0.199],
       [1e-3, 0.299, 0.7], [1e-3, 0.7, 0.299],
       [1e-3, 0.399, 0.6], [1e-3, 0.6, 0.399],
       [1e-3, 0.499, 0.5], [1e-3, 0.5, 0.499], [1e-3, 0.599, 0.4], [1e-3, 0.4, 0.599],
       [1e-3, 0.699, 0.3], [1e-3, 0.3, 0.699],
       [1e-3, 0.799, 0.2], [1e-3, 0.2, 0.799],
       [1e-3, 0.899, 0.1], [1e-3, 0.1, 0.899]],
       "0.0001":[[1e-4, 0.0999, 0.9], [1e-4, 0.9, 0.0999],
       [1e-4, 0.1999, 0.8], [1e-4, 0.8, 0.1999],
       [1e-4, 0.2999, 0.7], [1e-4, 0.7, 0.2999],
       [1e-4, 0.3999, 0.6], [1e-4, 0.6, 0.3999],
       [1e-4, 0.4999, 0.5], [1e-4, 0.5, 0.4999],[1e-4, 0.5999, 0.4], [1e-4, 0.4, 0.5999],
       [1e-4, 0.6999, 0.3], [1e-4, 0.3, 0.6999],
       [1e-4, 0.7999, 0.2], [1e-4, 0.2, 0.7999],
       [1e-4, 0.8999, 0.1], [1e-4, 0.1, 0.8999]]}
    
    if k0 not in [0, 1e-1, 1e-2, 1e-3, 1e-4]:  raise ValueError("Sparsity coefficient should be in [0, 1e-1, 1e-2, 1e-3, 1e-4] !")
    
    ks = pref_vects[str(k0)]

    print(f"{len(ks)} preference vectrors for k0 = {k0} !")

    return ks

if __name__ == "__main__":
    k0 = 1e-1
    print(f"\nPreference vectors:\n {get_pref_vects(k0)}\n")