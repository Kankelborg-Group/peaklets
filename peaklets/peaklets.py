import numpy as np
from numba import jit, prange

__all__ = [
    'pnpt',
]

### njit yields an order of magnitude improvement ###
### !!! parallel=True does not like one line of my code. I can't figure out why !!! ###

@jit(nopython=True, parallel=False)
def pnpt(data):
    """
    Positive Nonlinear Peak Transform
    """
    Nt = len(data) # number of elements in data array
    epsilon = 1e-15 
        # numpy.finfo not available to numba, and numba numerics are a little
        # less well behaved than numpy numerics. So this is set a bit larger
        # than numpy machine epsilon to be safe.
    almost = 1. - epsilon*10
    Nscales = int(np.floor(2*(np.log2(Nt))))
        # Estimate the number of scales to span Nt. Could be 1 too high.
    fscales = np.sqrt(2)**(2+np.arange(Nscales))
        # Array of scale sizes, as floating point
    Npk = 2*np.floor(almost*(fscales/2)).astype(np.int64)+1
        # Array of sizes for the peaklet arrays
    if Npk[-1]>Nt: # If true, Nscales is 1 too many.
        Nscales -= 1
        fscales = np.delete(fscales,-1)
        Npk = np.delete(Npk,-1)
    
    pklets = [np.array((1.,)),] # List of all the peaklet arrays.
    filters = np.zeros((Nscales+1, Nt))
    filters[0,:] = data # The narrowest scale is this easy.
    for i in prange(1, Nscales):
        x = np.arange(Npk[i]) - Npk[i]//2
        pklet = (1 + 2*x/fscales[i])*(1 - 2*x/fscales[i]) #+ 1
        pklets.append(pklet)
        
        # 3 loops for 3 cases as we slide pklet over data:
        for j0 in prange(-Npk[i]//2, 0):
            a = 0
            b  = j0 + Npk[i]
            a_pk = - j0
            b_pk = a_pk + b - a # equivalently, Npk[i]
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( data[a:b] / pklet[a_pk:b_pk] )
            filters[i,a:b] = np.maximum(filters[i,a:b], mod_pk)
        for j0 in prange(0, Nt-Npk[i]):
            a = j0
            b = j0 + Npk[i]
            mod_pk = pklet * np.nanmin( data[a:b] / pklet )
            filters[i,a:b] = np.maximum(filters[i,a:b], mod_pk)
        for j0 in prange(Nt-Npk[i], Nt-Npk[i]//2):
            a = j0
            b = Nt
            a_pk = 0
            b_pk = a_pk + b - a
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( data[a:b] / pklet[a_pk:b_pk] )
            filters[i,a:b] = np.maximum(filters[i,a:b], mod_pk)

    transform = np.empty((Nscales,Nt))
    for i in range(Nscales, 0, -1): # this loop picks up the DC using transform[Nscales]
        filters[i-1,:] = np.maximum(filters[i,:], filters[i-1,:]) # Fix NEGATIVES PROBLEM
        transform[i-1,:] = filters[i-1,:]-filters[i,:]
        
    return fscales, transform, filters, pklets