import numpy as np
from numba import jit, prange

__all__ = [
    'pnpt','pk_trunc_para','pk_parabola'
]

import numpy as np
from numba import jit, prange
import numba.typed

@jit(nopython=True, parallel=False)
def pk_trunc_para(Nt):
    """
    Truncated parabolic peaklets. The scale is interpreted as a full width at half max.
    All scales except the smallest are even integers, so that the end points are at
    exactly half maximum for the truncated parabolae.
    
    Input:
        Nt, the length of the time series to be transformed.
    Output:
        sc, a 1D numpy integer array of scales.
        pk, a list of 1D numpy float arrays, containing the peaklet
            functions associated with each element of sc. Note that
            len(pk[i]) = 1+sc[i].
    """
    scales=[1,2,4,]
    next_scale = 6
    if Nt <= next_scale:
        raise Exception("Require Nt > 6.")
    while next_scale < Nt:
        scales.append(next_scale)
        next_scale = scales[-1] + scales[-3]
    Nscales = len(scales)
    peaklets = [np.array((1.,)),]
    for i in prange(1,Nscales):
        x = np.arange(1+scales[i]) - scales[i]//2 # coordinate centered within peaklet
        peaklet = 0.5 + 0.5*(1. + 2.*x/scales[i])*(1. - 2.*x/scales[i])
        peaklets.append(peaklet)
    return np.array(scales), peaklets

@jit(nopython=True, parallel=False)
def pk_parabola(Nt):
    """
    Convex parabolic peaklets. The scale is roughly FWHM/sqrt(2).
    
    Input:
        Nt, the length of the time series to be transformed.
    Output:
        sc, a 1D numpy integer array of scales.
        pk, a list of 1D numpy float arrays, containing the peaklet
            functions associated with each element of sc. Note that
            len(pk[i]) = 1+sc[i].
    
    """
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
    
    # pklets = [np.array((1.,)),] # List of all the peaklet arrays.
    pklets = numba.typed.List([np.array((1.,)),])
    for i in prange(1, Nscales):
        x = np.arange(Npk[i]) - Npk[i]//2
        pklet = (1. + 2.*x/fscales[i])*(1. - 2.*x/fscales[i])
        pklets.append(pklet)
    return fscales, pklets

import dataclasses
@dataclasses.dataclass
class PeakletXform:
    scales:  np.ndarray
    xform:   np.ndarray
    filters: np.ndarray
    pklets:  np.ndarray
    


from typing import Callable
@jit(nopython=True, parallel=True)
def pkxform(data: np.ndarray, axis: int = -1, peaklet_func: Callable = pk_parabola) -> PeakletXform:
    """
    Peaklet transform on multi-dimensional arrays
    """
    axis_positive = axis % data.ndim
    data = np.moveaxis(data, axis, -1)
    shape_we_need = data.shape
    data = np.reshape(data, (-1,data.shape[-1]))
    scales,pklets = peaklet_func(Nt) #   get scales and peaklets
    filters = np.empty((len(scales)+1,data.shape[0],data.shape[-1]))    
    transform = np.empty((len(scales),data.shape[0],data.shape[-1]))
    for k in prange(data.shape[0]):
        transform[:,k,:], filters[:,k,:] = pnpt(data[k,:], pklets, scales)
    transform = np.reshape(transform, transform.shape[-1:]+shape_we_need)
    transform = np.moveaxis(transform, -1, axis_positive+1)  # back to original data shape, with leading dim.
    filters = np.reshape(filters, filters.shape[-1:]+shape_we_need)
    filters = np.moveaxis(filters, -1, axis_positive+1)  # back to original data shape, with leading dim.
    return PeakletXform(scales, transform, filters, pklets)

### njit yields an order of magnitude improvement ###
@jit(nopython=True, parallel=False)
def pnpt(data, pklets, scales):
    """
    Positive Nonlinear Peak Transform
    """
    Nt = len(data) # number of elements in data array
    Nscales = len(scales)
    filters = np.zeros((Nscales+1, Nt))
    filters[0,:] = data # The narrowest scale is this easy.

    for i in prange(1, Nscales):
        pklet = pklets[i]
        Npk = len(pklet)
        # 3 loops for 3 cases as we slide pklet over data:
        for j0 in prange(-Npk//2, 0):
            a = 0
            b  = j0 + Npk
            a_pk = - j0
            b_pk = a_pk + b - a # equivalently, Npk
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( data[a:b] / pklet[a_pk:b_pk] )
            filters[i,a:b] = np.maximum(filters[i,a:b], mod_pk)
        for j0 in prange(0, Nt-Npk):
            a = j0
            b = j0 + Npk
            mod_pk = pklet * np.nanmin( data[a:b] / pklet )
            filters[i,a:b] = np.maximum(filters[i,a:b], mod_pk)
        for j0 in prange(Nt-Npk, Nt-Npk//2):
            a = j0
            b = Nt
            a_pk = 0
            b_pk = a_pk + b - a
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( data[a:b] / pklet[a_pk:b_pk] )
            filters[i,a:b] = np.maximum(filters[i,a:b], mod_pk)

    transform = np.empty((Nscales,Nt))
    for i in range(Nscales, 0, -1): # this loop picks up the DC using transform[Nscales]
        # filters[i-1,:] = np.maximum(filters[i,:], filters[i-1,:]) # Fix NEGATIVES PROBLEM
        for j in prange(filters.shape[-1]):
            filters[i-1,j] = max(filters[i,j], filters[i-1,j])
        transform[i-1,:] = filters[i-1,:]-filters[i,:]
    
    return transform, filters