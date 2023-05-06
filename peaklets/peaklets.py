import numpy as np
from numba import jit, prange

__all__ = [
    'pqpt','pqpt_movie','pk_trunc_para','pk_parabola','pkxform','pkxform_optimized',
]

import numpy as np
from numba import jit, prange
import numba.typed
from typing import Callable
import matplotlib.pyplot as plt
import celluloid

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
    Convex parabolic peaklets. The scale is FWHM of the parabola,
        which is the distance between roots divided by np.sqrt(2).
    
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
    return fscales/np.sqrt(2), pklets

import dataclasses
@dataclasses.dataclass
class PeakletXform:
    scales:  np.ndarray
    xform:   np.ndarray
    filters: np.ndarray
    pklets:  np.ndarray

def pkxform(data: np.ndarray, axis: int = -1, peaklet_func: Callable = pk_parabola) -> PeakletXform:
    """
    Positive Nonlinear Peak Transform ("peaklet transform") on multi-dimensional arrays.
    """    
    axis_positive = axis % data.ndim
    data = np.moveaxis(data, axis, -1)
    shape_we_need = data.shape
    data = np.reshape(data, (-1,data.shape[-1]))
    
    (scales, transform, filters, pklets) = _pkxform(data, peaklet_func)
    
    transform = np.reshape(transform, transform.shape[:1]+shape_we_need)
    transform = np.moveaxis(transform, -1, axis_positive+1)  # back to original data shape, with leading dim.
    filters = np.reshape(filters, filters.shape[:1]+shape_we_need)
    filters = np.moveaxis(filters, -1, axis_positive+1)  # back to original data shape, with leading dim.
    
    return PeakletXform(scales, transform, filters, pklets)


@jit(nopython=True, parallel=True) # normally set parallel=True
def _pkxform(data: np.ndarray, peaklet_func: Callable = pk_parabola):
    """
    Private function containing the parts of pkxform() that can be jitted.
    """
    scales,pklets = peaklet_func(data.shape[-1]) #   get scales and peaklets
    filters = np.empty((len(scales)+1,data.shape[0],data.shape[-1]))    
    transform = np.empty((len(scales),data.shape[0],data.shape[-1]))
    for k in prange(data.shape[0]):
        transform[:,k,:], filters[:,k,:] = pqpt(data[k,:], pklets, scales)
    return (scales, transform, filters, pklets)

def _frame_movie(ax, camera, data, residual, transform, mod_pk, scale, a, b):
    """
    Add a frame to the animation. Private, called by pqpt_movie().
    """
    Nt = len(residual)
    ax.set_xlabel('time (samples)')
    ax.set_ylabel('signal')
    ax.text(0.05,0.95,'Scale = {:.1f}'.format(scale), transform=ax.transAxes, verticalalignment='top')
    p0, = ax.plot(data,'b',label='data', linewidth=2)
    p1, = ax.plot(residual,'b:',label='residual', linewidth=2)
    p2 = ax.fill_between(np.arange(Nt), transform, color=(0.9,0.8,1), label='transform')
    p3, = ax.plot(np.arange(a,b), mod_pk, 'm', label='peaklet')
    ax.plot(np.array((a,b)), np.array((0,0)), 'm.')
    ax.legend(handles=[p0,p1,p2,p3])
    camera.snap()
    return

def pqpt_movie(data, pklets, scales):
    """
    1D Positive Quasi-linear Peak Transform ("peaklet transform") for making animations.
    """   
    Nt = len(data) # number of elements in data array
    Nscales = len(scales)
    transform = np.zeros((Nscales, Nt))
    residual = data.copy()
    fig,ax = plt.subplots()
    camera = celluloid.Camera(fig)
    for i in range(Nscales-1, 0, -1):
        pklet = pklets[i]
        Npk = len(pklet)
        # 3 loops for 3 cases as we slide pklet over data:
        for j0 in prange(-Npk//2, 0): # To give the same result as pqpt_optimized, use -Npk//2+1 ?!?!
            a = 0
            b  = j0 + Npk
            a_pk = - j0
            b_pk = a_pk + b - a # equivalently, Npk
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( residual[a:b] / pklet[a_pk:b_pk] )
            transform[i,a:b] = np.maximum(transform[i,a:b], mod_pk)
            _frame_movie(ax, camera, data, residual, transform[i,:], mod_pk, scales[i], a, b)
        for j0 in prange(0, Nt-Npk):
            a = j0
            b = j0 + Npk
            mod_pk = pklet * np.nanmin( residual[a:b] / pklet )
            transform[i,a:b] = np.maximum(transform[i,a:b], mod_pk)
            _frame_movie(ax, camera, data, residual, transform[i,:], mod_pk, scales[i], a, b)
        for j0 in prange(Nt-Npk, Nt-Npk//2):
            a = j0
            b = Nt
            a_pk = 0
            b_pk = a_pk + b - a
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( residual[a:b] / pklet[a_pk:b_pk] )
            transform[i,a:b] = np.maximum(transform[i,a:b], mod_pk)
            _frame_movie(ax, camera, data, residual, transform[i,:], mod_pk, scales[i], a, b)
        residual -= transform[i,:]
    transform[0,:] = residual
        
    filters = np.zeros((Nscales+1,Nt))
    filters[0,:] = data
    for i in range(1,Nscales+1):
        filters[i,:] = filters[i-1,:] - transform[i-1,:]
    
    transform = np.maximum(transform, 0.0)
    filters = np.maximum(filters, 0.0)
    animation = camera.animate()
    plt.close(fig)
    return animation, transform, filters
    
@jit(nopython=True, parallel=False) 
def pqpt(data, pklets, scales):
    """
    Positive Quasi-linear Peak Transform ("peaklet transform") for 1D arrays.
    """
    Nt = len(data) # number of elements in data array
    Nscales = len(scales)
    transform = np.zeros((Nscales, Nt))
    residual = data.copy()
    
    for i in range(Nscales-1, 0, -1):
        pklet = pklets[i]
        Npk = len(pklet)
        # 3 loops for 3 cases as we slide pklet over data:
        for j0 in prange(-Npk//2, 0): # To give the same result as pqpt_optimized, use -Npk//2+1 ?!?!
            a = 0
            b  = j0 + Npk
            a_pk = - j0
            b_pk = a_pk + b - a # equivalently, Npk
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( residual[a:b] / pklet[a_pk:b_pk] )
            transform[i,a:b] = np.maximum(transform[i,a:b], mod_pk)
        for j0 in prange(0, Nt-Npk):
            a = j0
            b = j0 + Npk
            mod_pk = pklet * np.nanmin( residual[a:b] / pklet )
            transform[i,a:b] = np.maximum(transform[i,a:b], mod_pk)
        for j0 in prange(Nt-Npk, Nt-Npk//2):
            a = j0
            b = Nt
            a_pk = 0
            b_pk = a_pk + b - a
            mod_pk = pklet[a_pk:b_pk] * np.nanmin( residual[a:b] / pklet[a_pk:b_pk] )
            transform[i,a:b] = np.maximum(transform[i,a:b], mod_pk)
        residual -= transform[i,:]
    transform[0,:] = residual
    
    filters = np.zeros((Nscales+1,Nt))
    filters[0,:] = data
    for i in range(1,Nscales+1):
        filters[i,:] = filters[i-1,:] - transform[i-1,:]
    
    transform = np.maximum(transform, 0.0)
    filters = np.maximum(filters, 0.0)
    return transform, filters


def pkxform_optimized(data: np.ndarray, axis: int = -1, peaklet_func: Callable = pk_parabola) -> PeakletXform:
    """
    Positive Nonlinear Peak Transform ("peaklet transform") on multi-dimensional arrays.
    """
    axis_positive = axis % data.ndim
    data = np.moveaxis(data, axis, -1)
    shape_we_need = data.shape
    data = np.reshape(data, (-1, data.shape[-1]))

    scales, pklets = peaklet_func(data.shape[-1])  # get scales and peaklets
    transform, filters = pqpt_optimized(data, pklets, scales)
    transform = np.maximum(transform, 0.0)
    filters = np.maximum(filters, 0.0)

    transform = np.reshape(transform, transform.shape[:1] + shape_we_need)
    transform = np.moveaxis(transform, -1, axis_positive + 1)  # back to original data shape, with leading dim.
    filters = np.reshape(filters, filters.shape[:1] + shape_we_need)
    filters = np.moveaxis(filters, -1, axis_positive + 1)  # back to original data shape, with leading dim.

    return PeakletXform(scales, transform, filters, pklets)


@jit(nopython=True, parallel=True)
def pqpt_optimized(
        data: np.ndarray,
        pklets,
        scales,
):
    Nt = data.shape[~0]
    Nscales = len(scales)

    transform = np.zeros((len(scales), data.shape[0], data.shape[-1]))

    filters = np.zeros((len(scales) + 1, data.shape[0], data.shape[-1]))
    filters[0, :] = data

    for k in prange(data.shape[0]):

        residual = data[k].copy()

        for i in range(Nscales - 1, 0, -1):

            pklet = pklets[i]
            Npk = len(pklet)
            Rpk = Npk // 2

            for j0 in range(0, Nt):
                min_j0 = np.inf
                for j_pk in range(0, Npk):
                    r_pk = j_pk - Rpk
                    j = j0 + r_pk
                    if 0 <= j < Nt:
                        val = residual[j] / pklet[j_pk]
                        if val < min_j0:
                            min_j0 = val

                for j_pk in range(0, Npk):
                    r_pk = j_pk - Rpk
                    j = j0 + r_pk
                    if 0 <= j < Nt:
                        mod_pk = pklet[j_pk] * min_j0
                        transform[i, k, j] = max(transform[i, k, j], mod_pk)

            residual -= transform[i, k, :]

        transform[0, k, :] = residual

        for i in range(1, Nscales + 1):
            filters[i, k, :] = filters[i - 1, k, :] - transform[i - 1, k, :]

    return transform, filters

