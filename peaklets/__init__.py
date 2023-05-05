"""
.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    import peaklets as pk

    t = np.arange(100)
    t2 = np.maximum(t-45,0)
    signal1 = t2*np.exp(-t2/2)
    signal2 = 1/np.cosh((t-40)/10)
    signal3 = 1.5 + 1e-5*(t-35)*(100-t)*t

    signal = signal1+signal2+signal3  # total signal

    Nt = len(signal)
    scales, pklets = pk.pk_parabola(Nt)
    animation, transform, filters = pk.pqpt_movie(signal, pklets, scales)

    plt.rcParams["animation.embed_limit"] = 30
    HTML( animation.to_jshtml(fps=30, ) )

"""
from .peaklets import *
