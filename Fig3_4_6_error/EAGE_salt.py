#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP model.
"""

from urllib.request import urlretrieve
import gzip
import os
import numpy as np
import sys
import time
import segyio
import hdf5storage as h5mat
import shutil
from scipy import interpolate as intp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
from mpl_toolkits.axes_grid1 import make_axes_locatable
from compare_accuracy import compare_accuracy
from plot_comparison import plot_comparison

"""
    _______________________Constants for inversion_____________________________
"""
f0 = 9  # Center frequency of the source
srcx = 740  # x position of the source
tmax = 5
dt=0.0015
FDORDER = 12
MAXRELERROR = 1

"""
_______________________Download the velocity model_____________________________
"""

models_url = {
    'vp': 'http://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_salt/SEG_C3NA_Velocity.sgy'}


models_segy = {
    'vp': 'SEG_C3NA_Velocity.sgy'}

models_h5 = {
    'vp': 'SEG_C3NA_Velocity.mat'}

#From segy headers
NZ = 201
NY = 676
NX = 676
dh = 20

models = {'vp': None}

for par in models:
    if not os.path.isfile(models_h5[par]):
        if not os.path.isfile(models_segy[par]):
            urlretrieve(models_url[par], models_segy[par])
        with segyio.open(models_segy[par], "r", ignore_geometry=True) as segy:
            models[par] = [segy.trace[trid] for trid in range(segy.tracecount)]
            models[par] = np.transpose(np.array([segy.trace[trid]
                                                 for trid in
                                                 range(segy.tracecount)]))
            models[par] = np.reshape(models['vp'], [NZ, NY, NX])
            h5mat.savemat(models_h5[par], {par: models[par]},
                          appendmat=False,
                          format='7.3',
                          store_python_metadata=True,
                          truncate_existing=True)
    else:
        models[par] = h5mat.loadmat(models_h5[par])[par]

"""
____________Transform the velocity model to have an even grid spacing__________
"""

nab = 32
pre = 11
N = models['vp'].shape
models['vp'] = np.concatenate([np.zeros([pre+nab, N[1], N[2]])+1500,
                               models['vp']], axis=0)
N = models['vp'].shape
NZ, NY, NX = N
models['vs'] = models['vp'] * 0
models['rho'] = models['vp'] * 0 + 2000

"""
    _____________________Perform the comparison ___________________________
"""
compare_accuracy("EAGE_salt", models, f0, srcx, tmax, dh, dt,
                    FDORDER=FDORDER, MAXRELERROR=MAXRELERROR, sz=5, gz=5)

"""
    _____________________Plot the figure ___________________________
"""
plot_comparison("eage_seg", 50, 1, 1, 12.5, 0.2, clip=0.001)
