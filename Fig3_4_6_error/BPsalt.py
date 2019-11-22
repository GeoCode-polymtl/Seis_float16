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
f0 = 10  # Center frequency of the source
srcx = 10000  # x position of the source
tmax = 17

"""
_______________________Download the velocity model_____________________________
"""

models_url = {
    'vp': 'http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz',
    'rho': 'http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/density_z6.25m_x12.5m.segy.gz',
    'Salt': 'http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_saltindex.segy.gz',
    'water': 'http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_wbindex.segy.gz'}

models_gz = {
    'vp': 'vel_z6.25m_x12.5m_exact.segy.gz',
    'rho': 'density_z6.25m_x12.5m.segy.gz',
    'Salt': 'vel_z6.25m_x12.5m_saltindex.segy.gz',
    'water': 'vel_z6.25m_x12.5m_wbindex.segy.gz'}

models_segy = {
    'vp': 'vel_z6.25m_x12.5m_exact.segy',
    'rho': 'density_z6.25m_x12.5m.segy',
    'Salt': 'vel_z6.25m_x12.5m_saltindex.segy',
    'water': 'vel_z6.25m_x12.5m_wbindex.segy'}

models_h5 = {
    'vp': 'vel_z6.25m_x12.5m_exact.mat',
    'rho': 'density_z6.25m_x12.5m.mat',
    'Salt': 'vel_z6.25m_x12.5m_saltindex.mat',
    'water': 'vel_z6.25m_x12.5m_wbindex.mat'}


"""
____________Transform the velocity model to have an even grid spacing__________
"""

models = {'vp': None, 'rho': None}

for par in models:
    if not os.path.isfile(models_h5[par]):
        #Download the model if necessary
        if not os.path.isfile(models_segy[par]):
            urlretrieve(models_url[par], models_gz[par])
            with gzip.open(models_gz[par], 'rb') as infile:
                with open(models_segy[par], 'wb') as outfile:
                    for line in infile:
                        outfile.write(line)
            os.remove(models_gz[par])
        with segyio.open(models_segy[par], "r", ignore_geometry=True) as segy:
            models[par] = [segy.trace[trid] for trid in range(segy.tracecount)]
            models[par] = np.transpose(np.array([segy.trace[trid]
                                                 for trid in
                                                 range(segy.tracecount)]))
            # Interpolate to have an even grid spacing
            models[par] = models[par][:, :]
            gz, gx = np.mgrid[:models[par].shape[0], :models[par].shape[1]]
            x = np.arange(0, models[par].shape[1], 1)
            z = np.arange(0, models[par].shape[0], 1)
            interpolator = intp.interp2d(x, z, models[par])
            xi = np.arange(0, models[par].shape[1], 0.5)
            zi = np.arange(0, models[par].shape[0], 1)
            models[par] = interpolator(xi, zi)
            h5mat.savemat(models_h5[par], {par: models[par]},
                          appendmat=False,
                          format='7.3',
                          store_python_metadata=True,
                          truncate_existing=True)
    else:
        models[par] = h5mat.loadmat(models_h5[par])[par]

(NZ, NX) = models['rho'].shape
NZ = int(NZ / 2) * 2
NX = int(NX / 2) * 2
for par in models:
    models[par] = models[par][:NZ, :NX]
models['rho'] *= 1000
(NZ, NX) = models['rho'].shape
dh = 12.5 / 2.0


"""
_____________________________Plot models______________________________________
"""

fig, axs = plt.subplots(2, 1, figsize=(9 / 2.54, 8 / 2.54))
ims = {}
units = {'vp': 'm/s', 'rho': 'kg/m$^3$'}
titles = {'vp': 'a)', 'rho': 'b)'}
params = ['vp', 'rho']
for ii, par in enumerate(params):
    ims[par] = axs[ii].imshow(models[par] / 1000, interpolation='bilinear',
                              extent=[0, (NX + 1) * dh / 1000 / 2,
                                      (NZ + 1) * dh / 1000, 0])
    axs[ii].set_xlabel('x (km)')
    axs[ii].set_ylabel('z (km)')
    axs[ii].set_title(titles[par])
    divider = make_axes_locatable(axs[ii])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clr = plt.colorbar(ims[par], cax=cax)
    axs[ii].set_xticks(np.arange(0, 31, 5))
    axs[ii].set_yticks(np.arange(0, 13, 2))
    axs[ii].set_xticklabels([str(el) for el in np.arange(0, 61, 10)])
    cax.xaxis.tick_top()
    cax.set_xlabel(units[par], labelpad=8)
    cax.xaxis.set_label_position('top')
plt.tight_layout()
plt.savefig('BPmodel.eps', dpi=300)

models['vs'] = models['vp'] * 0

"""
    _____________________Perform the comparison ___________________________
    """
compare_accuracy("BPmodel", models, f0, srcx, tmax, dh)

"""
    _____________________Plot the figure ___________________________
    """
plot_comparison("BPmodel", 50, 20, 2, 63, 0.8)
