#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of FP16 and FP32 for the Marmousi-II model
"""
from urllib.request import urlretrieve
import tarfile
import numpy as np
import os
import sys

import segyio
import shutil
from compare_accuracy import compare_accuracy
from plot_comparison import plot_comparison

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 7})
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
    _______________________Constants for modeling_____________________________
"""
f0 = 40  # Center frequency of the source
srcx = 7000 # x position of the source
tmax = 5

"""
_______________________Download the velocity model_____________________________
"""
url = "https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz"
if not os.path.isfile("elastic-marmousi-model.tar.gz"):
    urlretrieve(url, filename="elastic-marmousi-model.tar.gz")
    tar = tarfile.open("elastic-marmousi-model.tar.gz", "r:gz")
    tar.extractall()
    tar.close()

models_segy = {
    'vp': './elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy',
    'vs': './elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy',
    'rho': './elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy'}

models_tar = {
    'vp': './elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz',
    'vs': './elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz',
    'rho': './elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy.tar.gz'}

models = {
    'vp': None,
    'vs': None,
    'rho': None}


for par in models:
    if not os.path.isfile(models_segy[par]):
        tar = tarfile.open(models_tar[par], "r:gz")
        tar.extractall(path="./elastic-marmousi-model/model")
        tar.close()
    with segyio.open(models_segy[par], "r", ignore_geometry=True) as segy:
        models[par] = [segy.trace[trid] for trid in range(segy.tracecount)]
        models[par] = np.transpose(np.array(models[par])[:, :])

(NZ, NX) = models['rho'].shape
NZ = int(NZ / 2) * 2
NX = int(NX / 2) * 2
for par in models:
    models[par] = models[par][:NZ, :NX]
models["rho"] = models["rho"] * 1000
dh = 1.25

"""
_____________________________Plot models______________________________________
"""
fig, axs = plt.subplots(3, 1, figsize=(9 / 2.54, 13 / 2.54))
ims = {}
units = {'vp': 'm/s', 'vs': 'm/s', 'rho': 'kg/m$^3$'}
titles = {'vp': 'a)', 'vs': 'b)', 'rho': 'c)'}
params = ['vp', 'vs', 'rho']
for ii, par in enumerate(params):
    ims[par] = axs[ii].imshow(models[par] / 1000, interpolation='bilinear',
                              extent=[0, (NX + 1) * dh / 1000 / 2,
                                      (NZ + 1) * dh / 1000, 0])
    axs[ii].set_xlabel('x (km)')
    axs[ii].set_ylabel('z (km)')
    axs[ii].set_title(titles[par])
    axs[ii].set_xticks(np.arange(0, 9, 0.5))
    axs[ii].set_yticks(np.arange(0, 4, 0.5))
    axs[ii].set_xticklabels([str(el) for el in np.arange(0, 18)])
    divider = make_axes_locatable(axs[ii])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clr = plt.colorbar(ims[par], cax=cax)
    cax.xaxis.tick_top()
    cax.set_xlabel(units[par], labelpad=8)
    cax.xaxis.set_label_position('top')
plt.tight_layout()
plt.savefig('marmousiII.eps', dpi=300)

"""
_____________________Perform the comparison ___________________________
"""
compare_accuracy("marm2", models, f0, srcx, tmax, dh)

"""
    _____________________Plot the figure ___________________________
"""
plot_comparison("marm2", 10, 20, 1, 16, 0.2)

