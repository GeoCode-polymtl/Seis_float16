#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the FP32 dataset to test inversion in fp32 and fp16
"""

import os
import numpy as np
import sys
import h5py as h5
import shutil
from urllib.request import urlretrieve
from SeisCL import SeisCL

"""
    _______________________Constants for inversion_____________________________
"""
f0 = 7.5  # Center frequency of the source
dsrc = 2  # Put a source every dsrc grid cell
dg = 2
seisout = 2
nab = 128

filepath = os.getcwd() + '/'

"""
_______________________Download the velocity model_____________________________
"""
url = "http://sw3d.cz/software/marmousi/little.bin/velocity.h@"
if not os.path.isfile("velocity.h@"):
    urlretrieve(url, filename="velocity.h@")
vel = np.fromfile("velocity.h@", dtype=np.float32)
vp = np.transpose(np.reshape(np.array(vel), [2301, 751]))

"""
    ____________________________Coarsen grid___________________________________
"""
vp = vp[::4, ::4]
(NZ, NX) = vp.shape
NZ = int(NZ / 2) * 2
NX = int(NX / 2) * 2
vp = vp[:NZ, :NX]

vp2 = np.zeros([vp.shape[0] + 4+2 * nab, vp.shape[1] + 2 * nab])
vp2[nab+4:-nab, nab:-nab] = vp
vp2[0:nab+4, :] = 1500
for ii in range(vp2.shape[1]):
    vp2[-nab:, ii] = vp2[-nab - 1, ii]
for ii in range(vp2.shape[0]):
    vp2[ii, 0:nab] = vp2[ii, nab]
    vp2[ii, -nab:] = vp2[ii, -nab - 1]
vp = vp2
rho = vp * 0 + 2000
vs = vp * 0
models = {'vp':vp, 'vs':vs, 'rho':rho}

"""
_____________________Simulation constants input file___________________________
"""
seis = SeisCL()
seis.csts['N'] = np.array([vp.shape[0], vp.shape[1]])
seis.csts['ND'] = 2  # Flag for dimension. 3: 3D, 2: 2D P-SV,  21: 2D SH
seis.csts['dh'] = dh = 16  # Grid spatial spacing

seis.csts['dt'] = 6 * dh / (7 * np.sqrt(2) * np.max(vp)) * 0.85
seis.csts['NT'] = int(7 / seis.csts['dt'])  # Number of time steps
seis.csts['freesurf'] = 0  # Include a free surface at z=0: 0: no, 1: yes
seis.csts['FDORDER'] = 4  # Order of the finite difference stencil.
seis.csts['MAXRELERROR'] = 0  # Taylor coefficients
seis.csts['f0'] = f0  # Central frequency

seis.csts['abs_type'] = 2  # Absorbing boundary type: 2: Absorbing layer
seis.csts['nab'] = nab  # Width in grid points of the absorbing layer
seis.csts['abpc'] = 6  # Exponential decay of the absorbing layer
seis.csts['FP16'] = 1  # Create data in FP32

"""
_________________________Sources and receivers_________________________________
"""

gx = np.arange(seis.csts['nab'] + 5,
               seis.csts['N'][-1] - seis.csts['nab'] - 5)
gx = gx * dh
gy = gx * 0 + seis.csts['N'][1] // 2 * dh
gz = gx * 0 + (seis.csts['nab'] + 5) * dh

for ii in range(seis.csts['nab'] + 5,
                seis.csts['N'][-1] - seis.csts['nab'] - 5, dsrc):
    idsrc = seis.src_pos_all.shape[1]
    toappend = np.zeros((5, 1))
    toappend[0, :] = (ii) * dh
    toappend[1, :] = 0
    toappend[2, :] = (seis.csts['nab'] + 5) * seis.csts['dh']
    toappend[3, :] = idsrc
    toappend[4, :] = 100
    seis.src_pos_all = np.append(seis.src_pos_all, toappend, axis=1)

    toappend = np.stack([gx,
                         gx * 0,
                         gz,
                         gz * 0 + idsrc,
                         np.arange(0, len(gx)) + seis.rec_pos_all.shape[1] + 1,
                         gx * 0 + 2,
                         gx * 0,
                         gx * 0], 0)
    seis.rec_pos_all = np.append(seis.rec_pos_all, toappend, axis=1)

"""
___________________________Create data__________________________________
"""

file = filepath + "marmfp321"
filedin = file + "_din.mat"
if not os.path.isfile(filedin):
    seis.set_forward(seis.src_pos_all[3, :], models, withgrad=False)
    shutil.copy2(seis.workdir + "/" + seis.file_csts, file + "_csts.mat")
    shutil.copy2(seis.workdir + "/" + seis.file_model, file + "_model.mat")
    seis.execute()
    sys.stdout.write('Forward calculation completed \n')
    sys.stdout.flush()
    shutil.copy2(seis.workdir + "/" + seis.file_dout, filedin)
    datain = h5.File(filedin, 'a')
    datain['p'] = datain['pout']
    datain['src'] = np.transpose(seis.csts['src'])
    datain.close()


