#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform RTM on marmousi
"""

import os
import numpy as np
import h5py as h5
from scipy.ndimage.filters import gaussian_filter
import sys
import shutil
from SeisCL import SeisCL

names = ['fp32', 'fp16io', 'fp16com']
filedata = os.getcwd() + '/marmfp32'
seis = SeisCL()
seis.file = os.getcwd() + '/marmfp32'
seis.read_csts(workdir="")
seis.file = 'SeisCL'
seis.file_datalist = filedata + '_din.mat'
seis.file_din = filedata + '_din.mat'

file = h5.File(filedata + '_model.mat', "r")
models = {'vp': gaussian_filter(np.transpose(file['vp']), sigma=3),
          'vs': np.transpose(file['vs']),
          'rho': np.transpose(file['rho'])}
file.close()

"""
    _________________Set inversion parameters for SeisCL____________________
"""
seis.csts['gradout'] = 1  # SeisCl has to output the gradient
seis.csts['scalerms'] = 0  # We don't scale each trace by the rms of the data
seis.csts['scalermsnorm'] = 0  # We don't scale each trave by the rms its rms
seis.csts['scaleshot'] = 0  # We don't scale each shots
seis.csts['back_prop_type'] = 1
seis.csts['restype'] = 1  # Migration cost function
seis.csts['tmin'] = 0*(np.float(seis.csts['NT'])-2) * seis.csts['dt']

for ii, FP16 in enumerate([1, 2, 3]):
    """
        _______________________Constants for inversion__________________________
    """
    filework = os.getcwd() + '/marmgrad_' + names[ii]
    seis.csts['FP16'] = FP16

    """
        _________________________Perform Migration______________________________
    """
    if not os.path.isfile(filework + '_gout.mat'):
        seis.set_forward(seis.src_pos_all[3, :], models, withgrad=True)
        seis.execute()
        shutil.copy2(seis.workdir + "/" + seis.file_gout, filework + '_gout.mat')
        sys.stdout.write('Gradient calculation completed \n')
        sys.stdout.flush()

