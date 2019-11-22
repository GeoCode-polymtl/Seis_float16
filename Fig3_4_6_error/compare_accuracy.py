#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of FP16 and FP32 for the Marmousi-II model
"""
import numpy as np
import os
import sys
import shutil
from SeisCL import SeisCL

def compare_accuracy(name, models, f0, srcx, tmax, dh, dt=None,
                        FDORDER=4, MAXRELERROR=0, sz=4, gz=1):

    filepath = os.getcwd() + '/'
    file = filepath + name

    """
    _____________________Simulation constants input file________________________
    """
    seis = SeisCL()

    seis.csts['N'] = np.array(models['rho'].shape)
    seis.csts['ND'] = len(models['rho'].shape)  # Flag for dimension.
    seis.csts['dh'] = dh  # Grid spatial spacing
    
    if dt is None:
        seis.csts['dt'] = 6 * dh / (7 * np.sqrt(2) * np.max(models['vp'])) * 0.85
    else:
        seis.csts['dt'] = dt
    seis.csts['NT'] = int(tmax / seis.csts['dt'])  # Number of time steps
    seis.csts['freesurf'] = 0  # Include a free surface at z=0: 0: no, 1: yes
    seis.csts['FDORDER'] = FDORDER  # Order of the finite difference stencil.
    seis.csts['MAXRELERROR'] = MAXRELERROR  # Set to 1
    seis.csts['f0'] = f0  # Central frequency

    seis.csts['abs_type'] = 2  # Absorbing boundary type: 2: Absorbing layer
    seis.csts['nab'] = 32  # Width in grid points of the absorbing layer
    seis.csts['abpc'] = 6  # Exponential decay of the absorbing layer

    """
    _________________________Sources and receivers______________________________
    """

    gx = np.arange(seis.csts['nab'] + 5,
                   seis.csts['N'][-1] - seis.csts['nab'] - 5)
    gx = gx * dh
    gy = gx * 0 + seis.csts['N'][1] // 2 * dh
    gz = gx * 0 + (seis.csts['nab'] + gz) * dh

    for ii in range(int(srcx / dh), int(srcx / dh) + 10, 10000):
        idsrc = seis.src_pos_all.shape[1]
        toappend = np.zeros((5, 1))
        toappend[0, :] = (ii) * dh
        toappend[1, :] = seis.csts['N'][1] // 2 * dh
        toappend[2, :] = (seis.csts['nab'] + sz) * dh
        toappend[3, :] = idsrc
        toappend[4, :] = 100
        seis.src_pos_all = np.append(seis.src_pos_all, toappend, axis=1)
        
        toappend = np.stack([gx,
                             gy,
                             gz,
                             gz * 0 + idsrc,
                             np.arange(0, len(gx)) + seis.rec_pos_all.shape[1] + 1,
                             gx * 0 + 2,
                             gx * 0,
                             gx * 0], 0)
        seis.rec_pos_all = np.append(seis.rec_pos_all, toappend, axis=1)


    """
        ___________________________FP32__________________________________
    """
    filename = file + "fp32_dout.mat"
    if not os.path.isfile(filename):
        seis.csts['FP16'] = 1
        seis.set_forward(seis.src_pos_all[3, :], models, withgrad=False)
        shutil.copy2(seis.workdir + "/" + seis.file_csts, file + "_csts.mat")
        shutil.copy2(seis.workdir + "/" + seis.file_model, file + "_model.mat")
        seis.execute()
        sys.stdout.write('Forward calculation completed \n')
        sys.stdout.flush()
        shutil.copy2(seis.workdir + "/" + seis.file_dout, filename)

    """
        __________________FP32 + truncate parameter to FP16_____________________
    """
    seis.csts['FP16'] = 1
    seis.csts['halfpar'] = 1
    filename = file + "fp32hp_dout.mat"
    if not os.path.isfile(filename):
        seis.set_forward(seis.src_pos_all[3, :], models, withgrad=False)
        seis.execute()
        sys.stdout.write('Forward calculation completed \n')
        sys.stdout.flush()
        shutil.copy2(seis.workdir + "/" + seis.file_dout, filename)

    """
        ___________________________FP16 IO__________________________________
    """
    seis.csts['FP16'] = 2
    seis.csts['halfpar'] = 0
    filename = file + "fp16_2_dout.mat"
    if not os.path.isfile(filename):
        seis.set_forward(seis.src_pos_all[3, :], models, withgrad=False)
        seis.execute()
        sys.stdout.write('Forward calculation completed \n')
        sys.stdout.flush()
        shutil.copy2(seis.workdir + "/" + seis.file_dout, filename)

    """
        ___________________________FP16 COMP__________________________________
    """
    seis.csts['FP16'] = 3
    seis.csts['halfpar'] = 0
    filename = file + "fp16_3_dout.mat"
    if not os.path.isfile(filename):
        seis.set_forward(seis.src_pos_all[3, :], models, withgrad=False)
        seis.execute()
        sys.stdout.write('Forward calculation completed \n')
        sys.stdout.flush()
        shutil.copy2(seis.workdir + "/" + seis.file_dout, filename)

    """
        ___________________________FP32 holberg_________________________________
    """
    seis.csts['FP16'] = 1
    seis.csts['halfpar'] = 0
    if MAXRELERROR == 1:
        seis.csts['FDORDER'] = 8
    else:
        seis.csts['MAXRELERROR'] = 1
    filename = file + "fp32holberg_dout.mat"
    if not os.path.isfile(filename):
        seis.set_forward(seis.src_pos_all[3, :], models, withgrad=False)
        seis.execute()
        sys.stdout.write('Forward calculation completed \n')
        sys.stdout.flush()
        shutil.copy2(seis.workdir + "/" + seis.file_dout, filename)
