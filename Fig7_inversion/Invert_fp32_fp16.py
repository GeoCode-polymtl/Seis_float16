#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inversion on the marmousi model
"""

from invflow.SeisCL import SeisCL
from invflow.Inverter import Inverter, EnableCL, InvertError
import tensorflow as tf
import os
import numpy as np
from shutil import copyfile
import hdf5storage as h5mat
import h5py as h5
from scipy.ndimage.filters import gaussian_filter
import sys


"""
    _________________________________Std in_____________________________________
"""
if len(sys.argv) > 1:
    FP16 = int(sys.argv[1])
else:
    FP16 = 1

"""
    _______________________Constants for inversion_____________________________
"""

batch = 6
iter_per_freq = 40
passes = 1
freqs = [1.2, 1.5, 2, 3, 3.5, 4, 4.5, 5, 6, 7, 9, 11, 13, 15, 17]
if FP16 == 1:
    namefp = 'fp32'
elif FP16 == 2:
    namefp = 'fp16io'
elif FP16 == 3:
    namefp = 'fp16com'
else:
    raise BaseException('input must be 1,2 or 3')
filepath = os.getcwd() + '/marminv_' + namefp


"""
_____________________________Read input files__________________________________
"""
filedata = os.getcwd() + '/marmfp321'
copyfile(filedata + '_csts.mat', filepath + '_csts.mat')
copyfile(filedata + '_model.mat', filepath + '_model.mat')
copyfile(filedata + '_din.mat', filepath + '_din.mat')
csts = h5mat.loadmat(filepath + '_csts.mat')
models = h5mat.loadmat(filepath + '_model.mat')
nab = csts['nab']

"""
    _________________Create an initial smoothed model__________________________
"""
for ii in range(nab+6, models['vp'].shape[0] - nab):
    models['vp'][ii, nab:-nab] = 1500 + 15 * (ii - nab - 1)

"""
    _________________Set inversion parameters for SeisCL_______________________
"""
F = SeisCL()
for key in csts:
    F.csts[key] = csts[key]
F.input_residuals = False  # Invflow here takes a node that output residuals
F.csts['back_prop_type'] = 1  # Inversion in the frequency domain
F.read_src = True  # We read the source from the data file
F.csts['rmsout'] = 1  # SeisCl has to output the cost function value
F.csts['resout'] = 0  # SeisCl has to output the cost function value
F.csts['seisout'] = 0
F.csts['HOUT'] = 1
F.csts['gradout'] = 1  # SeisCl has to output the gradient
F.csts['scalerms'] = 0  # We don't scale each trave by the rms of the data
F.csts['scalermsnorm'] = 0  # We don't scale each trave by the rms its rms
F.csts['scaleshot'] = 0  # We don't scale each shots
F.csts['gradfreqs'] = np.array([F.csts['f0']])  # The frequencies we use
F.csts['FP16'] = FP16
F.csts['nmax_dev'] = 1
F.file_datalist = filepath + '_csts.mat'  # The file listing all the data
F.file_din = filepath + '_din.mat'  # The file containing the data

"""
    _________________________Set the Inverter object___________________________
"""
with tf.name_scope('model'):
    m = [tf.get_variable(name='vp', initializer=models['vp']),
         tf.get_variable(name='vs', initializer=models['vs']),
         tf.get_variable(name='rho', initializer=models['rho'])
         ]
mtotrain = [m[0]]  # We invert for vp, vs and rho

with tf.name_scope('data'):
    dataid = tf.placeholder(dtype=tf.int16, name='data_ids')

with tf.name_scope('Forward'):
    thisop = F.op(dataid, m, name='Forward')
    output = tf.sqrt(thisop.op[0])
    Hessian = thisop.opH
costfun = tf.losses.mean_squared_error(0, output, weights=1.0)
opt = tf.train.GradientDescentOptimizer(10.0)

with tf.name_scope('regularize'):
    nab = csts['nab']
    bnd = tf.constant(np.array([[nab + 6, -nab], [nab, -nab]]), name='boundary')
    bnds = [bnd for thism in mtotrain]  # We dont invert in the water layer
    filtscale = tf.Variable(0, name='filtscale', trainable=False)
    filtscales = [filtscale for thism in
                  mtotrain]  # We will filter the gradient

inv = Inverter(opt,
               costfun,
               mtotrain,
               bnds=bnds,
               filtscales=filtscales,
               hstep_div=2,
               lbfgs=8,
               scale0=0.02,
               Hessian=Hessian,
               maxmin=[[1400, 5800]])

"""
    ______________Set variables to monitor during inversion____________________
"""
tf.summary.scalar("cost", inv.costfun)               #Monitor the cost function
summary_op = tf.summary.merge_all()       #These can be viewed with tensorboard

"""
    _________________________Perform the inversion_____________________________
"""
tf.logging.set_verbosity(tf.logging.ERROR)
if not os.path.isdir(filepath + "logs"):
    os.mkdir(filepath + "logs")
for ii, freq in enumerate(freqs):
    thislog = filepath + "logs/freq" + str(freq)
    """
        _____________________Hooks to control the inversion_________________
    """
    niter = iter_per_freq
    hooks = [tf.train.StopAtStepHook(last_step=niter),
             # Max number of iterations
             tf.train.SummarySaverHook(save_steps=1,  # Monitor at each step
                                       summary_op=summary_op),
             tf.train.CheckpointSaverHook(checkpoint_dir=thislog,
                                          save_steps=iter_per_freq,
                                          saver=tf.train.Saver(
                                              max_to_keep=None))
             ]

    """
        _________________________Create HDF5 file to save___________________
    """
    if not os.path.isdir(thislog):
        os.mkdir(thislog)

    file = h5.File(thislog + '/results.mat', 'a')
    if 'rms' not in file:
        file.create_dataset('rms', (niter,), dtype='float')
    if 'vp' not in file:
        file.create_dataset('vp',
                            (
                            niter, models['vp'].shape[0], models['vp'].shape[1])
                            , dtype='float')
    if 'vp0' not in file:
        file.create_dataset('vp0',
                            (models['vp'].shape[0], models['vp'].shape[1])
                            , dtype='float')
        file['vp0'][:, :] = models['vp']
    if 'rho0' not in file:
        file.create_dataset('rho0',
                            (models['rho'].shape[0], models['rho'].shape[1])
                            , dtype='float')
        file['rho0'][:, :] = models['rho']

    file.flush()

    """
        _________________________Perform the inversion______________________
    """


    def sesscreate():
        return tf.train.MonitoredTrainingSession(checkpoint_dir=thislog,
                                                 save_checkpoint_secs=None,
                                                 hooks=hooks,
                                                 save_summaries_steps=1)


    with EnableCL(target_gpus=[0],
                  session=sesscreate) as sess:

        try:
            step = inv.global_step.eval(session=sess)
        except RuntimeError:
            pass

        if step < iter_per_freq:
            if freq != freqs[0]:
                inv.inner_step.load(0, sess)
                sigma = 3000 / freq / 8 / F.csts['dh']
                minv[nab+6:-nab, nab:-nab] = gaussian_filter(minv[nab+6:-nab, nab:-nab],
                                                          sigma)
                m[0].load(minv, sess)

        while not sess.should_stop():

            step = inv.global_step.eval(session=sess)
            print('iter:%d, iter_freq:%f freq:%f err:%f' % (
                step,
                inv.inner_step.eval(session=sess),
                freq,
                inv.costfun.eval(session=sess)))
            file['vp'][step, :, :] = m[0].eval(session=sess)
            if inv.failed:
                file['rms'][step] = np.NAN
            else:
                file['rms'][step] = inv.costfun.eval(session=sess)
            file.flush()

            # Frequency used for gradient
            F.csts['gradfreqs'] = np.array([freq])
            F.csts['fmin'] = 0.8 * freq
            F.csts['fmax'] = 1.2 * freq
            # Length of grad smooth
            filtscale.load(3000 / freq / 8 / F.csts['dh'], sess)
            np.random.seed(ii*iter_per_freq + step)
            this_batch = np.random.choice(F.src_pos_all[3, :],
                                          int(batch),
                                          replace=False)

            try:
                inv.run(sess, feed_dict={dataid: this_batch})
            except InvertError as msg:
                print(msg)
    minv = file['vp'][-1, :, :]
    file.close()
