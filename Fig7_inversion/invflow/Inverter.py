#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:16:36 2017

@author: gabrielfabien-ouellet
"""
import math
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
import os
from invflow.Forward import FclassError


class InvertError(Exception):
    pass

def norm_grad( grad, bnds, scale, filtscale):

    nd=bnds.shape[0]
    for (n,bnd) in enumerate(bnds):
        if bnd[0]:
            inds=[Ellipsis,]*nd
            inds[n]=slice(0,bnd[0])
            grad[tuple(inds)]=0
        
        if bnd[1]:
            inds=[Ellipsis,]*nd
            inds[n]=slice(bnd[1],grad.shape[n])
            grad[tuple(inds)]=0

    grad*=scale

    inds=[0,]*nd
    for (n,bnd) in enumerate(bnds):
        indmax=grad.shape[n];
        if bnd[1]:
            indmax=bnd[1]
        inds[n]=slice(bnd[0], indmax)
    if filtscale:
        grad[inds]=gaussian_filter(grad[inds],int(filtscale))

    return grad


class EnableCL():
    
    def __init__(self, tf_gpus=[], target_gpus=[0], session=tf.Session):
        self.cuda_d =None
        self.session = session
        try:  
            self.cuda_d = os.environ['CUDA_VISIBLE_DEVICES']
            gpus = [ii for ii in self.cuda_d.split(',')]
            gpus = [gpu for gpu in gpus if int(gpu) >= 0]
            self.egpus = [gpu for gpu in gpus if int(gpu) in tf_gpus]
            self.dgpus = [gpu for gpu in gpus if int(gpu) in target_gpus]
        except KeyError: 
            pass
    def __enter__(self):
        if self.cuda_d:
            gpus = self.egpus + ["-1"] + self.dgpus
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpus)
        self.started = self.session()
        if self.cuda_d:
            gpus = self.dgpus + ["-1"]
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpus)
        return self.started

    def __exit__(self, exc_type, exc_value, traceback):
        self.started.close()
        if self.cuda_d:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_d

class Inverter():
    
    def __init__(self, opt, costfun, totrain, gradph=None, linesearch=True, lbfgs=0,
                 bnds=None,filtscales=None, scale0=0.001, global_step=None,
                 Hessian=None, damping=0.0001, maxmin=None, wolfitermax=4,
                 wolfc1=0, wolfc2=0.95, lbfgs_pre=4, hstep_div = 6):
        
        self.l=lbfgs
        self.totrain=totrain
        self.damping=damping
        self.Hessian=Hessian
        self.maxmin=maxmin
        self.failed=False
        self.lbfgs_prestep = int(lbfgs / lbfgs_pre)
        self.linesearch = linesearch

        self.wolfitermax = wolfitermax
        self.wolfc1= wolfc1
        self.wolfc2 = wolfc2
        self.validlbfgs = [True] * self.l
        self.hstep_div = hstep_div

        if not isinstance(totrain, list):
            totrain=[totrain]
        if bnds is None:
            bnds = [ [[0,0]]*len(m) for m in totrain]
        if filtscales is None:
            filtscales = [0]*len(totrain)
            
        with tf.name_scope('Inverter'):
            
            self.inner_step= tf.Variable(0, 
                                         name='inner_step', 
                                         trainable=False, 
                                         dtype=tf.int64)
            
            
            if global_step is None:
                self.global_step = tf.train.get_or_create_global_step()
            else:
                self.global_step = global_step
                
            self.app_grad_num = tf.Variable(0, 
                                               name='app_grad_num', 
                                               trainable=False, 
                                               dtype=tf.int64)
                
            self.step=tf.Variable(totrain[0].dtype.as_numpy_dtype(1.0), 
                                  name='step', 
                                  trainable=False)
            
            with tf.name_scope('CostFunc'):
                self.costfun=tf.Variable(costfun.dtype.as_numpy_dtype(0),
                                         name='costfun', 
                                         trainable=False)
                self.calc_cost=tf.assign(self.costfun, costfun)


            with tf.name_scope('Gradient'):
                if gradph is None:
                    gr = opt.compute_gradients(costfun,
                                               var_list=totrain,
                                               colocate_gradients_with_ops=True)
                else:
                    gr = opt.compute_gradients(gradph,
                                               var_list=totrain,
                                               colocate_gradients_with_ops=True)
                self.grads=[]
                self.calc_grad=[]
                for (g,v) in gr:
                    with tf.name_scope('grad'+v.name.split(':')[0]): 
                        var = tf.Variable(v.initialized_value(),
                                          trainable=False)
                    self.grads.append( (var,v) )
                    self.calc_grad.append(tf.assign(var,g))

 
            with tf.name_scope('UpdateDir'):
                with tf.variable_scope('updir'):
                    self.updirs=[]
                    self.set_updir=[]
                    for (g,v) in self.grads:
                        with tf.name_scope('updir'+v.name.split(':')[0]): 
                            with tf.name_scope('initialize'):
                                 initvar=tf.zeros_like(g.initialized_value())
                            var = tf.Variable(initvar,
                                              trainable=False)
                            self.updirs.append( (var,v) )
                            self.set_updir.append(tf.assign( var ,g))
                        
                if lbfgs and linesearch:
                    self.build_lbfgs()
                    
                with tf.name_scope('Regularize'):
                    self.scales = []
                    self.calc_scales =[]
                    self.scale_updir = []
                    for (ii,(g,v)) in enumerate(self.updirs):
                        
                        if self.Hessian is not None:
                            H = self.Hessian[ii]
                            toscale = g / (tf.abs(H)
                                        + self.damping*tf.reduce_max(tf.abs(H)))
                        else:
                            toscale = g

                        with tf.name_scope('Scale'+v.name.split(':')[0]):
                            var = tf.Variable(v.dtype.as_numpy_dtype(0), 
                                              name='scale', 
                                              trainable=False)
                            self.scales.append(var)
                            with tf.name_scope('calc_scale'):
                                calc=(tf.reduce_max(tf.abs(v))*scale0
                                      /tf.reduce_max(tf.abs(toscale)))
                            self.calc_scales.append(tf.assign(var,calc))
                        with tf.name_scope('Condition'+v.name.split(':')[0]):
                            
                            inputs=[toscale, 
                                    bnds[ii], 
                                    self.scales[ii], 
                                    filtscales[ii]]
                            condgrad = tf.py_func(norm_grad, 
                                                  inputs,
                                                  [g.dtype], 
                                                  name='condition_grad')[0]
                            self.scale_updir.append(tf.assign(self.updirs[ii][0],
                                                              condgrad))
            if self.maxmin is not None:
                with tf.name_scope('Clip'):
                    self.clip = []
                    for (ii,m) in enumerate(self.totrain):
                        clipped =tf.clip_by_value(m,
                                                  self.maxmin[ii][0],
                                                  self.maxmin[ii][1])
                        self.clip.append(tf.assign(m,clipped))

            if linesearch:
                with tf.name_scope('WolfSearch'):

                    apnewstep=[]
                    apstep=[]
                    apstepcl=[]
                    with tf.name_scope('Apply_new_step'):
                        self.stepp=tf.Variable(totrain[0].dtype.as_numpy_dtype(1),
                                               name='stepp',
                                               trainable=False)

                        self.stepin= tf.placeholder(name='stepinput',
                                                dtype=totrain[0].dtype)
                        prevstep=tf.assign(self.stepp, self.step)

                        with tf.control_dependencies([prevstep]):
                                newstep =tf.assign(self.step, self.stepin)
                        with tf.control_dependencies([newstep]):
                            for (ii, (g,v) ) in enumerate(self.updirs):
                                apnewstep.append(( tf.check_numerics((self.step-self.stepp)*g, 'grad'), v))
                            self.apgr_new = opt.apply_gradients(apnewstep,
                                                                name='apply',
                                                                global_step=self.app_grad_num)
                    with tf.name_scope('Apply_step'):
                        for (ii, (g,v) ) in enumerate(self.updirs):
                            apstep.append( (tf.check_numerics(self.step*g, 'grad'), v) )
                        self.apgr = opt.apply_gradients(apstep,
                                                        name='apply',
                                                        global_step=self.app_grad_num)
                    with tf.name_scope('Remove_step'):
                        for (ii, (g,v) ) in enumerate(self.updirs):
                            apstepcl.append((tf.check_numerics(-self.step*g, 'grad'), v))
                        self.apgr_cl =  opt.apply_gradients(apstepcl,
                                                            name='apply',
                                                            global_step=self.app_grad_num)

                    with tf.name_scope('Prod_grad_up'):
                        prod=[]
                        for ii in range(0,len(self.grads)):
                            prod.append(-tf.reduce_sum(self.grads[ii][0]*
                                                       self.updirs[ii][0]))
                        self.prod_gr_up = tf.add_n(prod)
            else:
                checks = [tf.check_numerics(g, message='Gradient contains NaN') for g in self.updirs]
                with tf.control_dependencies(checks):
                    self.apply_gradient = opt.apply_gradients(self.updirs,
                                                              global_step=self.global_step)

                  
    # def backtracking_line_search_wolf(self, sess, feed_dict):
    #
    #     c1=10^-5
    #     c2=0.9
    #     itermax=2
    #
    #     rms0 = self.costfun.eval(session=sess)
    #     rms=float("inf")
    #     prod0= self.prod_gr_up.eval(session=sess)
    #     prod=prod0
    #     n=0
    #     tau=0.7
    #
    #     while (n<itermax
    #            and (rms>rms0+self.step.eval(session=sess)*c1*prod0 or -prod>-c2*prod0)):
    #
    #         if n==0:
    #             sess.run(self.apgr, feed_dict=feed_dict)
    #         elif rms>rms0+self.step.eval(session=sess)*c1*prod0:
    #             sess.run(self.apgr_new,
    #                      feed_dict={self.stepin:self.step.eval(session=sess)*tau})
    #         elif -prod>-c2*prod0:
    #             sess.run(self.apgr_new,
    #                      feed_dict={self.stepin:self.step.eval(session=sess)/tau})
    #
    #         sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)
    #         rms = self.costfun.eval(session=sess)
    #         prod= self.prod_gr_up.eval(session=sess)
    #         n+=1
    #
    #
    #     if rms>rms0+self.step.eval(session=sess)*c1*prod0 or -prod>-c2*prod0:
    #         self.failed=True
    #         if rms>rms0+self.step.eval(session=sess)*c1*prod:
    #             sess.run(self.apgr_cl, feed_dict=feed_dict)
    #     else:
    #         self.failed=False

    
    def bisection_wolf_line_search(self, sess, feed_dict):
        
        c1=self.wolfc1
        c2=self.wolfc2
        itermax=self.wolfitermax
        if self.inner_step.eval(session=sess)==0:
            itermax *= 2
        alpha=0
        beta=float("inf")

        rms0 = self.costfun.eval(session=sess)   
        rms = float("inf")
        prod0 = self.prod_gr_up.eval(session=sess)
        prod = self.prod_gr_up.eval(session=sess)
        n=0
        step =self.step.eval(session=sess)

        self.failed = True
        if step>float('Inf') or step<=0:
            self.step.load(1.0, sess)
            print('Invalid step size, resetting to 1')
        if not math.isfinite(rms0) or not math.isfinite(prod0):
            raise InvertError('Cannot perform line search\n')

        if prod0 > 0:
            raise InvertError('Search direction is not a descent direction\n')

        while (n<itermax and (rms>rms0+self.step.eval(session=sess)*c1*prod0
                              or -prod > -c2*prod0)):
            
            if n==0:
                sess.run(self.apgr, feed_dict=feed_dict)
            elif rms > rms0+self.step.eval(session=sess)*c1*prod0:
                beta=self.step.eval(session=sess)
                if alpha==0:
                    sess.run(self.apgr_new, feed_dict={self.stepin:0.7*beta})
                else:
                    sess.run(self.apgr_new,
                             feed_dict={self.stepin:0.5*(alpha+beta)})
            elif -prod>-c2*prod0:
                alpha=self.step.eval(session=sess)
                if beta == float("inf"):
                    sess.run(self.apgr_new, feed_dict={self.stepin:alpha/0.7}) 
                else:
                    sess.run(self.apgr_new,
                             feed_dict={self.stepin:0.5*(alpha+beta)})
            
            try:
                sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict) 
                rms = self.costfun.eval(session=sess)  
                prod = self.prod_gr_up.eval(session=sess)
            except (FclassError, tf.errors.InternalError, tf.errors.AbortedError):
                rms = float("inf")
                prod = 0.0
                print('Failed evaluation during line search')

            print('    linesearch %d, rms0=%f, rms=%f, step=%f prod=%f prod0=%f, rmscond=%f'%(n,
                                                                 rms0,
                                                                 rms,
                                                                 self.step.eval(session=sess),
                                                                 prod,
                                                                 prod0,
                                                                  rms0 + self.step.eval(session=sess) * c1 * prod0
                                                                 ))
            n += 1

        if rms > rms0+self.step.eval(session=sess)*c1*prod0:
            print('Line search failed, insufficient cost decrease')
            sess.run(self.apgr_cl, feed_dict=feed_dict)
            print('Canceling applied step')
        elif -prod > -c2*prod0:
            print('Line search failed, insufficient curvature decrease')
        else:
            self.failed = False



    def build_lbfgs(self):
        
        with tf.name_scope('LBFGS'):
            self.set_y1 = [None]*self.l
            self.set_y2 = [None]*self.l
            self.set_s = [None]*self.l
            self.set_rho = [None]*self.l
            self.set_alpha = [None]*self.l
            self.set_beta = [None]*self.l
            self.loopa = [None]*self.l
            self.loopb = [None]*self.l
            self.y = [None]*self.l
            self.s = [None]*self.l
            self.reset_y = [None] * self.l
            self.reset_s = [None] * self.l
            self.rho = [None]*self.l
            self.alpha = [None]*self.l
            self.beta = [None]*self.l
            
            with tf.variable_scope('y'):
                 for ii in range(0,self.l):
                     self.y[ii]=[]
                     self.set_y1[ii]=[]
                     self.set_y2[ii]=[]
                     self.reset_y[ii] = []
                     with tf.variable_scope(str(ii)):
                         for (g,v) in self.grads:
                             with tf.name_scope('initialize'):
                                 initvar=tf.zeros_like(g.initialized_value())
                             var=tf.Variable(initvar, 
                                             name='y'+v.name.split(':')[0], 
                                             trainable=False )
                             self.y[ii].append(var)
                             self.set_y1[ii].append(tf.assign(var,-g))
                             self.set_y2[ii].append(tf.assign_add(var,g))
                             self.reset_y[ii].append(tf.assign_add(var, var*0))
            with tf.variable_scope('s'):
                 for ii in range(0,self.l):
                     self.s[ii]=[]
                     self.set_s[ii]=[]
                     self.reset_s[ii] = []
                     with tf.variable_scope(str(ii)):
                         for (g,v) in self.updirs:
                             with tf.name_scope('initialize'):
                                 initvar=tf.zeros_like(g.initialized_value())
                             var=tf.Variable(initvar, 
                                             name='s'+v.name.split(':')[0], 
                                             trainable=False )
                             self.s[ii].append(var)
                             self.set_s[ii].append(tf.assign(var,-self.step*g))
                             self.reset_s[ii].append(tf.assign(var, 0*var))
                         
            with tf.variable_scope('rho'):
                for ii in range(0,self.l):
                    with tf.variable_scope('rho_'+str(ii)):
                        self.rho[ii]=tf.Variable(self.grads[0][0].dtype.as_numpy_dtype(0), 
                                            name='rho', 
                                            trainable=False )
                        prod=[]
                        for (jj,(g,v)) in enumerate(self.updirs):
                             prod.append( tf.reduce_sum(self.s[ii][jj]*self.y[ii][jj]) )
                        self.set_rho[ii]=tf.assign(self.rho[ii],1.0/tf.add_n(prod))  

            with tf.variable_scope('alpha'):
                for ii in range(0,self.l):
                    with tf.variable_scope('alpha_'+str(ii)):
                        self.alpha[ii]=tf.Variable(self.grads[0][0].dtype.as_numpy_dtype(0), 
                                            name='alpha', 
                                            trainable=False )
                        prod=[]
                        for (jj,(g,v)) in enumerate(self.updirs):
                             prod.append( tf.reduce_sum(self.s[ii][jj]*
                                                         self.updirs[jj][0]) )
                        self.set_alpha[ii]=tf.assign(self.alpha[ii],
                                                     self.rho[ii]*tf.add_n(prod)) 
            with tf.variable_scope('beta'):
                for ii in range(0,self.l):
                    with tf.variable_scope('beta_'+str(ii)):
                        self.beta[ii]=tf.Variable(self.grads[0][0].dtype.as_numpy_dtype(0), 
                                            name='beta', 
                                            trainable=False )
                        prod=[]
                        for (jj,(g,v)) in enumerate(self.updirs):
                             prod.append( tf.reduce_sum(self.y[ii][jj]*
                                                         self.updirs[jj][0]) )
                        self.set_beta[ii]=tf.assign(self.beta[ii],
                                                    self.rho[ii]*tf.add_n(prod))                        
            
            with tf.name_scope('loopa'):    
                for ii in range(0,self.l):
                    self.loopa[ii]=[]
                    with tf.variable_scope(str(ii)):
                        for jj in range(0,len(self.grads)):
                            var= tf.assign_add(self.updirs[jj][0],
                                               -self.alpha[ii]*self.y[ii][jj])
                            self.loopa[ii].append(var) 
                        
            with tf.name_scope('loopb'):    
                for ii in range(0,self.l):
                    self.loopb[ii]=[]
                    with tf.variable_scope(str(ii)):
                        for jj in range(0,len(self.grads)):
                            var= tf.assign_add(self.updirs[jj][0],
                                               (self.alpha[ii]-self.beta[ii])
                                               *self.s[ii][jj])
                            self.loopb[ii].append(var) 
                            
    def two_loops_lbfgs(self,sess,feed_dict=None):
        
        itern=self.inner_step.eval(session=sess)
        if itern == 0:
            for ind in range(self.l):
                sess.run(self.reset_s[ind], feed_dict=feed_dict)
                sess.run(self.reset_y[ind], feed_dict=feed_dict)

        sess.run(self.set_updir,feed_dict=feed_dict)
        for ii in range(itern-1,max([itern-self.l-1,-1]),-1):
            ind=ii%self.l
            if self.validlbfgs[ind]:
                sess.run(self.set_rho[ind],feed_dict=feed_dict)
                sess.run(self.set_alpha[ind],feed_dict=feed_dict)
                sess.run(self.loopa[ind],feed_dict=feed_dict)
        sess.run(self.scale_updir,feed_dict=feed_dict)
        for ii in range(max([itern-self.l,0]),itern):
            ind=ii%self.l
            if self.validlbfgs[ind]:
                sess.run(self.set_beta[ind],feed_dict=feed_dict)
                sess.run(self.loopb[ind],feed_dict=feed_dict)
            
    def lbfgs(self, sess, feed_dict=None):

        if ((self.inner_step.eval(session=sess) > self.lbfgs_prestep+1) 
                and not self.failed and self.hstep_div > 0):
            self.step.load(self.step.eval(session=sess)/self.hstep_div, sess)
            self.two_loops_lbfgs(sess, feed_dict)
            sess.run(self.apgr, feed_dict=feed_dict)
            rms0 = self.costfun.eval(session=sess)
            try:
                sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)
                rms = self.costfun.eval(session=sess)
                print('Applying half step rms0: %f, rms: %f' % (rms0, rms))
                if rms > rms0:
                    print('Half step too large, canceling')
                    sess.run(self.apgr_cl, feed_dict=feed_dict)
                    sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)
                elif self.maxmin is not None:
                    sess.run(self.clip, feed_dict=feed_dict)
            except (FclassError, tf.errors.InternalError, tf.errors.AbortedError):
                print('cancelling step without line search')
                sess.run(self.apgr_cl, feed_dict=feed_dict)
                sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)

            self.step.load(self.step.eval(session=sess) * self.hstep_div, sess)
            self.step.load(math.exp((math.log(self.step.eval(session=sess)))/1.1),
                           sess)
        else:
            if not self.failed:
                self.step.load(math.exp((math.log(self.step.eval(session=sess))) / 1.1),
                    sess)
            sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)

        ind = self.inner_step.eval(session=sess)%self.l
        if self.inner_step.eval(session=sess) == 0:
            sess.run(self.set_updir)
            sess.run(self.calc_scales, feed_dict=feed_dict)
        self.two_loops_lbfgs(sess, feed_dict)
        sess.run([self.set_y1[ind]], feed_dict=feed_dict)
        self.bisection_wolf_line_search(sess, feed_dict)
        if not self.failed:
            sess.run([self.set_s[ind]], feed_dict=feed_dict)
            sess.run([self.set_y2[ind]], feed_dict=feed_dict)
            if self.maxmin is not None:
                sess.run(self.clip, feed_dict=feed_dict)
                self.inner_step.load(self.inner_step.eval(session=sess) + 1,
                                     sess)
            self.validlbfgs[ind] = True
        else:
            self.validlbfgs[ind] = False
        self.global_step.load(self.global_step.eval(session=sess) + 1, sess)
            
    def gradient_descent(self, sess, feed_dict=None):

        if not self.failed and self.hstep_div>0:
            self.step.load(self.step.eval(session=sess)/self.hstep_div, sess)
            sess.run(self.set_updir, feed_dict=feed_dict)
            sess.run(self.scale_updir, feed_dict=feed_dict)
            sess.run(self.apgr, feed_dict=feed_dict)
            rms0 = self.costfun.eval(session=sess)
            try:
                sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)
                if self.costfun.eval(session=sess) > rms0*1.1:
                    sess.run(self.apgr_cl, feed_dict=feed_dict)
                elif self.maxmin is not None:
                    sess.run(self.clip, feed_dict=feed_dict)
            except (FclassError, tf.errors.InternalError, tf.errors.AbortedError):
                print('cancelling step without line search')
                sess.run(self.apgr_cl, feed_dict=feed_dict)
                sess.run([self.calc_grad, self.calc_cost] ,feed_dict=feed_dict)
            self.step.load(self.step.eval(session=sess) * self.hstep_div, sess)
        else:
            sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)


        if self.inner_step.eval(session=sess):
            sess.run(self.set_updir)
            sess.run(self.calc_scales, feed_dict=feed_dict)
        else:
            sess.run(self.set_updir, feed_dict=feed_dict)
        sess.run(self.scale_updir, feed_dict=feed_dict)
        self.bisection_wolf_line_search(sess, feed_dict)
        if not self.failed:
            if self.maxmin is not None:
                sess.run(self.clip, feed_dict=feed_dict)
        self.global_step.load(self.global_step.eval(session=sess) + 1, sess)
        self.inner_step.load(self.inner_step.eval(session=sess) + 1, sess)

    def tfopt(self, sess, feed_dict=None):
        if self.maxmin is not None:
            sess.run(self.clip, feed_dict=feed_dict)
        sess.run([self.calc_grad, self.calc_cost], feed_dict=feed_dict)
        if self.inner_step.eval(session=sess) == 0:
            sess.run(self.set_updir)
            sess.run(self.calc_scales, feed_dict=feed_dict)
        else:
            sess.run(self.set_updir, feed_dict=feed_dict)
        sess.run(self.scale_updir, feed_dict=feed_dict)
        sess.run(self.apply_gradient, feed_dict=feed_dict)
        self.inner_step.load(self.inner_step.eval(session=sess) + 1, sess)
        
    def run(self, sess, feed_dict=None):

        if self.linesearch:
            if self.l:
                self.lbfgs(sess, feed_dict)
            else:
                self.gradient_descent(sess, feed_dict)
        else:
            self.tfopt(sess, feed_dict)
    
    
    
    
