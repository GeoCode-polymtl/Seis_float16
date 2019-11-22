
# -*- coding: utf-8 -*-
"""
Base class for the forward model operator
"""
import numpy as np
import copy    
from shutil import rmtree
import os
import subprocess
import tensorflow as tf
import time
    
class FclassError(Exception):
    pass

class Subprocess_op:
    """Class to create a Tensorflow operator from a program called through
           subprocess.
    """  
    def __init__(self,F, dataid, m, name=None):
        """Create a Subprocess_op
    
        Args:
          F: A Forward object describing how to launch the program
          dataid: A placeholder inputing the ids of the data to be used 
                  (could also contain the actual data) 
          m: A list of tensors containing the model parameters that can be optimized
             (all memebers of m do not need to be actually optimized in a run )
          name: A name to be given for this op

        """
        self.F=copy.copy(F)
        self.dataid=dataid
        self.m=m
        self.name=name
        self._op=None
        self._opH=None
        self.workdir='./scratch'+''.join([str(k) for k in np.random.randint(0,9,15) ])+'/'    

    @property
    def workdir(self):
        return self.__workdir

    @workdir.setter
    def workdir(self, workdir):
        """Create a unique working directory for this op
    
        Args:
          workdir: unique name of the directory. If empty directory is given,
          clean the previous working dir if it exist

        """
        if hasattr(self,'workdir') and os.path.exists(self.workdir):
            rmtree(self.workdir)
        self.__workdir=workdir
        if workdir and not os.path.exists(workdir):
            os.makedirs(workdir)  
    
    @property    
    def opH(self):
        if not self._opH:
            workdir=self.workdir
            read_Hessian=self.F.read_Hessian
            output=[]
            
            # Actual gradient:
            for m in self.m:
                def _Hessian(param_name):
    
                    try:
                        name = [param_name.decode('ascii')]
                        output = np.array(read_Hessian(workdir, name)[0],
                                          dtype=m.dtype.as_numpy_dtype)
    #                    output= np.array( read_Hessian(workdir, names), 
    #                                      dtype=np.float64 ) 
                    except (FclassError, OSError) as msg:
                        raise FclassError(msg)
#                        raise tf.errors.AbortedError(None,None,msg)
                    
                    return output
                
                # Build the Hessian operator
                vstring= m.name.split(':')[0]
                output.append(tf.py_func(_Hessian, 
                              [vstring],
                              [m.dtype],
                              stateful=True,
                              name=vstring)[0])
            self._opH = output
        return self._opH
        
    @property    
    def op(self):
        """Create the op on a first call, then only returns it

        """
        if not self._op:
            
            #No reference to outside objects are permitted in custom op functions _Forward and _Adjoint
            #Copy here the reference to avoid this
            workdir=self.workdir
            set_forward = self.F.set_forward
            set_backward = self.F.set_backward
            callcmd=self.F.callcmd(self.workdir)
            input_residuals=self.F.input_residuals
            read_data=self.F.read_data
            read_rms=self.F.read_rms
#            write_residuals=self.F.write_residuals
            write_residuals=None
            read_grad=self.F.read_grad
            
            
            # Actual forward:
            def _Forward( dataid, m, param_names):

                #Try to launch the forward code,
                nmax=2
                n=0
                success=False
                msg=''
                while n<nmax and not success:
                    try:
                        names = [name.decode('ascii') for name in param_names]
                        set_forward(dataid,
                                    dict(zip(names, m)),
                                    workdir,
                                    withgrad=not input_residuals)
                        pipes = subprocess.Popen(callcmd,
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE,
                                                 shell=True)
                        date, std_err = pipes.communicate()
                        if std_err:
                            print(std_err.decode())
                            raise FclassError(std_err.decode())
        
                        if input_residuals:
                            output = np.float32(read_data(workdir))
                        else:
                            output = np.float32(read_rms(workdir))
                        success = True
                    except (FclassError, OSError) as msg:
                        n += 1
                        if n == nmax:
                            raise FclassError(msg)
#                            raise tf.errors.AbortedError(None,None,msg)
                return output
                    
            # Actual gradient:
            def _Adjoint_fun( m, param_names, residuals):
                try:
                    names = [name.decode('ascii') for name in param_names]
                    if input_residuals:
                        set_backward(workdir, residuals)
                        pipes = subprocess.Popen(callcmd,
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE,
                                                 shell=True)
                        date, std_err = pipes.communicate()
                        if std_err:
                            print(std_err.decode())
                            raise FclassError(std_err.decode())
                    output = np.array(read_grad(workdir, names),
                                      dtype=np.float64)
                except (FclassError, OSError) as msg:
                    raise FclassError(msg)
                return output
            
            #The gradient function can only input and output tensor objects. To evaluate the gradient and output an numpy array, 
            #we must still create a new py_func containing the gradient function. When building graph, function in py_fun are not
            #evaluated, whereas the grad function of tf.RegisterGradient is. The latter is problematic because when initalizing the graph, we 
            #not data is transmited and subprocess calls or any operation on tensors value will fail.  
            def _Adjoint(op, grad ):                
                return op.inputs[0], tf.py_func(_Adjoint_fun, [op.inputs[1], op.inputs[2], grad], tf.float64 ), op.inputs[2]
            
            param_names=[t.name.split(':')[0] for t in self.m] 
            for param in self.F.params:
                if param not in param_names:
                    raise tf.errors.InvalidArgumentError(None, None,'This op require param %s to be defined in m'%param)
            # Build the Forward operator overriding the gradient with our own function
            with tf.name_scope(self.name, "Forward", [self.dataid,self.m, param_names]) as name:

                # Need to generate a unique name to avoid duplicates:
                rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
                
                tf.RegisterGradient(rnd_name)(_Adjoint)
                with tf.get_default_graph().gradient_override_map({"PyFunc": rnd_name}):
                    self._op=tf.py_func(_Forward, [self.dataid,self.m, param_names], [tf.float32], stateful=True, name=name)
        return self._op
    
    def __del__(self):
        self.workdir='' #this should delete workdir
        if os.path.exists(self.workdir):
            rmtree(self.workdir)
        
        
        
class Forward:
    """Base class for Forward model operators.

        
      This class defines the API to add Ops to call an outside program for data
      and gradient . You never use this class directly, but instead instantiate
      one of its subclasses such as `SeisCL`.
    
      ### Usage
    
      ```python
      # Instantiate a forward object with data and parameters.
      F = SeisCL()
      dataid=tf.placeholder(dtype=tf.int16)
      m=[tf.get_variable(name='vp', initializer=vp0),
       tf.get_variable(name='vs', initializer=vs0),
       tf.get_variable(name='rho', initializer=rho0)
       ]
      # Add two Ops to the graph  earch with their own working directory.
      # Each op can be used a standard ops within tensorflow
      op1=F.op()
      op2=F.op()
      ```
    
      You can then begin the inversion with Tensorflow training (assuming op1
      output the rms value)
    
      ```python
      # Use training algorithms of Tensorflow:
      costfun = tf.losses.mean_squared_error(0, op1, weights=1.0 )
      opt = tf.train.GradientDescentOptimizer(10.0) 
      ```
      """

    def __init__(self):      
        """Create a new Forward model.
    
        This must be overloaded by the constructors of subclasses.

        """
        self.params=[] #list of required model parameters
        self.input_residuals=False
             
    def set_forward(self, jobids, params, workdir):
        """Sets any files or parameters before executing the forward pass of
           modeling program
    
        Args:
          jobids: The handle to he batch of data (or the data itself)
          params: A dict containing parameter name: numpy_array of model parameters
    
        Returns:
          Void
        """
        raise NotImplementedError()

    def set_backward(self):
        """Sets any files or parameters before executing the backward pass of
           modeling program

        Args:
          jobids: The handle to he batch of data (or the data itself)
          params: A dict containing parameter name: numpy_array of model parameters

        Returns:
          Void
        """
        raise NotImplementedError()
        
    def callcmd(self,workdir, os='Darwin'):
        """This function should return the command to call the Forward model 
           through subprocess
    
        Args:
          workdir: The directory from which the command should be called
          os: String with the os name, if call command changes with the os 
    
        Returns:
          A string containing the command to be called by subprocess
        """
        raise NotImplementedError()

    def read_data(self,workdir):
        """This function should read the data and output it in a numpy array
    
        Args:
          workdir: The directory of the data file 
    
        Returns:
          A list of numpy arrays containing the data
        """
        raise NotImplementedError()
                   
    def read_grad(self, workdir, param_names):
        """This function should read the gradient and output it in a dictionary 
           of numpy array
    
        Args:
          workdir: The directory of the data file
          param_names: A list containing strings with the name of variables 
                       for which to load the gradient
    
        Returns:
          A list of numpy arrays containing the gradients
        """
        raise NotImplementedError()
        
    def read_Hessian(self, workdir, param_names):
        """This function should read the diagonal approximate Hessian and 
           output it in a dictionary of numpy array
    
        Args:
          workdir: The directory of the data file
          param_names: A list containing strings with the name of variables 
                       for which to load the gradient
    
        Returns:
          A list of numpy arrays containing the Hessians
        """
        raise NotImplementedError()
        
    def read_rms(self, workdir):
        """This function should read the objective function value and output it
           along a normalization factor
    
        Args:
          workdir: The directory of the data file
    
        Returns:
          A tuple with (rms_value, rms_norm) for the batch of data
        """
        raise NotImplementedError()
    
    def op(self, dataid, m, name=None):
        """This function output an instance of Subprocess_op which can output
         a Tensorflow operator for the forward and grad with the attribute op 
           computation
    
        Args:
          dataid: A placeholder inputing the ids of the data to be used 
                  (could also contain the actual data) 
          m: A list of tensors containing the model parameters that can be optimized
             (all memebers of m do not need to be actually optimized in a run )
          name: A name to be given for this op
                  
    
        Returns:
          A Subprocess_op instance with an attribut op giving an operator that 
          can output the data and compute the gradient for the
          function called through subprocess
        """
        return Subprocess_op(self,dataid,m)
    
    
                   
        


    
        