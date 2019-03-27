from  scipy import *
from  pylab import *
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from assimulo.solvers import CVode

class Newmark(Explicit_ODE):
    
    tol=1.e-6
    maxit=100
    maxsteps=1000000000000
    
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["h"] = 0.001
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    def _set_h(self,h):
      self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]
        
    h=property(_get_h,_set_h)
        
    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = [t]
        yres = [y]
        k = 4
        
        for i in range(self.maxsteps):
          "do something"
          if t >= tf:
                break
          self.statistics["nsteps"] += 1
          t_np1, y_np1 = self.step_Newmark(tres[-1:-(k+1):-1], yres[-1:-(k+1):-1], h)
            
          tres.append(t_np1)
          yres.append(y_np1)
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum ' +
                'number of steps')
        
        return ID_PY_OK, tres, yres
    
    def step_Newmark(self,T,Y, h, hht=0):
        f=self.problem.rhs

        if(hht==0):
          #Newmark
          gamma = 1/2
          beta = 0
        elif(hht==1):
          #HHT method
          alpha = 0
          gamma = (1-2*alpha)/2
          beta = ((1-alpha)/2)**2
        
        
        t_n=T
        p_n, v_n, a_n = Y
        
        # predictor
        t_np1=t_n+h
        p_np1_i=p_n   # zero order predictor
        v_np1_i=v_n
        a_np1_i=a_n
        
        # corrector with fixed point iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            
            p_np1_ip1 = p_n + h*v_n + 1/2*h**2*((1-2*beta)*a_n + 2*beta*a_np1)
            v_np1_ip1 = v_n + h*((1-gamma)*a_n + gamma*a_np1)
            if(hht==0):
              a_np1 = f(p_np1_ip1, v_np1_ip1, t_np1)
            elif(hht==1):
              a_np1 = (1+alpha)*f(p_np1_ip1, v_np1_ip1, t_np1) - alpha*f(p_n,v_n,t_n)
            
            if SL.norm(p_np1_ip1-p_np1_i) < self.tol:
                return t_np1,p_np1_ip1
#            p_np1_i=p_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge ' +
                                         'within % iterations'%i)
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : \
            {name} \n'.format(name=self.problem.name),       verbose)
        self.log_message(' Step-length                    : \
            {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : \
            '+str(self.statistics["nsteps"]),                verbose)               
        self.log_message(' Number of Function Evaluations : \
            '+str(self.statistics["nfcns"]),                 verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : Newmark',                              verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)