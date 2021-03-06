from  scipy import *
from  pylab import *
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from assimulo.solvers import CVode
from BDF2_wuRNl45 import BDF_2 as bdf2
from BDF3_pr1task2 import BDF_3 as bdf3

class BDF_4(Explicit_ODE):
    """
    BDF-4
    """
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
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            if i <= k-3:  # initial step
                t_np1,y_np1 = self.step_EE(tres[-1], yres[-1], h)
            elif i==k-2:
                t_np1,y_np1 = bdf2.step_BDF2(self, tres[-1:-(k-1):-1], yres[-1:-(k-1):-1], h)
            elif i==k-1:
                t_np1,y_np1 = bdf3.step_BDF3(self, tres[-1:-(k):-1], yres[-1:-(k):-1], h)
            else:   
                t_np1, y_np1 = self.step_BDF4(tres[-1:-(k+1):-1], yres[-1:-(k+1):-1], h)
            
            tres.append(t_np1)
            yres.append(y_np1)

            t = t_np1
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum ' +
                'number of steps')
        
        return ID_PY_OK, tres, yres
    
    def step_EE(self, t, y, h):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        return t + h, y + h*f(t, y) 
        
    def step_BDF4(self,T,Y, h):
        """
        BDF-4 with Fixed Point Iteration and Zero order predictor
        """
        alpha=[25. , -48. , 36., -16., 3.]
        f=self.problem.rhs
        
        t_n,t_nm1,t_nm2,t_nm3=T
        y_n,y_nm1,y_nm2,y_nm3=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1+alpha[3]*y_nm2+alpha[4] \
                       *y_nm3)+h*12*f(t_np1,y_np1_i))/alpha[0]
            if SL.norm(y_np1_ip1-y_np1_i) < self.tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
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
        self.log_message(' Solver            : BDF4',                              verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)