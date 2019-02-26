from  scipy import *
from  pylab import *
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from assimulo.solvers import CVode
from BDF2_wuRNl45 import BDF_2 as bdf2

import assimulo.problem as apr

class BDF_3(Explicit_ODE):
    """
    BDF-3
    """
    tol=1.e-6
    maxit=100     
    maxsteps=10000000
    
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
        k = 3
        
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            
            # winding up
            if i <= k-1:
              t_np1,y_np1 = self.step_EE(tres[-1],yres[-1], h)
            # continue computing
            elif i <= k:
              t_np1,y_np1 = bdf2.step_BDF2(self,tres[-1:-k:-1],yres[-1:-k:-1], h)
            else:  
              t_np1, y_np1 = self.step_BDF3(tres[-1:-(k+1):-1], yres[-1:-(k+1):-1], h)
            
            tres.append(t_np1)
            yres.append(y_np1)
            
            t = t_np1
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    
    def step_EE(self, t, y, h):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        return t + h, y + h*f(t, y) 
        
    def step_BDF3(self,T,Y, h):
        """
        BDF-3 with Fixed Point Iteration and Zero order predictor
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1+(alpha_3*y_nm2)=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
        """
        alpha=[11., -18. , 9. , -2.]
        f=self.problem.rhs
        
        t_n,t_nm1,t_nm2=T
        y_n,y_nm1,y_nm2=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1+alpha[3]*y_nm2)+h*6*f(t_np1,y_np1_i))/alpha[0]
            if SL.norm(y_np1_ip1-y_np1_i) < self.tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF3',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)

# # Construct RHS
# def lambda_func(var1, var2, k):
#   taljare = sqrt(var1**2+var2**2) -1
#   namnare = sqrt(var1**2+var2**2)
  
#   l_func = k * taljare/namnare
#   return l_func

# def pend_rhs(t,x):
#   k = 10**5
#   a1 = x[0] #blue
#   a2 = x[1] #orange
#   a3 = x[2] #green
#   a4 = x[3] #red
#   return np.array([a3, a4, -a1*lambda_func(a1,a2,k), -a2*lambda_func(a1,a2,k) - 1])


# # Settup of explicit problem
# # and initial values (are these any good)
# x0 = [1, 0, 0, 0]
# pend_prob = apr.Explicit_Problem(pend_rhs, x0, 0)
# pend_prob.name = 'Stiff spring Pendulum'

# # Settup of implicit solver
# pend_solv = BDF_3(pend_prob)
# t, y = pend_solv.simulate(5)
# pend_solv.plot()
