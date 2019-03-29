import scipy as sp
import scipy.linalg as SL
import numpy as np
import math
import assimulo.problem as apr
import assimulo.solvers as aso
from assimulo.explicit_ode import Explicit_ODE
from assimulo.implicit_ode import Implicit_ODE
from assimulo.ode import *
import matplotlib as plt


class newmark(Explicit_ODE):

    """
    newmark
    """
    tol=1.e-6
    maxit=100
    maxsteps=100000
    

    
    def __init__(self, problem):

        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["h"] = 0.001
        self.alpha = -1/3
        self.gamma = 1/2
        self.HHT = False
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    def _set_h(self,h):
      self.options["h"] = float(h)

    def _set_constants(self, alpha, gamma):
      self.alpha = alpha
      self.gamma = gamma

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


        #nbr of initial steps
        k = 1
        
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            if i <= k-1:  # initial step
                t_np1,y_np1 = self.step_ee(tres[-1], yres[-1], h)
            else:
                t_np1, y_np1 = self.step_newmark(self, tres[-1], yres[-1:2], h)  
            
            tres.append(t_np1)
            yres.append(y_np1)

            t = t_np1
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum ' +
                'number of steps')
        
        return ID_PY_OK, tres, yres


    def step_ee(self, t, y, h):
      self.statistics["nfcns"] += 1
      f = self.problem.rhs
      return t + h, y + h*f(t, y)

    def step_newmark(self,T,Y, h):
        """
        Newmark
        """
        # beta = 0

        f=self.problem.rhs


        t_n = T
        t_n1 = t_n + h
        # Position, Velocity and Acceleration
        p_n, v_n, a_n = Y
        # predictor
        # t_np1=t_n+h
        p_n1=p_n   # zero order predictor CHECK IF CORRECT!
        # corrector with fixed point iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            if (HHT):
              gamma = (1-2*alpha) / 2
              beta = (1 - alpha)**2 / 4
              a_n1 = (1 - alpha) * f(p_n1, t_n1) - alpha * f(p_n, t_n)
              p_n1 = p_n + h*v_n + 1/2 * h**2 * ((1 - 2*beta) * a_n + 2*beta * a_n1)
              v_n1 = v_n + h*((1 - gamma)*a_n + gamma * a_n1)
            else:
                p_n1 = p_n + h*v_n + 1/2 * h**2 * a_n
                a_n1 = f(p_n1, t_n1)
                v_n1 = v_n + h*((1 - gamma)*a_n + gamma * a_n1)
            if SL.norm(p_n-p_n1) < self.tol:
                return t_np1,y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge ' +
                                         'within % iterations'%i)
