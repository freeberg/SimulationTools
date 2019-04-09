import scipy as sp
import scipy.linalg as SL
import scipy.optimize as spo
import numpy as np
import math
import assimulo.problem as apr
import assimulo.solvers as aso
from assimulo.explicit_ode import Explicit_ODE
from assimulo.implicit_ode import Implicit_ODE
from assimulo.ode import *
import matplotlib as plt
from problem_class import sec_ord_prob


class newmark(Explicit_ODE):
  
  """
  newmark
  """
  tol=0.5
  maxit=100
  maxsteps=1000000000
    

    
  def __init__(self, problem):
    
    Explicit_ODE.__init__(self, problem) #Calls the base class
    
    #Solver options
    self.options["h"] = 0.001
    self.alpha = -1/3
    self.beta = 1/2
    self.gamma = 1/2
    self.HHT = False
    
    #Statistics
    self.statistics["nsteps"] = 0
    self.statistics["nfcns"] = 0
  def _set_h(self,h):
      self.options["h"] = float(h)
  
  def _set_constants(self, alpha, beta, gamma):
      self.alpha = alpha
      self.beta = beta
      self.gamma = gamma
    
  def _set_HHT(self, boolean=False):
    self.HHT = boolean
  
  def _get_h(self):
    return self.options["h"]
        
  h=property(_get_h,_set_h)
        
  def integrate(self, t, y, tf, opts):
    """
    _integrates (t,y,yd,ydd) values until t > tf
    """
    
    h = self.options["h"]
    h = min(h, abs(tf-t))
    
    #Lists for storing the result
    yd = np.array([0.89, 0.09, 0., 0.])
    ydd = np.array([0.,0.,0.,0.])    #causing a problem
    tres = [t]
    yres = [y]
    vres = [yd]
    ares = [ydd]
        
    for i in range(self.maxsteps):
        if t >= tf:
            break
        self.statistics["nsteps"] += 1
    
        t_np1, y_np1, v_np1, a_np1 = self.step_newmark(tres[-1], yres[-1], vres[-1], ares[-1], h)
        
        
        tres.append(t_np1)
        yres.append(y_np1)
        vres.append(v_np1)
        ares.append(a_np1)
    
        t = t_np1
    
        h=min(self.h,np.abs(tf-t))
    else:
      raise Exception('Final time not reached within maximum ' +
                                   'number of steps')
                                   
    return ID_PY_OK, tres, yres
  
  
  def step_newmark(self, T, Y, V, A, h):
      """
      Newmark
      """  
  
      t_n = T
      t_n1 = t_n + h
      # Position, Velocity and Acceleration
      p_n = Y
      v_n = V
      a_n = A
      
      def eq_syst(x):
        t_n1 = t_n + h
        f=self.problem.rhs_given
        
        if (self.HHT):
                gamma = (1-2*self.alpha) / 2
                beta = (1 - self.alpha)**2 / 4
                a = x[0] - p_n - h*v_n - 1/2*h**2 * ((1 - 2*beta) * a_n + 2*beta * x[2])
                b = x[1] - v_n - h*((1 - gamma)*a_n + gamma * x[2])
                c = x[2] - (1 + self.alpha) * f(t_n1, x[0]) + self.alpha * f(t_n, p_n)
        else:
                a = x[0] - p_n - h*v_n - 1./2.*h**2.*(((1.-2.*self.beta)*a_n ) + 2*self.beta*x[2])
                b = x[1] - v_n - h*((1 - self.gamma)*a_n + self.gamma * x[2])
                c = x[2] - f(t_n1, p_n) #,[x1])
        return np.hstack((a, b, c))
      
      guess = np.array([p_n, v_n, a_n])
      x = spo.fsolve(eq_syst, guess)
      p_n1 = x[0:4]
      v_n1 = x[4:8]
      a_n1 = x[8:]
      return t_n1,p_n1,v_n1,a_n1

  
