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
    yd = 0
    ydd = 0
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
    return Explicit_ODE.ID_PY_OK, tres, yres
  
  
  def step_newmark(self, T, Y, V, A, h):
      """
      Newmark
      """
  
      f=self.problem.rhs
  
  
      t_n = T
      t_n1 = t_n + h
      # Position, Velocity and Acceleration
      p_n = Y
      v_n = V
      a_n = A
      # predictor
      p_np1 = p_n + h*f(t_n, p_n)  
      # corrector with fixed point iteration
      for i in range(self.maxit):
          self.statistics["nfcns"] += 1
          if (self.HHT):
            gamma = (1-2*self.alpha) / 2
            beta = (1 - self.alpha)**2 / 4
            p_n1 = p_n + h*v_n + 1/2 * h**2 * ((1 - 2*beta) * a_n + 2*beta * a_n1)
            v_n1 = v_n + h*((1 - gamma)*a_n + gamma * a_n1)
            a_n1 = (1 + self.alpha) * f(t_n1, p_n1, v_n1) - self.alpha * f(t_n, p_n, v_n)
          else:
              p_n1 = p_n + h*v_n + 1/2 * h**2 * ((1-2*self.beta)*a_n )#+ 2*self.beta*a_n1)
              a_n1 = f(t_n1, p_n1) #f(t_n1, p_n1, v_n1)
              v_n1 = v_n + h*((1 - self.gamma)*a_n + self.gamma * a_n1)
          if SL.norm(p_np1-p_n1) < self.tol:
              return t_n1,p_n1,v_n1,a_n1
      else:
          raise Exception('Corrector could not converge ' +
                                       'within % iterations'%i)
