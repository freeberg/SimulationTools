#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:44:40 2019

@author: annie
"""
from scipy import *
from pylab import *
import numpy as np
import math as m

import assimulo.problem as apr
# from assimulo.explicit_ode import Explicit_ODE


class sec_ord_prob(apr.Explicit_Problem):
  """y'' = rhs(t,y,y')
            
  Note: the result of a simulation returns value of derivative as well
      To resolve, say y has n components:
      t,y = sim.simulate(10.0)
      (y,dy) = (y[:,:n],y[:,n:])
  """
  #Sets the initial conditons directly into the problem
  y0 = [0.9, 0.1]
  yd0 = [0.0, 0.0]

  def __init__(self, rhs, y0, yd0, t0=0):
    self.rhs_given = rhs
    
    def _set_dmp(d = False): # set damping outside of the class
        self.dmp = d
    
    def newrhs(t, y, yd, **kwargs):
      """transform y'' = rhs(t,y,y') into
      y' = v
      v' = rhs(t,y,v)
      and pass that into Explicit_Problem
      """
      self.dmp = False

      yyd = np.hstack((y, yd))
      n=len(y)#copy entire y

      if self.dmp:
        dv = rhs(t, yyd[:n], yyd[n:])
      else:
        dv = rhs(t, yyd[:n])
      
      return dv # return whatever the rhs has calculated

#    yyd0 = np.hstack((y0, yd0))
    # need to stack y0 and yd0 together as initial condition for newrhs
    super(sec_ord_prob, self).__init__(newrhs, y0, yd0, t0)
  