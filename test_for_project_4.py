#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:01:39 2019

@author: annie
"""
from scipy import *
from pylab import *
import numpy as np

from newmark import newmark

import assimulo.problem as apr
import assimulo.solvers as aso

# Construct RHS
def lambda_func(var1, var2, k):
  numerator = sqrt(var1**2+var2**2) -1
  dinomerator = sqrt(var1**2+var2**2)
  
  l_func = k * numerator/dinomerator
  return l_func

def pend_rhs(t,x):
  k = 10**3
  a1 = x[0] #xdd
  a2 = x[1] #ydd
  a3 = x[2] #x
  a4 = x[3] #y
  
  #extend rhs to be able to use new problem class....
  return np.array([a3, a4, -a1*lambda_func(a1,a2,k), -a2*lambda_func(a1,a2,k) - 1])


# Settup of explicit problem
# and initial values (are these any good)
x0 = [0.9, 0.1, 0, 0]
pend_prob = apr.Explicit_Problem(pend_rhs, x0, 0)
pend_prob.name = 'Stiff spring Pendulum'

# Settup of implicit solver
pend_solv = newmark(pend_prob)

#Simulate
simulation_time = 20

pend_solv.simulate(simulation_time)
pend_solv.plot()




