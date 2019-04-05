#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:01:39 2019

@author: annie
"""
from scipy import *
from pylab import *
import numpy as np
import math as m

from newmark_by_ay import newmark
from problem_class import sec_ord_prob

import assimulo.problem as apr

# Construct RHS
def lambda_func(var1, var2, k):
  numerator = m.sqrt(var1**2+var2**2) -1
  dinomerator = m.sqrt(var1**2+var2**2)
  
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


# The question is if the p, v, and a should each be of size 4
# as it is now x0 contains both position and acceleration...


# Settup of explicit problem
# and initial values (are these any good)
x0 = [0.9, 0.1, 0., 0.]
pend_prob = sec_ord_prob(pend_rhs, x0, 0.)
pend_prob.name = 'Stiff spring Pendulum'

# Settup of explicit solver
pend_solv = newmark(pend_prob)
alpha = 0.
beta = 0.
gamma = 1.
pend_solv._set_constants(alpha, beta, gamma)
pend_solv._set_HHT(False)

#Simulate
simulation_time = 0.8

pend_solv.simulate(simulation_time)
pend_solv.plot()




