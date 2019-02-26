from scipy import *
from pylab import *
import numpy as np

from BDF4_pr1task2 import BDF_4
from BDF3_pr1task2 import BDF_3

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
  a1 = x[0]
  a2 = x[1]
  a3 = x[2]
  a4 = x[3] 
  return np.array([a3, a4, -a1*lambda_func(a1,a2,k), -a2*lambda_func(a1,a2,k) - 1])


# Settup of explicit problem
# and initial values (are these any good)
x0 = [0.9, 0.1, 0, 0]
pend_prob = apr.Explicit_Problem(pend_rhs, x0, 0)
pend_prob.name = 'Pendulum Position'

# Settup of implicit solver
pend_solv = aso.CVode(pend_prob)

bdf3_solv = BDF_3(pend_prob)
bdf4_solv = BDF_4(pend_prob)

#Simulate
simulation_time = 20

tc, yc = pend_solv.simulate(simulation_time)
pend_solv.plot([1., 1., 0., 0.])

# t, y = bdf3_solv.simulate(simulation_time)
# bdf3_solv.plot([1., 1., 1., 1.])

# t, y = bdf4_solv.simulate(simulation_time)
# bdf4_solv.plot([1., 1., 1., 1.])


