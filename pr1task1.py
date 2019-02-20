from scipy import *
from pylab import *
import numpy as np

import assimulo.problem as apr
import assimulo.solvers as aso
from BDF3_pr1task2 import BDF_3
from BDF4_pr1task2 import BDF_4

# Construct LHS
def lambda_func(var1, var2, k):
  taljare = sqrt(var1**2+var2**2) -1
  namnare = sqrt(var1**2+var2**2)
  
  l_func = k * taljare/namnare
  return l_func

def pend_rhs(t,x):
  k = 10**3
  a1 = x[0] #blue
  a2 = x[1] #orange
  a3 = x[2] #green
  a4 = x[3] #red
  return np.array([a3, a4, -a1*lambda_func(a1,a2,k), -a2*lambda_func(a1,a2,k) - 1])


# Settup of explicit problem
# and initial values (are these any good)
x0 = [0.9, 0.1, 0, 0]
pend_prob = apr.Explicit_Problem(pend_rhs, x0, 0)
pend_prob.name = 'Pendulum'

# Settup of implicit solver
pend_solv = BDF_4(pend_prob)
pend_solv.simulate(0.5)

# Plot solution
pend_solv.plot()