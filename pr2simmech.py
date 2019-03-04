from  scipy import *
import scipy.optimize as op
from  pylab import *
import numpy as np
import matplotlib.pyplot as plt
import assimulo.problem as apr
import assimulo.solvers as aso

from BDF4_pr1task2 import BDF_4
from BDF3_pr1task2 import BDF_3

import squeezer_HsnppkU as sq

# Res
def squeezer_i3 (t, y, yp):
  r = sq.squeezer(t, y, yp, 3)
  return r

def squeezer_i2 (t, y, yp):
  r = sq.squeezer(t, y, yp, 2)
  return r

def g(y):
  y = np.insert(y, 1, 0) #for theta = 0 since only have six equations for seven variables
  g = sq.squeezer(0, y, zeros(7,), 0) 
  return g  

def get_initial():
  y0 = zeros(6,)
  q0 = op.fsolve(g, y0)
  q0 = np.insert(q0, 1, 0)
  return q0


# Settup of explicit problem
# and initial values (are these any good)
y0,yp0 = sq.init_squeezer()
my_init = get_initial()
print(" ", y0[:7] - my_init)
posIndex = list(range(0,7))
velocityIndex = list(range(7,14))
lambdaIndex = list(range(14,20))
t0 = 0.
algvar = np.ones(np.size(y0))


squ_prob = apr.Implicit_Problem(squeezer_i2, y0, yp0, t0)
algvar[lambdaIndex] = 0
algvar[velocityIndex] = 1
sim = aso.IDA(squ_prob)
sim.atol = np.ones(np.size(y0))*1e-7
sim.atol[lambdaIndex] = 1e-5
sim.atol[velocityIndex] = 1e-5

sim.rtol = 1e-8
sim.algvar = algvar
sim.suppress_alg = True
t, y, yd = sim.simulate(0.03, 5000)

sim.plot()
#pltVector = (y[:,:7]+1*np.pi)%(2*np.pi)-1*np.pi


