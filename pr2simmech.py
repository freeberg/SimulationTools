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
def plot_squ_simulations(s):

  y0,yp0 = sq.init_squeezer()
  my_init = get_initial()
  print(" ", y0[:7] - my_init)
  posIndex = list(range(0,7))
  velocityIndex = list(range(7,14))
  lambdaIndex = list(range(14,20))
  t0 = 0.
  algvar = np.ones(np.size(y0))

  if '3' in s:
    squ_prob = apr.Implicit_Problem(squeezer_i3, y0, yp0, t0)
    algvar[lambdaIndex] = 0
    algvar[velocityIndex] = 0
    squ_sim3 = aso.IDA(squ_prob)
    squ_sim3.atol = np.ones(np.size(y0))*1e-7
    squ_sim3.atol[lambdaIndex] = 1e-1
    squ_sim3.atol[velocityIndex] = 1e-5

    squ_sim3.rtol = 1e-8
    squ_sim3.algvar = algvar
    squ_sim3.suppress_alg = True
    t3, y3, yd3 = squ_sim3.simulate(0.03, 5000)

    squ_sim3.plot()
    input("Press Enter to continue...")

  if '2' in s:
    squ_prob = apr.Implicit_Problem(squeezer_i2, y0, yp0, t0)
    algvar[lambdaIndex] = 0
    algvar[velocityIndex] = 1
    squ_sim2 = aso.IDA(squ_prob)
    squ_sim2.atol = np.ones(np.size(y0))*1e-7
    squ_sim2.atol[lambdaIndex] = 1e-5
    squ_sim2.atol[velocityIndex] = 1e-5

    squ_sim2.rtol = 1e-8
    squ_sim2.algvar = algvar
    squ_sim2.suppress_alg = True
    t2, y2, yd2 = squ_sim2.simulate(0.03, 5000)
    
    squ_sim2.plot()
    input("Press Enter to continue...")

  if '1' in s:
    print("not implemented index 1 yet")
    input("Press Enter to continue...")
  
  if '23c' in s:
    pltVector = (y3[:,:7]+1*np.pi)%(2*np.pi)-1*np.pi
    beta_plt = plt.plot(t3, pltVector,'.' ) 
    plt.title('Index3')
    plt.ylabel('Vinkel (rad)')
    plt.xlabel('Tid (s)')
    plt.legend(["beta", "theta", "gamma", "phi", "delta", "omega", "epsilon"], loc = 'lower left')
    plt.show()

plot_squ_simulations("123c")