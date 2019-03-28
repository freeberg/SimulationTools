from  scipy import *
import scipy.optimize as op
import scipy.linalg as linalg
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

def squeezer_i1 (t, y):
  if (t == 0):
        y, yd0 = sq.init_squeezer()
        return yd0
  y, m, ff, gp, gpp = sq.squeezer(t, y, zeros(7), 1)
  invM = linalg.inv(m)
  a = dot(gp, dot(invM, gp.T))
  b = gpp + dot(gp,(dot(invM, ff)))
  lamb = linalg.solve(a, b)
  x = ff - dot(gp.T, lamb)
  w = dot(invM, x)
  yp = zeros(14)
  yp[0:7] = y[7:14]
  yp[7:14] = w
  return yp

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
def plot_squ_simulations(s, lagrange=False):

  y0,yp0 = sq.init_squeezer()
  my_init = get_initial()
  print(" ", np.abs(y0[:7] - my_init))
  pos = list(range(0,7))
  velo = list(range(7,14))
  lamb = list(range(14,20))
  t0 = 0.
  algvar = np.ones(np.size(y0))

############ INDEX-3 ###############
  if '3' in s:
    squ_prob = apr.Implicit_Problem(squeezer_i3, y0, yp0, t0)
    algvar[lamb] = 0
    algvar[velo] = 0
    squ_sim3 = aso.IDA(squ_prob)
    squ_sim3.atol = np.ones(np.size(y0))*1e-7
    squ_sim3.atol[lamb] = 1e-1
    squ_sim3.atol[velo] = 1e-5

    squ_sim3.rtol = 1e-8
    squ_sim3.algvar = algvar
    squ_sim3.suppress_alg = True
    t3, y3, yd3 = squ_sim3.simulate(0.03, 5000)
    if lagrange:
      p = (y3[:,lamb])#+1*np.pi)%(2*np.pi)-1*np.pi
    else:
      p = (y3[:,pos]+1*np.pi)%(2*np.pi)-1*np.pi 
    plt.plot(t3, p, '-', ms=0.7)
    plt.title('Index-3')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    #plt.legend(["beta", "theta", "gamma", "phi", "delta", "omega", "epsilon"], loc = 'lower left')
    plt.show()
    
    input("Press Enter to continue...")

############ INDEX-2 ###############
  if '2' in s:
    squ_prob = apr.Implicit_Problem(squeezer_i2, y0, yp0, t0)
    algvar[lamb] = 0
    algvar[velo] = 1
    squ_sim2 = aso.IDA(squ_prob)
    squ_sim2.atol = np.ones(np.size(y0))*1e-7
    squ_sim2.atol[lamb] = 1e-5
    squ_sim2.atol[velo] = 1e-5

    squ_sim2.rtol = 1e-8
    squ_sim2.algvar = algvar
    squ_sim2.suppress_alg = True
    t2, y2, yd2 = squ_sim2.simulate(0.03, 5000)
    if lagrange:
      p = (y2[:,lamb])#+1*np.pi)%(2*np.pi)-1*np.pi
    else:
      p = (y2[:,pos]+1*np.pi)%(2*np.pi)-1*np.pi 
    plt.plot(t2, p, '-', ms=0.7)
    plt.title('Index-2')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    #plt.legend(["beta", "theta", "gamma", "phi", "delta", "omega", "epsilon"], loc = 'lower left')
    plt.show()

    input("Press Enter to continue...")

############ INDEX-1 ###############
  if '1' in s:
    squ_prob = apr.Explicit_Problem(squeezer_i1, y0[0:14], t0)
    squ_sim1 = aso.RungeKutta34(squ_prob)
    squ_sim1.atol = np.ones(14)*1e-7
    #yp = squeezer_i1(0.01, y0[0:14])
    #print(yp)
    t1, y1 = squ_sim1.simulate(0.03, 10000)
    # squ_sim1.plot()
    p = (y1[:,pos]+1*np.pi)%(2*np.pi)-1*np.pi 
    plt.plot(t1, p, '-', ms=0.7)
    plt.title('Index-1')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend(["beta", "theta", "gamma", "phi", "delta", "omega", "epsilon"], loc = 'lower left')
    plt.show()
    input("Press Enter to continue...")

  if '3t' in s:
    dt = [1000000 * np.absolute(t3[i+1] - t3[i]) for i in range(len(t3)-1)]
    print((dt[100:120]))
    print((t3[100:120]))
    plt.plot(t3[0:len(t3)-1], dt, '.', ms=0.7)
    plt.show()
  
  if '2t' in s:
    dt = [1000000 * np.absolute(t2[i+1] - t2[i]) for i in range(len(t2)-1)]
    print((dt[100:120]))
    print((t2[100:120]))
    plt.plot(t2[0:len(t2)-1], dt, '.', ms=0.7)
    plt.show()


###### Difference in lagrange ######
  if '23c' in s:
    if lagrange:
      p = np.absolute((y3[:,lamb] - y2[:,lamb]))#+1*np.pi)%(2*np.pi)-1*np.pi
    else:
      p = ((y3[:,pos]+1*np.pi)%(2*np.pi)-1*np.pi) - ((y2[:,pos]+1*np.pi)%(2*np.pi)-1*np.pi) 
    plt.plot(t3, p, '-', ms=0.7)
    plt.title('Diff index-3 and index-2')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    #plt.legend(["beta", "theta", "gamma", "phi", "delta", "omega", "epsilon"], loc = 'lower left')
    plt.show()

# Run default all indexes
plot_squ_simulations("23c", True)

# Run simulations to compare Lagrange Multipliers
# lagrange = True
# plot_squ_simulations("23c", lagrange)