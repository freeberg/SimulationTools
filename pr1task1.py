from  scipy import *
from  pylab import *
import assimulo.problem as apr
import assimulo.solvers as aso

tol=1.e-8     
maxit=100     
maxsteps=500




#Define the rhs
def rhs(t,y):
    y1dot = y[2]
    y2dot = y[3]
    y3dot = -y[0] * lamb(y[0], y[1], 50)
    y4dot = -y[1] * lamb(y[0], y[1], 50) - 1
    return array([y1dot, y2dot, y3dot, y4dot])
    
#The lambda function
def lamb(y1, y2, k):
    return k * (sqrt(y1**2 + y2**2) - 1) / sqrt(y1**2 + y2**2)

pi = 3.14
x0 = array([pi/2, 1, 0.2, 0.1])

pend_prob = apr.Explicit_Problem(rhs, x0, 0)

pend_cvode = aso.CVode(pend_prob)
pend_cvode.simulate(10)
pend_cvode.plot()