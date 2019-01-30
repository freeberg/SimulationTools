from  scipy import *
from  pylab import *
import assimulo.problem as apr
import assimulo.solvers as aso

g = 9.81
e = 1

def pend_rhs(t,x):
    alpha1 = x[0]
    alpha2 = x[1]
    return array([alpha2, -g/e * alpha1])

pi = 3.14
x0 = array([pi/2, 0])

pend_prob = apr.Explicit_Problem(pend_rhs, x0, 0)

pend_cvode = aso.CVode(pend_prob)
pend_cvode.simulate(10)
pend_cvode.plot()