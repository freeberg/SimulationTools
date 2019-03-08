import scipy
import scipy.optimize as so
import scipy.linalg as sl
import numpy
import math
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
import matplotlib.pyplot as plt

#    zpp, phipp_s, phipp_b = yp[3], yp[4], yp[5]
#    zp, phip_s, phip_b = yp[0], yp[1], yp[2]
#    lambdap_1, lambdap_2  = yp[6], yp[7]
#    z, phi_s, phi_b = y[0], y[1], y[2]
#    v, u_s, u_b = y[3], y[4], y[5]
#    lambda_1, lambda_2  = y[6], y[7]

m_s = 3.0e-4 # Mass of sleeve [kg]
j_s = 5.0e-9 # Moment of inertia of the sleeve [kgm]
m_b = 4.5e-3 # Mass of bird [kg]
# #  #masstotal=mS+mB # total mass
j_b = 7.0e-7 # Moment of inertia of bird [kgm]
r_0 = 2.5e-3 # Radius of the bar [m]
r_s = 3.1e-3 # Inner Radius of sleeve [m]
#h_s = 5.8e-3 # 1/2 height of sleeve [m]
h_s = 2e-2
l_s = 1.0e-2 # verical distance sleeve origin to spring origin [m]
l_g = 1.5e-2 # vertical distance spring origin to bird origin [m]
h_b = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]           
l_b = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
c_p = 5.6e-3 # rotational spring constant [N/rad]
g  = 9.81 #  [m/s^2]
global hack
hack = 0

def res(t,y,yp,sw):
    #print('res')
    if sw[0]:
        return motion_state1(t,y,yp)
    elif sw[1]:
        return motion_state2(t,y,yp)
    elif sw[2]:
        return motion_state3(t,y,yp)
    else:
        print('wrong sw in res')
        return -1               

def state_events(t, y, yp, sw):
    
    phi_b=y[2]
    phi_s=y[1]
    lambda_1=y[6]
    
    if sw[0]:   
        e_1 = h_s*phi_s + ( r_s -  r_0)
        e_2 = h_s*phi_s - ( r_s -  r_0)

    elif sw[1]:
        e_1 = lambda_1
        e_2 = 1

    else: 
        e_1 = lambda_1  
        e_2 = h_b*phi_b-(l_s+l_g-l_b-r_0)
    
    
    return numpy.array([e_1, e_2])

def moi(solver):
    y = solver.y
    yp = solver.yd
    a = ( j_b+ m_b* l_g* l_g)
    I =  m_b* l_g*yp[0] + ( m_b* l_s* l_g)*yp[1] + a*yp[2]
    phip_b = I/a;
    yp[2] = phip_b
    y[5] = phip_b
    yp[0] = 0
    yp[1] = 0
    y[3] = 0
    y[4] = 0
    solver.y = y
    solver.yd = yp 


def handle_event(solver, event_info):
    
    phip_b=solver.yd[2]

    state_info = event_info[0]
  
    if solver.sw[0]:
        if (phip_b < 0 and state_info[0]):
            print('phip_b before moi', phip_b)
            moi(solver)
            print('phip_b after moi', solver.yd[2])
            solver.sw=[0,1,0,0]
            print('state 2')
        elif (phip_b > 0 and state_info[1]): 
            print('phip_b before moi', solver.yd[2])
            moi(solver)
            print('phip_b after moi', solver.yd[2])
            solver.sw=[0,0,1,0]
            print('state 3')
    elif solver.sw[1]:
        if (state_info[0]):
            solver.sw=[1,0,0,0]
            #solver.yd[3] = -9.82
            print('state 1')
    elif solver.sw[2]:
        if (phip_b < 0 and state_info[0]):
            solver.sw=[1,0,0,0]
            #solver.yd[3]=-9.82
            print('state 1')
        elif (phip_b > 0 and state_info[1]):
            #solver.y[4] = -solver.y[4] these are the lines that Claus suggested, we changed them. 
            solver.y[5] = -solver.y[5]
            #solver.yp[1] = -solver.yp[1]
            solver.yd[2] = -solver.yd[2]
            print('state 4')
            global hack
            hack = hack + 1
            print('state 3')



def motion_state1(t,y, yp):
    lambda_1, lambda_2  = y[6], y[7]
    
    r = default_residual(y, yp)
    r[0] = r[0]
    r[1] = r[1] + lambda_1
    r[2] = r[2] + lambda_2
    r[3] = lambda_1
    r[4] = lambda_2
    
    return r
def motion_state2(t,y, yp):
    zp, phi_s, phip_s = yp[0], y[1], yp[1]
    lambda_1, lambda_2  = y[6], y[7]

    
    r = default_residual(y, yp)
    r[0] = r[0] + lambda_2
    r[1] = r[1] + h_s*lambda_1+r_s*lambda_2
    r[2] = r[2]
    r[3] = ( r_s -  r_0) +  h_s*phi_s
    #r[3] = yp[4]
    r[4] = zp +  r_s*phip_s
    #r[4] = yp[3]+r_s*yp[4]

    return r

def motion_state3(t,y,yp):

    zp, phi_s, phip_s = yp[0], y[1], yp[1]
    lambda_1, lambda_2  = y[6], y[7]

    r = default_residual(y, yp)
    r[0] = r[0] + lambda_2
    r[1] = r[1] -  h_s*lambda_1+ r_s*lambda_2
    r[2] = r[2]
    #r[3] = 0
    r[3] = ( r_s -  r_0) -  h_s*phi_s
    #r[3] = yp[4]
    r[4] = zp +  r_s*phip_s
    #r[4] = yp[3]+r_s*yp[4]

    #r[6] = 0
    return r

def default_residual(y, yp):
    zpp, phipp_s, phipp_b = yp[3], yp[4], yp[5]
    zp, phip_s, phip_b = yp[0], yp[1], yp[2]
    z, phi_s, phi_b = y[0], y[1], y[2]
    v, u_s, u_b = y[3], y[4], y[5]

    r = numpy.zeros(numpy.size(y))
    r[0] = ( m_s +  m_b)*zpp +  m_b* l_s*phipp_s+ m_b* l_g*phipp_b+( m_s+ m_b)* g
    r[1] = ( m_b* l_s)*zpp+( j_s+ m_b* l_s* l_s)*phipp_s + ( m_b* l_s* l_g)*phipp_b - c_p*(phi_b-phi_s) +  m_b* l_s* g
    r[2]  =  m_b* l_g*zpp + ( m_b* l_s* l_g)*phipp_s+ ( j_b +  m_b* l_g* l_g)*phipp_b -  c_p*(phi_s-phi_b)+ m_b* l_g* g
    r[3] = 63 
    r[4] = 63
    r[5] = v - zp
    r[6] = u_s - phip_s
    r[7] = u_b - phip_b
    return r

if (-1):
    print('good')
else:
    print('-1 ger inte true')
          
t0 = 0;
startsw = [1,0,0,0]
y0 = numpy.array([0.5, 0,0, -0, 0, 0.5,-1e-4,0])
yd0 =  numpy.array([-0, 0, 0.5,-g, 1e-12, 0, 0, 0])

w0 = -0.91
y0 = numpy.array([0.5, 0,0, -0, w0, w0,-1e-4,0])
yd0 =  numpy.array([-0, w0, w0,-g, 1e-12, 0, 0, 0])

#y0 = numpy.array([4.83617428e-01, -3.00000000e-02, -2.16050178e-01, 1.67315232e-16, -5.39725367e-14, -1.31300925e+01, -7.20313572e-02, -6.20545138e-02])
#yd0 = numpy.array([1.55140566e-17, -5.00453439e-15, -1.31302838e+01, 6.62087352e-13, -2.13577297e-10, 2.21484026e+02, -4.67637454e+00, -2.89824658e+00])
#startsw = [0,1, 0, 0]

problem = Implicit_Problem(res, y0, yd0, t0, sw0=startsw)

problem.state_events = state_events
problem.handle_event = handle_event
problem.name = 'Woodpecker'

phipIndex = [4, 5]
lambdaIndex = [6, 7]
sim = IDA(problem)
sim.rtol = 1e-6

sim.atol[phipIndex] = 1e8
sim.algvar[phipIndex] = 1
sim.atol[lambdaIndex] = 1e8
sim.algvar[lambdaIndex] = 1 

sim.suppress_alg = True
ncp = 500

tfinal = 2
t, y, yd = sim.simulate(tfinal, ncp)
y = y[:,[ 0,  ]]
plt.plot(t, y)
plt.legend(["z", "phi_s", "phi_b", "zp", "phip_s", "phip_b", "lambda_2", "lambda_2"], loc = 'lower left')
print("Number of pecks", hack)
plt.ylabel('Höjd (m)')

plt.xlabel('Tid (s)')
plt.show()
print(y[-1, :])
print(yd[-1, :])







####################################################################################33#




def res2(self,t,y,yp,sw):
    print('res2')
    if sw[0]:
        return motion_state1(t,y,yp)
    elif sw[1]:
        return motion_state2(t,y,yp)
    elif sw[2]:
        return motion_state3(t,y,yp)
    else:
        print('wrong sw in res')
        return -1 


def variablesI2(self, y):
    self.m_s = 3.0e-4 # Mass of sleeve [kg]
    self.j_s = 5.0e-9 # Moment of inertia of the sleeve [kgm]
    self.m_b = 4.5e-3 # Mass of bird [kg]
    #masstotal=mS+mB # total mass
    self.j_b = 7.0e-7 # Moment of inertia of bird [kgm]
    self.r_0 = 2.5e-3 # Radius of the bar [m]
    self.r_s = 3.1e-3 # Inner Radius of sleeve [m]
    self.h_s = 5.8e-3 # 1/2 height of sleeve [m]
    self.l_s = 1.0e-2 # verical distance sleeve origin to spring origin [m]
    self.l_g = 1.5e-2 # vertical distance spring origin to bird origin [m]
    self.h_b = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]           
    self.l_b = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
    self.c_p = 5.6e-3 # rotational spring constant [N/rad]
    self.g  = 9.81 #  [m/s^2] 

def variables(y):
  global m_s, j_s, m_b, j_b, r_0, r_s, h_s, l_s, l_g, h_b, l_b, c_p, g