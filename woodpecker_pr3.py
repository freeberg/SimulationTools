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
global pecks
pecks = 0 

from scipy import *
import numpy as np
import assimulo.problem as apr
import assimulo.solvers as aso
import matplotlib.pyplot as plt

def state_events(t, y, yp, sw):
    
    lmb1 = y[6]
    fi_s, fi_b = y[1:3]

    if sw[0]:
        e_1 = h_s*fi_s + r_s - r_0
        e_2 = h_s*fi_s - r_s + r_0
        #print("In state", sw)
    elif sw[1]:
        e_1 = lmb1
        e_2 = 1
        print("In state", sw)
    elif sw[2]:
        e_1 = lmb1  
        e_2 = h_b*fi_b-(l_s+l_g-l_b-r_0)
        print("In state", sw)
    elif sw[3]:
    #Never really used, but needed if start state is 4
        e_1 = 1
        e_2 = 1

    return np.array([e_1, e_2])

def mom_conserv(solver):
    print("In mom")
    zp = solver.y[3]
    fi_sp = solver.y[4]
    fi_bp = solver.y[5]
    new_fi_bp = ((m_b*l_g)*zp + (m_b*l_s*l_g)*fi_sp) / (j_b+ m_b*l_g*l_g) + fi_bp
    solver.y[3:6] = np.array([0, 0, new_fi_bp])
    solver.yd[0:3] = np.array([0, 0, new_fi_bp])
    

def handle_event(solver, event_info):
    print("In handle")
    state_info = event_info[0]
    fi_bp = solver.yd[2]

    if solver.sw[0]:
        if (fi_bp < 0 and state_info[0]):
            mom_conserv(solver)
            solver.sw=[0,1,0,0]
        elif (fi_bp > 0 and state_info[1]): 
            mom_conserv(solver)
            solver.sw=[0,0,1,0]
    elif solver.sw[1]:
        if (state_info[0]):
            solver.sw=[1,0,0,0]
    elif solver.sw[2]:
        if (fi_bp < 0 and state_info[0]):
            solver.sw=[1,0,0,0]
        elif (fi_bp > 0 and state_info[1]):
            add_peck(solver)
    elif solver.sw[3]:
        add_peck(solver)
        solver.sw = [1,0,0,0]



def stateI(t,y, yp):
    lmb1, lmb2  = y[6], y[7]
    zpp, fi_spp, fi_bpp = yp[3], yp[4], yp[5]
    zp, fi_sp, fi_bp = yp[0], yp[1], yp[2]
    z, fi_s, fi_b = y[0], y[1], y[2]
    v, u_s, u_b = y[3], y[4], y[5]

    r = zeros(8)
    r[0] = freefall(zpp, fi_bpp, fi_spp)
    r[1] = eq1(zpp, fi_b, fi_bpp, fi_s, fi_spp, [lmb1, 0])
    r[2] = eq2(zpp, fi_b, fi_bpp, fi_s, fi_spp, lmb2)
    r[3] = lmb1
    r[4] = lmb2
    r[5] = v - zp
    r[6] = u_s - fi_sp
    r[7] = u_b - fi_bp
    
    return r


def stateII(t,y, yp):
    lmb1, lmb2  = y[6], y[7]
    zpp, fi_spp, fi_bpp = yp[3], yp[4], yp[5]
    zp, fi_sp, fi_bp = yp[0], yp[1], yp[2]
    z, fi_s, fi_b = y[0], y[1], y[2]
    v, u_s, u_b = y[3], y[4], y[5]

    r = zeros(8)
    r[0] = freefall(zpp, fi_bpp, fi_spp, lmb2)
    r[1] = eq1(zpp, fi_b, fi_bpp, fi_s, fi_spp, [h_s*lmb1, r_s*lmb2])
    r[2] = eq2(zpp, fi_b, fi_bpp, fi_s, fi_spp)
    r[3] = r_s - r_0 + h_s*fi_s
    r[4] = zp - r_s*fi_sp
    r[5] = v - zp
    r[6] = u_s - fi_sp
    r[7] = u_b - fi_bp

    return r


def stateIII(t,y,yp):
    lmb1, lmb2  = y[6], y[7]
    zpp, fi_spp, fi_bpp = yp[3], yp[4], yp[5]
    zp, fi_sp, fi_bp = yp[0], yp[1], yp[2]
    z, fi_s, fi_b = y[0], y[1], y[2]
    v, u_s, u_b = y[3], y[4], y[5]

    r = zeros(8)
    r[0] = freefall(zpp, fi_bpp, fi_spp, lmb2)
    r[1] = eq1(zpp, fi_b, fi_bpp, fi_s, fi_spp, [-h_s*lmb1, r_s*lmb2])
    r[2] = eq2(zpp, fi_b, fi_bpp, fi_s, fi_spp)
    r[3] = r_s - r_0 - h_s*fi_s
    r[4] = zp + r_s*fi_sp
    r[5] = v - zp
    r[6] = u_s - fi_sp
    print("u_s", u_s, "fi_sp", fi_sp)
    r[7] = u_b - fi_bp

    return r



def default_residual(y, yp):
    zpp, phipp_s, phipp_b = yp[3], yp[4], yp[5]
    zp, phip_s, phip_b = yp[0], yp[1], yp[2]
    z, phi_s, phi_b = y[0], y[1], y[2]
    v, u_s, u_b = y[3], y[4], y[5]

    r = np.zeros(np.size(y))
    r[0] = ( m_s +  m_b)*zpp +  m_b* l_s*phipp_s+ m_b* l_g*phipp_b+( m_s+ m_b)* g
    r[1] = ( m_b* l_s)*zpp+( j_s+ m_b* l_s* l_s)*phipp_s + ( m_b* l_s* l_g)*phipp_b - c_p*(phi_b-phi_s) +  m_b* l_s* g
    r[2] = m_b* l_g*zpp + ( m_b* l_s* l_g)*phipp_s+ ( j_b +  m_b* l_g* l_g)*phipp_b -  c_p*(phi_s-phi_b)+ m_b* l_g* g
    r[3] = 0
    r[4] = 0
    r[5] = v - zp
    r[6] = u_s - phip_s
    r[7] = u_b - phip_b
    return r

def stateIV(t,y,yp):
    return stateIII(t,y,yp)

def add_peck(solver):
    global pecks
    pecks = pecks + 1
    solver.yd[2] = -solver.yd[2]
    solver.y[5] = -solver.y[5]

def freefall(zpp, fi_bpp, fi_spp, constraint=0):
    return (m_s + m_b)*zpp + m_b* l_s*fi_spp + m_b*l_g*fi_bpp + (m_s+m_b)* g + constraint

def eq1(zpp, fi_b, fi_bpp, fi_s, fi_spp, constraints):
    return (m_b*l_s)*zpp + (j_s+m_b*l_s*l_s)*fi_spp + (m_b*l_s*l_g)*fi_bpp - c_p*(fi_b-fi_s) + m_b*l_s*g + constraints[0] + constraints[1]

def eq2(zpp, fi_b, fi_bpp, fi_s, fi_spp, constraint=0):
    return m_b*l_g*zpp + m_b*l_s*l_g*fi_spp + (j_b+m_b*l_g*l_g)*fi_bpp - c_p*(fi_s-fi_b) + m_b* l_g* g + constraint

def woodpecker(t, y, yp, sw):
    if sw[0]:
        print('state 1')
        return stateI(t,y,yp)
    elif sw[1]:
        print('state 2')
        return stateII(t,y,yp)
    elif sw[2]:
        print('state 3')
        return stateIII(t,y,yp)
    elif sw[3]:
        print('state 4')
        return stateIV(t,y,yp)


####### FINAL TASK #########

global s
s = [0, 1, 0, 0]
global newS
newS = True
global prev_ev
prev_ev = [1,1]


def woodpecker_if(t, y, yp):
    global s
    global newS
    global prev_ev
    #print("s", s, "prev_ev", prev_ev)
    fi_bp=yp[2]
    phi_b=y[2]
    phi_s=y[1]
    lambda_1= y[6] #y[4] - yp[1]
    ns = s

######### STATE I ###########
    if s[0]:
        print("st 1")
        e_1 = h_s*phi_s + ( r_s -  r_0)
        e_2 = h_s*phi_s - ( r_s -  r_0)
        if (newS):
            newS = False
        else:
            if (prev_ev[0] * e_1) < 0 and fi_bp < 0:
                print("before mom", yp)
                y,yp = mom_conserv_if(y, yp)
                print("after mom", yp)
                ns=[0,1,0,0]
                newS == True
            elif (prev_ev[1] * e_2) < 0 and fi_bp > 0:
                print("before mom", yp)
                y,yp = mom_conserv_if(y, yp)
                print("after mom", yp)
                ns=[0,0,1,0]
                newS = True
        
        prev_ev = np.array([e_1, e_2])
        

######### STATE II ###########
    elif s[1]:
        # print("st 2")
        e_1 = lambda_1
        e_2 = 1
        print("e_1 * prev_ev in s2", e_1*prev_ev[0])
        if (newS):
            newS = False
        else:
            e = e_1*prev_ev[0]
            print(e)
            if e < 0:
                print("end up here?")
                ns=[1,0,0,0]
                newS = True
        
        prev_ev = np.array([e_1, e_2])


######### STATE III ###########
    elif s[2]:
        e_1 = lambda_1  
        e_2 = h_b*phi_b-(l_s+l_g-l_b-r_0)
        if (newS):
            print("st 3 on", t)
            newS = False
        else:
            print("e_1", e_1, "e_2", e_2)
            if (prev_ev[0] * e_1) < 0 and fi_bp < 0:
                print("change to one")
                ns=[1,0,0,0]
                newS = True
            elif (prev_ev[1] * e_2) < 0 and fi_bp > 0:
                y, yp = add_peck_if(y, yp)
            #else:
               # print("fi_bp", fi_bp)
        prev_ev = np.array([e_1, e_2])

    if newS:
        s = ns
        print("new state", s)
    if ns[0]:
        return stateI(t,y,yp)
    if ns[1]:
        return stateII(t,y,yp)
    if ns[2]:
        return stateIII(t,y,yp)
    if ns[3]:
        print("state 4??")
        return [0,0,0,0,0,0,0,0]




def mom_conserv_if(y, yp):
    print("In mom")
    zp = y[3]
    fi_sp = y[4]
    fi_bp = y[5]
    new_fi_bp = ((m_b*l_g)*zp + (m_b*l_s*l_g)*fi_sp) / (j_b+ m_b*l_g*l_g) + fi_bp
    y[3:6] = np.array([0, 0, new_fi_bp])
    yp[0:3] = np.array([0, 0, new_fi_bp])

    return y, yp

def add_peck_if(y, yp):
    global pecks
    print("peck")
    pecks = pecks + 1
    yp[2] = -yp[2]
    y[5] = -y[5]
    return y, yp























t0 = 0;
startsw = [1,0,0,0]
y0 = np.array([0.5, 0,0, -0, 0, 0.5,-1e-4,0])
yd0 =  np.array([-0, 0, 0.5,-g, 1e-12, 0, 0, 0])

w0 = -0.91
y0 = np.array([0.5, 0,0, -0, w0, w0,-1e-4,0])
yd0 =  np.array([-0, w0, w0,-g, 1e-12, 0, 0, 0])

y0 = np.array([4.83617428e-01, -3.00000000e-02, -2.16050178e-01, 1.67315232e-16, -5.39725367e-14, -1.31300925e+01, -7.20313572e-02, -6.20545138e-02])
yd0 = np.array([1.55140566e-17, -5.00453439e-15, -1.31302838e+01, 6.62087352e-13, -2.13577297e-10, 2.21484026e+02, -4.67637454e+00, -2.89824658e+00])
startsw = [0, 1, 0, 0]

problem = apr.Implicit_Problem(woodpecker_if, y0, yd0, t0)#, sw0=startsw)

# problem.state_events = state_events
# problem.handle_event = handle_event
problem.name = 'Woodpecker'

phipIndex = [4, 5]
lambdaIndex = [6, 7]
sim = aso.IDA(problem)
sim.rtol = 1e-6

sim.atol[phipIndex] = 1e8
sim.algvar[phipIndex] = 1
sim.atol[lambdaIndex] = 1e8
sim.algvar[lambdaIndex] = 1 

sim.suppress_alg = True
ncp = 500

tfinal = 0.1
t, y, yd = sim.simulate(tfinal, ncp)
y = y[:,[ 0,  ]]
plt.plot(t, y)
plt.legend(["z", "phi_s", "phi_b", "zp", "phip_s", "phip_b", "lambda_2", "lambda_2"], loc = 'lower left')
print("Number of pecks", pecks)
plt.ylabel('Höjd (m)')

plt.xlabel('Tid (s)')
plt.show()



