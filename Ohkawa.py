# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:23:27 2018

@author: rarbo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:38:10 2018

@author: rarbo
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mp
import time
import numpy.ma as ma
import Part1 as p1




"""Returns madnetic field at a given position"""
def mag(r, k=1):
    return np.array([0, 0, k])
    
"""Returns electric field at a given position"""
def ele(r, c=1, r_0=1):
    E = np.empty((r.size//3,3))
    theta = np.arctan2(r[:,1],r[:,0])
    rad = (r[:,0]**2 + r[:,1]**2)**(1/2)
    E[:,0] = c*np.cos(theta)*rad
    E[:,1] = c*np.sin(theta)*rad
    E[:,2] = 0
    return r_0*E
    #return np.array([r[:,0],r[:,1],0])
 
"""Implements a single discrete push, updates velocity and position"""
def push(v, x, dt, m, tau, v_th, t_0, k=1, c=1, r_0=1):
    B = mag(x, k=k)
    E = ele(x, c=c, r_0=r_0)
    
    k1 = lorentz(v, m, B, E, dt/2, tau, t_0)
    v1 = v + k1*dt/2
    x1 = x + v*dt/2
    
    k2 = lorentz(v1, m, mag(x1, k=k), ele(x1, c=c, r_0=r_0), dt/2, tau, t_0)
    v2 = v + k2*dt/2
    x2 = x + v1*dt/2
    
    k3 = lorentz(v2, m, mag(x2, k=k), ele(x2, c=c, r_0=r_0), dt, tau, t_0)
    v3 = v + k3*dt
    x3 = x + v2*dt
    
    k4 = lorentz(v3, m, mag(x3, k=k), ele(x3, c=c, r_0=r_0), dt, tau, t_0)

    sigma = v_th*((2 / (3* t_0 *tau * dt)))**(1/2)
    
    return x + dt*(v + 2*v1 + 2*v2 + v3)/6, v + dt*((k1 + 2*k2 + 2*k3 + k4)/6 + (t_0 / v_0)*randaccel(sigma, v.size // 3))
    

def lorentz(v, m, B, E, dt, tau, t_0):
    """Returns lorentz acceleration for a given set of parameters"""
    return E + np.cross(v, B) - (t_0/tau)*v# + rand


def randaccel(sigma, num):
    return sigma*np.random.randn(num, 3)
"""
def tau(m1, m2, num2, V, T, Z):
    Returns tau for collision of two species
    top = (6 * np.sqrt(2) * np.pi**(3/2) * epsilon_0**2 * m1 * T**(3/2))
    bottom = (11 * np.sqrt(m2) * Z**2 * e**4 * num2/V)
    return top/bottom
"""
    
def radius(x, y):
    """ Returns radial position in the x,y plane """
    return (x**2 + y**2)**(1/2)

"""Returns kinetic energy"""
def kinetic(v, m):
    return (1/2)*m*(np.linalg.norm(v))**2


def architime(num, r, dt, m, q, tau, rad, h, U, k=1, c=1):
    """ Computes exit time and radial velocity for a set of particles moving
    through an Ohkawa filter"""
    ext = 0
    dele = []
    sepr = 0
    seph = 0
    iteration = 0
    t_ext = []
    
    v_th = (3 * U / m)**(1/2)
    v = np.array(v_th*np.random.randn(num, 3))/v_0
    
    t_0 = m / (q * B_0)

    #storex = [0]
    #storey = [0]
    #storez = [0]
    
    r_0 = (m/q)*(E_0/B_0**2)
    ratio = .999
    vrad = []
    #out = []
    
    #print("Number of particles =", v.size//3)
    
    while (ext / num) < ratio:
        iteration +=1
        r, v = push(v, r, dt, m, tau, v_th, t_0, k=k, c=c, r_0=r_0)
        dele = np.where(radius(r[:,0], r[:,1])*r_0 >= rad)[0]
        if dele.size != 0:
            for i in range (len(dele)):
                t_ext.append(iteration)
            vrad.append(radius(v[dele, 0], v[dele, 1]))
            ext += len(dele)
            sepr += len(dele)
            #out.append(r[dele])
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        dele = np.where(r[:,2]*r_0 > h)[0]
        if dele.size != 0:
            ext += len(dele)
            seph += len(dele)
            #out.append(r[dele])
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        dele = np.where(r[:,2]*r_0 < -h)[0]
        if dele.size != 0:
            ext += len(dele)
            seph += len(dele)
            #out.append(r[dele])
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        if iteration == 150000: 
            return t_ext, vrad
        
        #print("Exited particles =", exit)
        #print("Iteration number ", iteration)
        #iteration += 1
        #print("Number of iterations = ", iteration)
        #r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
    return t_ext, vrad

def archimatrix(num, r, dt, m, q, tau, rad, h, U, k=1, c=1):
    """ Runs a set of particles through the Ohkawa filter until all particles but
    num*(1-ratio) have exited"""
    ext = 0
    dele = []
    sepr = 0
    seph = 0
    iteration = 0
    
    v_th = (3 * U / m)**(1/2)
    v = np.array(v_th*np.random.randn(num, 3))/v_0
    
    t_0 = m / (q * B_0)
    
    #storex = [0]
    #storey = [0]
    #storez = [0]
    
    r_0 = (m/q)*(E_0/B_0**2)
    ratio = .995
    #out = []
    iteh = []
    iterad  = []
    textrad = []
    texth = []
    itetot = []
    exttot = []
    #print("Number of particles =", v.size//3)
    while (ext / num) < ratio:
        iteration +=1
        if iteration % 20000 == 0:
            print("Iteration number ", iteration)
            print("Number exited = ", ext)
        r, v = push(v, r, dt, m, tau, v_th, t_0, k=k, c=c, r_0=r_0)
        dele = np.where(radius(r[:,0], r[:,1])*r_0 >= rad)[0]
        if dele.size != 0:
            ext += len(dele)
            sepr += len(dele)
            for i in range(len(dele)):
                iterad.append(iteration)
                #textrad.append(ext)
                # itetot.append(iteration)
                #exttot.append(ext)
            #out.append(r[dele])
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        dele = np.where(r[:,2]*r_0 > h)[0]
        if dele.size != 0:
            ext += len(dele)
            seph += len(dele)
            for i in range(len(dele)):
                iteh.append(iteration)
                #texth.append(ext)
                #itetot.append(iteration)
                #exttot.append(ext)
            #out.append(r[dele])
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        dele = np.where(r[:,2]*r_0 < -h)[0]
        if dele.size != 0:
            ext += len(dele)
            seph += len(dele)
            
            for i in range(len(dele)):
                iteh.append(iteration)
                #texth.append(ext)
                #itetot.append(iteration)
                #exttot.append(ext)
            #out.append(r[dele])
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        if iteration == 500000: 
            return sepr, seph
        
        #print("Exited particles =", exit)
        #print("Iteration number ", iteration)
        #iteration += 1
        #print("Number of iterations = ", iteration)
        #r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
    """
    print("Iterations = ", iteration)
    print("Radial = ", sepr)
    print("Axial = ", seph)
    """
    #ThreedPlot(storex, storey, storez)
    
    if seph > sepr:
        return sepr, num - sepr, iteh, iterad, textrad, texth, itetot, exttot
    return sepr/ratio, seph/ratio, iteh, iterad, textrad, texth, itetot, exttot#, iteration, np.concatenate(out)

def valfunc(x):
    """ Returns the value function evaluated at x"""
    if x < 0.00001:
        x = 0.00001
    return (1 - 2*x)*np.log((1-x)/x)

def ohkawa(n1, n2, r1, r2, dt, m1, m2, tau1, tau2, q1, q2, rad, h, U, k=1, c=1):
    """ Runs two instances of archimatrix to return the separated particles.
    Returns ((seprl, sephl), (seprh, sephh))"""
    return archimatrix(n1, r1, dt, m1, q1, tau1, rad, h, U, k=k, c=c), archimatrix(n2, r2, dt, m2, q2, tau2, rad, h, U, k=k, c=c)

def seppower(n1, n2, r1, r2, dt, m1, m2, tau1, tau2, q1, q2, rad, h, U, k=1, c=1):
    """ Computes the separative power by first separating the two elements"""
    
    inmass = n1 + n2
    inconc = n1 / inmass
    
    m_c = (e*B_0**2) / (4*E_0)
    d_m_h = m2 - m_c
    w_E = -E_0/B_0
    v_t_h = (2 * U / m2)**(1/2)
    v_t_l = (2 * U / m1)**(1/2)
    
    L_star = h / np.log(-4 * rad * w_E * (m_c * d_m_h)**(1/2)/(m2 * v_t_h))
    
    F = feed(v_t_l, inconc, m2, m_c, d_m_h, w_E, L_star)
    
    sep = ohkawa(n1, n2, r1, r2, dt, m1, m2, tau1, tau2, q1, q2, rad, h, U, k=1, c=1)
    
    print("Radial Al = ", sep[0][0])
    print("Radial Sr = ", sep[1][0])
    print("Axial Al = ", sep[0][1])
    print("Axial Sr = ", sep[1][1])
    
    
    
    wamass = sep[0][0] + sep[1][0]
    
    if wamass == 0:
        return 0
    waconc = sep[0][0] / wamass
    
    promass = sep[0][1] + sep[1][1]
    if promass == 0:
        return 0
    proconc = sep[0][1] / promass
    
    if waconc == 0:
        waconc = 0.00000001
    if waconc > .99999999:
        waconc = .99999999
    if proconc == 0:
        proconc = 0.00000001
    if proconc > .99999999:
        proconc = .99999999
    """
    print("Mass of Input, Waste, Product", inmass, wamass, promass)
    print("Concentration of Al in Input, Wast, Product", inconc, waconc, proconc)
    """
    
    flp = proconc*promass/(inconc*inmass)
    flw = 1 - flp
    if flw == 0:
        flw = 0.00000001
    fhp = (1 - proconc)*promass / ((1 - inconc)*inmass)
    fhw = 1 - fhp
    if fhw == 0 :
        fhw = 0.00000001
    
    """
    ide = sepfactorideal(sep[0][1], sep[1][1], sep[0][0], sep[1][0])
    ex = sepfactorexp(sep[0][1], sep[1][1], n1, n2)
    
    Vx = valfunc(proconc)
    Vy = valfunc(waconc)
    Vz = valfunc(inconc)
    """
    

    #print("promass = ", promass)
    #print("promass/inmass = ", promass/inmass)
    print("flw =", flw)
    print("fhp =", fhp)
    
    psv = psuedovolume(v_t_l, m1, m_c, w_E, v_t_h, h)
    
    
    #swork = promass*Vx + wamass*Vy - inmass*Vz
    
    #inflow = flow(num1 + num2, rad, inconc, m1, m2, c*E_0, k*B_0, L, T)
    #proflow = 
    #waflow
    
    return (flp*inconc - fhp*(1-inconc))*np.log(flp/fhp)+ (flw*inconc - fhw*(1 - inconc))*np.log(flw/fhw), F*psv
  
""" Motion inside a particle filter for a single particle """
def motion(r, v, dt, m, tau, T, zmax, xmax):
    
    storex = [r[0][0]]
    storey = [r[0][1]]
    storez = [r[0][2]]
    
    r_0 = (m/q)*(E_0/B_0**2)
    
    while 1:
        r, v = push(v, r, dt, m, tau, T)
        storex.append(r[0][0])
        storey.append(r[0][1])
        storez.append(r[0][2])
        if r[0][0]*r_0 <= 0 or r[0][0]*r_0 >= xmax:
            break
        if r[0][2]*r_0 <= 0 or r[0][2]*r_0 >= zmax:
            break
        

    mp.pyplot.plot(storex, storey)
    ThreedPlot(storex, storey, storez)
    return storex, storey, storez
   
def ThreedPlot(storex, storey, storez):
    """ Draws a 3d plot of a particle's tragectory in space"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(storex, storey, zs=storez)
    
def randomv(sigma, num):
    """ Initializes num particles of random velocity, sampled from a normal
    distribution with mean zero and std sigma (v_thermal)"""
    return sigma*np.random.randn(num, 3)
        

def rotate(delta_m, V, a, B_bar, E_bar, T):
    """ Gives rotational kinetic energy """
    return (delta_m*V**2*E_bar**2)/(a**2*B_bar**2*.16**2*2*T)

def sepfactorexp(n1p, n2p, n10, n20):
    """ Gives an experiemental form of the seperative factor """
    c_p = n1p / (n1p + n2p)
    c_0 = n10 / (n10 + n20)
    if c_0 < 0.000001:
        c_0 = 0.000001
    if c_p > 0.999999:
        c_p = 0.999999
    return (c_p*(1 - c_0)) / (c_0*(1 - c_p))

def sepfactorideal(n1r, n2r, n1a, n2a):
    """ Gives an ideal form of the seperative factor """
    if n2r == 0:
        n2r = 1
        
    if n1a == 0:
        n1a = 1
    
    return (n1r * n2a) / (n1a * n2r)

def omega(m, B, E):
    """ Computes the w_E value of a particle in the Ohkawa filter"""
    e = 1.602e-19
    return ((e*B) / (2*m))*(1 - (1 - 4 * E * m / (B**2 * e))**(1/2))

def logalpha(m1, m2, a, B, E, U):
    """ Computes log(alpha) with w = w"""
    return (m2 - m1)*omega(m1, B, E)**2*a**2 / (2 * U)

def logalpha2(m1, m2, a, B, E, U):
    """ Computes log(alpha) with w = w_E. This is the correct log(alpha)"""
    return (m2 - m1) * (E/B)**2 * a**2 / (2 * U)

def tex(a, m_h, E, B, U):
    omega_E = -E/B
    m_c = e*B**2 / (4*E)
    delta_m_h = m_h - m_c
    v_r = (3 * U / m_h)**(1/2)
    return np.log(-4*a*omega_E * (m_c*delta_m_h)**(1/2) / (m_h *v_r)) / (-2*omega_E*(m_c*delta_m_h)**(1/2) / m_h)
    

def flow(n, a, z, m_l, m_h, E, B, L, T):
    e = 1.602e-19
    A = np.pi * a**2
    n_0 = n / (A*L)
    w_E = -E / B
    m_c = (e * B**2) / (4 * E)
    deltm_h = m_h - m_c
    v_tl = ((3 * k_b * T) / m_l)**(1/2)
    v_th = ((3 * k_b * T) / m_h)**(1/2)
    L_star = L / np.log( -4 * a * w_E * (m_c * deltm_h)**(1/2) / (m_h * v_th))
    
    top = n_0 * A * v_tl
    bottom = (1 - z) * v_tl * m_h / (-2 * w_E * L_star * (m_c * deltm_h)**(1/2)) + z/(2*np.pi)**(1/2)

    return top/bottom

def spf(z, m_h, m_l, E, B, a, U):
    m_c = (e*B**2) / (4*E)
    d_m_h = m_h - m_c
    w_E = -E/B
    v_t_h = (3 * U / m_h)**(1/2)
    d_m = m_h - m_l
    d_m_l = m_c - m_l
    T = U
    v_t_l = (3 * U / m_l)**(1/2)
    L = 2*a
    
    L_star = L / np.log(-4 * a * w_E * (m_c * d_m_h)**(1/2)/(m_h * v_t_h))
   
    return (z*((m_c * d_m_h * w_E**2 * L_star**2) / (m_h**2 * v_t_h**2) -
               np.log(-v_t_h / (L_star * w_E * np.pi**(1/2) * (m_c*d_m)**(1/2)
               / m_h))) + (1 - z)*(( m_c * d_m_l * 2 * w_E**2 * a**2) /
               (m_l * T) - np.log(4 * (2 * w_E**2 * a**2 * m_c * d_m)**(3/2)
               / (3 * np.pi**(1/2) * v_t_l**3 * m_l**3))))

def feed(v_t_l, z, m_h, m_c, d_m_h, w_E, L_star):
    """ Gives the feed factor (with some factors cancelled out) from Fetterman paper"""
    return v_t_l / ( (((1-z)*v_t_l*m_h)/(-2*w_E*L_star*(m_c*d_m_h)**(1/2))) + z/(2*np.pi)**(1/2))

def psuedovolume(v_t_l, m_l, m_c, w_E, v_t_h, L):
    """ Gives 1/normalized V factor from the Fetterman paper"""
    return 10*(-v_t_l * m_l) / (4 * m_c * w_E * v_t_h * L * (2*np.pi)**(1/2))
               
def spv(z, m_h, m_l, E, B, a, U):
    """Computes equation number 25 from the Fetterman paper"""
    m_c = (e*B**2) / (4*E)
    d_m_h = m_h - m_c
    w_E = -E/B
    v_t_h = (2 * U / m_h)**(1/2)
    d_m = m_h - m_l
    d_m_l = m_c - m_l
    T = U
    v_t_l = (2 * U / m_l)**(1/2)
    L = 2*a
    
    L_star = L / np.log(-4 * a * w_E * (m_c * d_m_h)**(1/2)/(m_h * v_t_h))
    
    F = feed(v_t_l, z, m_h, m_c, d_m_h, w_E, L_star)
    
    psv = psuedovolume(v_t_l, m_l, m_c, w_E, v_t_h, L)
    
    
    return (F*psv*(z*((m_c * d_m_h * w_E**2 * L_star**2) / (m_h**2 * v_t_h**2) -
               np.log(-v_t_h / (L_star * w_E * np.pi**(1/2) * (m_c*d_m)**(1/2)
               / m_h))) + (1 - z)*(( m_c * d_m_l * 2 * w_E**2 * a**2) /
               (m_l * T) - np.log(4 * (2 * w_E**2 * a**2 * m_c * d_m)**(3/2)
               / (3 * np.pi**(1/2) * v_t_l**3 * m_l**3)))))
               
def spa(z, m_h, m_l, E, B, a, U):
    """Computes equation number 32 from the Fetterman paper with given z"""
    m_c = (e*B**2) / (4*E)
    d_m_h = m_h - m_c
    w_E = -E/B
    v_t_h = (2 * U / m_h)**(1/2)
    d_m_l = m_c - m_l
    T = U
    v_t_l = (2 * U / m_l)**(1/2)
    L = 2*a
    
    L_star = L / np.log(-4 * a * w_E * (m_c * d_m_h)**(1/2)/(m_h * v_t_h))
    
    F = feed(v_t_l, z, m_h, m_c, d_m_h, w_E, L_star)
    
    psv = psuedovolume(v_t_l, m_l, m_c, w_E, v_t_h, L)
    
    
    #(F*psv*(z*((m_c * d_m_h * w_E**2 * L_star**2) / (m_h**2 * v_t_h**2)
     #          ) + (1 - z)*(( m_c * d_m_l * 2 * w_E**2 * a**2) / (m_l * T))) - 
    
    return (F*psv*(m_c/T)*(z*d_m_h*w_E**2*L_star**2 / m_h + (1 - z)*d_m_l*2*w_E**2*a**2 / m_l))           
               
def eq32light(E, B, a, T):
    """Computes equation number 32 from the Fetterman paper with z = 1 """
    m_c = e*B**2 / (4*E)
    d_m_h = (1/2) * m_c
    L = 2*a
    m_h = (3/2) * m_c
    v_t_h = (2 * T / m_h)**(1/2)
    w_E = -E/B
    
    return (10 * -d_m_h * w_E * L) / (2 * m_h * v_t_h) * (1 / np.log(-4 * a * w_E * (m_c * d_m_h)**(1/2) / (m_h * v_t_h))**2)
    
def eq32heavy(E, B, a, T):
    """Computes equation number 32 from the Fetterman paper with z = 0 """
    m_c = e*B**2 / (4*E)
    d_m_l = (1/3) * m_c
    w_E = -E/B
    d_m_h = d_m_l
    m_h = (4/3) * m_c
    m_l = (2/3) * m_c
    v_t_h = (2* T / m_h)**(1/2)
    
    return 10* (2 * d_m_l * w_E**2 * a**2 * (m_c * d_m_h)**(1/2)) / (T * (2 * np.pi * m_h * m_l)**(1/2) * np.log(-4 * a * w_E * (m_c * d_m_h)**(1/2) / (m_h * v_t_h)))

def seppower2(n1, n2, radl, radh, axl, axh, m1, m2, rad, h, U):
    """Computes separative power with a prior separation completed """
    inmass = n1 + n2
    inconc = n1 / inmass
    rl  = n1*radl / (axl+radl)
    rh  = n2*radh / (axh+radh)
    al = n1 - rl
    ah = n2 - rh
    
    wamass = rl + rh
    
    if wamass == 0:
        return 0
    waconc = rl / wamass
    
    promass = al + ah
    if promass == 0:
        return 0
    proconc = al / promass
    
    if waconc == 0:
        waconc = 0.00000001
    if waconc > .99999999:
        waconc = .99999999
    if proconc == 0:
        proconc = 0.00000001
    if proconc > .99999999:
        proconc = .99999999

    flp = proconc*promass/(inconc*inmass)
    flw = 1 - flp
    if flw == 0:
        flw = 0.00000001
    fhp = (1 - proconc)*promass / ((1 - inconc)*inmass)
    fhw = 1 - fhp
    if fhw == 0 :
        fhw = 0.00000001
    
    return (flp*inconc - fhp*(1-inconc))*np.log(flp/fhp)+ (flw*inconc - fhw*(1 - inconc))*np.log(flw/fhw)

def archierror(num, r, dt, m, q, tau, rad, h, U, ratio, k=1, c=1):
    """ Moves a set of particles through an ohkawa filter, does not compute
    weighted results """
    
    ext = 0
    dele = []
    sepr = 0
    seph = 0
    iteration = 0
    
    v_th = (3 * U / m)**(1/2)
    v = np.array(v_th*np.random.randn(num, 3))/v_0
    
    t_0 = m / (q * B_0) 
    r_0 = (m/q)*(E_0/B_0**2)

    while (ext / num) < ratio:
        iteration +=1
        if iteration % 20000 == 0:
            print("Iteration number ", iteration)
            print("Number exited = ", ext)
        r, v = push(v, r, dt, m, tau, v_th, t_0, k=k, c=c, r_0=r_0)
        dele = np.where(radius(r[:,0], r[:,1])*r_0 >= rad)[0]
        if dele.size != 0:
            ext += len(dele)
            sepr += len(dele)
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        dele = np.where(r[:,2]*r_0 > h)[0]
        if dele.size != 0:
            ext += len(dele)
            seph += len(dele)
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        dele = np.where(r[:,2]*r_0 < -h)[0]
        if dele.size != 0:
            ext += len(dele)
            seph += len(dele)
            r, v = np.delete(r, dele, 0), np.delete(v, dele, 0)
        if iteration == 500000: 
            return sepr, seph

    return sepr, seph, num

def testkinetic(N, n):
    
    krk = []
    kdt = []
    m = 1
    v = np.array([1, 0, 0])
    standard = p1.kinetic(v, m)
    print("standard is ", standard)
    
    for j in (N/n):
        x = np.array([1, 0, 0])
        v = np.array([1, 0, 0])
        for i in n:
            x, v = p1.push(v, x, j, 1, m)
            #push(v, x, dt, m, tau, 0, 0, k=1, c=0, r_0=.13654)
        print("current is ", p1.kinetic(v, m))
        krk.append(p1.kinetic(v, m))
        
   
    for j in (N/n):
        x = np.array([1, 0, 0])
        v = np.array([1, 0, 0]) 
        for i in n:
            x, v = p1.pushOld(v, x, j, 1, m)
        print("current is ", p1.kinetic(v, m))
        kdt.append(p1.kinetic(v, m))
    
    """
N = 10
n = np.arange(100, 1000, step=5)

#st1, st2 = testkinetic(N, n)

plt.plot(N/n, 100*(0.5-np.array(st1)/0.5))
#plt.title('Absolute Runge Kutta Error')
#plt.xlabel('dt')
#plt.ylabel('Percent Error (%)')
"""
    #plt.plot(N/n, (np.array(kdt)-standard)/standard, N/n, (standard-np.array(krk))/standard)
    #print(krk, kdt)
    return krk, kdt

def ratioerror(light, heavy, m1, m2, rad, h, U, n1, n2):
    """ Gives the maximum and minimum values of separative power for a predetermined cut-off ratio"""
    maxmin = []
    maxmin.append(seppower2(n1, n2, light[2]-light[1], heavy[2]-heavy[1], light[1], heavy[1], m1, m2, rad, h, U))
    maxmin.append(seppower2(n1, n2, light[2]-light[1], heavy[0], light[1], heavy[2]-heavy[0], m1, m2, rad, h, U))
    maxmin.append(seppower2(n1, n2, light[0], heavy[2]-heavy[1], light[2]-light[0], heavy[1], m1, m2, rad, h, U))
    maxmin.append(seppower2(n1, n2, light[0], heavy[0], light[2]-light[0], heavy[2]-heavy[0], m1, m2, rad, h, U))
    
    return max(maxmin), min(maxmin)


def ratiotest(n1, n2, dt, m1, m2, q1, q2, tau1, tau2, rad, h, U):
    #Test error dependence on cutoff ratioa
    
    maxes = []
    mins = []
    x1 = np.zeros((30000, 3))
    x = np.zeros((30000,3))
    for rat in ratio:
        light = archierror(30000, x, dt, m1, q1, tau1, rad, h, U, rat)
        heavy = archierror(30000, x1, dt, m2, q2, tau2, rad, h, U, rat)
    
        minmax = ratioerror(light, heavy, m, m2, rad, h, U, n1, n2)
        maxes.append(minmax[0])
        mins.append(minmax[1])
        
    
    plt.plot(ratio, maxes, ratio, mins)

epsilon_0 = 8.854e-12
k_b = 1.3806e-23
Z = 90
dt = .01
num1 = 1
num2 = 1
U = 1e-17
m = 4.48e-26
m2 = 3*m
tau = 0.001048689138576779
tau2 = 100000000000

k_b = 1.3806e-23
e = 1.602e-19

q1 = e
q2 = e

rad = .4
h = 2*rad
V = 1000
B_0 = .16
E_0 = V*2/(rad**2)
v_0 = E_0 / B_0
t_0 = (4.48e-26)/(1.602e-19 * B_0)
#print(v_0)
n1 = 50000
n2 = 1500
#v1 = randomv((3 * T)**(1/2), 1000) / v_0
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
#print(T * t_0**2)
#print(v1)
#alph1 = []
#alph2 = []
volt = []
al1 = []
#al2 = []
t_ext_tbl = []
lis = []
vs = []
#se1 = []
#se2 = []
sv1 = []
sv2 = []
#maxes3 = []
#mins3 = []

ratio = [0.8, 0.9, 0.93, 0.95, 0.98, .982, .985, .9875, .99, .995, .999]
#ohk3 = ohkawa(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U)





"""
print(logalpha2(m, m2, rad, B_0, E_0, U))
"""



"""
for i in np.linspace(1, 5, num=17):
    V = i*1000
    E_0 = V*2/(rad**2)
    v_0 = E_0 / B_0
    tx = tex(rad, m2, E_0, B_0, U)
    volt.append(tx)
    print(tx)
    hold  = architime(n2, x1, dt, m2, q2, tau2, rad, h, U)
    t_ext = dt*(m2 / (e * B_0)) * np.array(hold[0])
    print(np.median(t_ext))
    t_ext_tbl.append(t_ext)
    lis.append(np.median(t_ext))
    vs.append(v_0*np.concatenate(hold[1]))
"""    

# Code for comparing to Fetterman graph


y1 = []
y2 = []
y3 = []
#y4 = []
xi = []
psv1 = []
psv2 = []
psv3 = []

#y5 = []
y6 = []
y7 = []


(10 / (4*(1-(1/2)**(1/2))))

m = (1/2) * (e*B_0**2) / (4*E_0)
m2 = 3*m

#hold = ohkawa(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U)

#store = ohkawa(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U)
#print(store)

"""
MCMF Light
for x in np.linspace(1, 10, num=100):
    xi.append(x)
    y.append((3**(1/2)) / 4 * x**2 * np.exp(-x / 2))
    """

#y1.append(1)
#y1.pop()
psv4 = 0
E_0 = V*2/(rad**2)
v_0 = E_0 / B_0
stos = []
#seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1)
# original linspace: V from 350 to 2000, num =10

inmass = n1 + n2
inconc = n1 / inmass
for i in np.linspace(320, 2500, num=12):
    V = i
    E_0 = V*2/(rad**2)
    v_0 = E_0 / B_0
    
    m = (1/2) * (e*B_0**2) / (4*E_0)
    m2 = 3*m

    x1 = np.zeros((n2, 3))
    x = np.zeros((n1,3))
    al1.append(logalpha2(m, m2, rad, B_0, E_0, U))
    sto = seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1)
    y1.append(eq32light(E_0, B_0, rad, 3/2 *U))#/#sto[1])
    y2.append(spa(inconc, m2, m, E_0, B_0, rad, 3/2 *U))#/sto[1])
    y3.append(spv(inconc, m2, m, E_0, B_0, rad, 3/2 *U))#/sto[1])
    y6.append(sto[0])
    y7.append(sto[1])
    stos.append(sto)
    psv3.append(sto[1])
    #al2.append(logalpha2(m, m2, rad, B_0, E_0, U))
    m_c = (e*B_0**2) / (4*E_0)
    d_m_h = m2 - m_c
    w_E = -E_0/B_0
    v_t_h = (2 * U / m2)**(1/2)
    v_t_l = (2 * U / m)**(1/2)
    
    L_star = h / np.log(-4 * rad * w_E * (m_c * d_m_h)**(1/2)/(m2 * v_t_h))
    
    F = feed(v_t_l, inconc, m2, m_c, d_m_h, w_E, L_star)

    #psv4 = F*psuedovolume(v_t_l, m, m_c, w_E, v_t_h, h)
    psv1.append(F*psuedovolume(v_t_l, m, m_c, w_E, v_t_h, h))
    #psv2.append(F*psuedovolume(v_t_l, m, m_c, w_E, v_t_h, h))



"""
for x in np.linspace(1,10, num = 100):
    xi.append(x)
    y4.append((10*x*(3/(8*np.pi))**(1/2))/np.log((2*(3*x/2)**(1/2))))



for x in np.linspace(1,10, num=100):
    xi.append(x)
    #y4.append(((10 / (4*(1-(1/2)**(1/2))))*(x/6)**(1/2)) / np.log10((1/(1-(1/2)**(1/2)))*(2*x/3)**(1/2))**2)
    y4.append((10*(x/6)**(1/2)) / np.log(4*(x/3)**(1/2))**2)
"""
"""    
plt.plot(al1, y1, al1, y2, xi, y4, al1, y3)
plt.axis([1,10,1,20])
plt.yscale('symlog')
"""

"""
for i in np.linspace(250, 1500, num=40):
    x1 = np.zeros((n2, 3))
    x = np.zeros((n1,3))
    V = i
    E_0 = V*2/(rad**2)
    v_0 = E_0 / B_0
    
    m = (.5) * (e*B_0**2) / (4*E_0)
    m2 = 3*m
    
    al1.append(logalpha2(m, m2, rad, B_0, E_0, U))
    #se1.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau2, q1, q2, rad, h, U))
    sv1.append(spv(inconc, m2, m, E_0, B_0, rad, U))
"""
"""
V = 1365
E_0 = V*2/(rad**2)
for i in np.linspace(0.001, 10, num=40):
    x1 = np.zeros((n2, 3))
    x = np.zeros((n1,3))
    x1 = np.zeros((n2, 3))
    x = np.zeros((n1,3))
    B_0 = i
    v_0 = E_0 / B_0
    m = .5 * (e*B_0**2) / (4*E_0)
    print(m)
    m2 = 3*m
    al2.append(logalpha2(m, m2, rad, B_0, E_0, U))
    #se2.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau2, q1, q2, rad, h, U))
    sv2.append(spv(inconc, m2, m, E_0, B_0, rad, U))
""" 

#motion(x, v, dt, m, tau, T / (t_0**2), h, rad)
#print(archimatrix(v, x, dt, m, q1, tau, T / (t_0**2), rad, h, k=1, c=1))
#print(seppower(n1, n2, x, x1, dt, m, m2, tau, tau2, q1, q2, rad, h, U))

"""
for i in range(100):
    x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
    alph1 = []
    alph2 = []
    V = (i+1)*15
    volt.append(V)
    for j in range(100):
        B_0 = (j+1)*.16
        E_0 = V*2/(rad**2)
        v_0 = E_0 / B_0
        alph1.append(logalpha(m, m2, rad, B_0, E_0, U))
        alph2.append(logalpha2(m, m2, rad, B_0, E_0, U))
    alph12.append(alph1)
    alph22.append(alph2)
    #sepp.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau2, q1, q2, rad, h, U))

"""

"""

inmass = n1 + n2
inconc = (n1) / inmass
n235000 = []
n240000 = []


for i in range(4):
    #ohk = ohkawa(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U
    #power.append(seppower2(n1, n2, ohk[0][1], ohk[1][0], axl, axh, m1, m2, rad, h, U))
    n240000.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
print(np.var(n240000))
print(np.mean(n240000))

n1 = 35000
n2 = 35000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(4): 
    n235000.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
print(np.var(n235000))
print(np.mean(n235000))
"""
""""
n1 = 30000
n2 = 30000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(5): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 25000
n2 = 25000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(6): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 20000
n2 = 20000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(7): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 17500
n2 = 17500
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(7): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 17500
n2 = 17500
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(7): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 15000
n2 = 15000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(8): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 12000
n2 = 12000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(9): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 10000
n2 = 10000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(10): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 8000
n2 = 8000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(12): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

n1 = 5000
n2 = 5000
x1 = np.zeros((n2, 3))
x = np.zeros((n1,3))
inmass = n1 + n2
inconc = (n1) / inmass

for i in range(12): 
    power.append(seppower(n1, n2, x, x1, dt, m, m2, tau, tau, q1, q2, rad, h, U, k=1, c=1))
pows.append(np.var(power))
avg.append(np.mean(power))

#varn, meann, and ns hold the info for graph of variance as function of n
"""


    






