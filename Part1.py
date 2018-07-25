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



"""Returns madnetic field at a given position"""
def mag(position):
    return np.array([0, 0, 1])
    
"""Returns electric field at a given position"""
def ele(position):
    return np.array([0, 0, 0])
 
"""Implements a single discrete push, updates velocity and position"""
def push(v, x, dt, q, m):
    B = mag(x)
    E = ele(x)
    
    k1 = lorentz(v, q, m, B, E)
    v1 = v + k1*dt/2
    x1 = x + v*dt/2
    
    k2 = lorentz(v1, q, m, mag(x1), ele(x1))
    v2 = v + k2*dt/2
    x2 = x + v1*dt/2
    
    k3 = lorentz(v2, q, m, mag(x2), ele(x2))
    v3 = v + k3*dt
    x3 = x + v2*dt
    
    k4 = lorentz(v3, q, m, mag(x3), ele(x3))
    
    return x + dt*(v + 2*v1 + 2*v2 + v3)/6, v + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
"""Returns lorentz acceleration for a given set of parameters"""
def lorentz(v, q, m, B, E):
    return q*(E + np.cross(v, B))/m

def pushOld(velocity, position, dt, q, m):
    B = mag(position)
    E = ele(position)
    velocity = vel(velocity, q, m, B, E, dt)
    position = pos(position, velocity, dt)
    return position, velocity

"""Updates velocity based on second order error"""
def vel(v, q, m, B, E, dt):
    return v + q*(E + dt*np.cross(v, B))/m

"""Pushes the position"""
def pos(position, v, dt):
    return position + dt*v
    
"""Returns cyclotron frequency"""
def cyclotron(q, B, m):
    return abs(q)*(np.linalg.norm(B)**2)/m

"""Returns Larmor Radius"""
def larmor(v, q, B, m):
    return (np.linalg.norm(v - np.dot(v, B/np.linalg.norm(B)))**2)/cyclotron(q, B, m)

"""Returns kinetic energy"""
def kinetic(v, m):
    return (1/2)*m*(np.linalg.norm(v))**2

def ThreedPlot(storex, storey, storez):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(storex, storey, zs=storez)

""" Motion inside a particle filter for a single particle """
def motion(r, v, dt, q, m, ymax, xmax):
    
    storex = [r[0]]
    storey = [r[1]]
    storez = [r[2]]
    storev = [v]
    
    while true:
        r, v = push(v, r, dt, q, m)
        storex.append(r[0])
        storey.append(r[1])
        storez.append(r[2])
        if r[0] <= 0 or r[0] >= xmax:
            break
        if r[1] <= 0 or r[1] >= ymax:
            break
        
    return storex, storey, storez
   # mp.pyplot.plot(storex, storey)
    # ThreedPlot(storex, storey, storez)


""" Returns the separation percentage of two particle types """
def seperateloop(start, top, right, q1, q2, m1, m2, num1, num2, sigma):
    
    sep1x = 0
    sep1y = 0
    sep2x = 0
    sep2y = 0
    
    for i in range(num1):
        v = np.array(sigma*np.random.randn(3) + (top + right) / 2)
        r = start
        while 1:
            r, v = push(v, r, dt, q1, m1)
            if out(r[0], right):
                sep1x += 1
                break
            if out(r[1], top):
                sep1y += 1
                break
    
    for i in range(num2):
        r = start
        v = np.array(sigma*np.random.randn(3) + (top + right) / 2)
        push(v, r, dt, q2, m2)
        while 1:
            r, v = push(v, r, dt, q2, m2)
            if out(r[0], right):
                sep2x += 1
                break
            if out(r[1], top):
                sep2y += 1
                break
            
    return sep1x, sep1y, sep2x, sep2y
   # return (abs(sep1x - sep2x) + abs(sep1y - sep2y)) / (num1 + num2)

def out(pos, bound):
    return pos <= 0 or pos >= bound

def init(start, num):
    r = np.empty((num1,3))
    r[:,0] = start[0]
    r[:,1] = start[1]
    r[:,2] = start[2]
    return r

def randomv(sigma, num):
    v = np.empty((num, 3))
    for i in range(num):
        v[i] = np.array(sigma*np.random.randn(3) + (top + right) / 2)
    return v 

def seperatemat(start, top, right, q1, q2, m1, m2, num1, num2, sigma, dt):
    sep1x = 0
    sep1y = 0
    sep2x = 0
    sep2y = 0
    
    r = init(start, num1)   
    v = randomv(sigma, num1)
    exit = 0
    
    while exit < num1:
        delete = []
        r, v = push(v, r, dt, q1, m1)
        for i in range(r.shape[0]):
            if out(r[i,0], right):
                delete.append(i)
                exit += 1
                sep1x +=1
            if out(r[i,1], top):
                delete.append(i)
                exit += 1
                sep1y += 1
        r, v = np.delete(r, delete, 0), np.delete(v, delete, 0)
        
        
    r = init(start, num2)
    v = randomv(sigma, num2)
    exit = 0
    
    while exit < num2:
        delete = []
        r, v = push(v, r, dt, q2, m2)
        for i in range(r.shape[0]):
            if out(r[i,0], right):
                delete.append(i)
                exit += 1
                sep2x +=1
            if out(r[i,1], top):
                delete.append(i)
                exit += 1
                sep2y += 1
        r, v = np.delete(r, delete, 0), np.delete(v, delete, 0)
        
    return sep1x, sep1y, sep2x, sep2y

if __name__ == "__main__":    
    
    true = 1
    
    """ Specify parameters of separator """
    ymax = 5
    xmax = 5
    q1 = 1
    m1 = 352
    q2 = 2
    m2 = 200
    dt = .002
    num1 = 29000
    num2 = 29000
    
    v0 = 2.5
    sigma = 1/3
    
    r = np.array([2, 2, 2])
    v = np.array(sigma*np.random.randn(3))# + (top + right) / 2)
    
    
    #start = time.time()
    #print(seperateloop(r, top, right, q1, q2, m1, m2, num1, num2, sigma))
    #end = time.time()
    #print(end - start)
    
    r = np.array([2, 2, 2])
    start = time.time()
    #print(seperatemat(r, top, right, q1, q2, m1, m2, num1, num2, sigma, dt))
    end = time.time()
    print(end - start)
    """
    storex = []
    storey = []
    
    for i in range(100):
        r = np.array([2, 2, 2])
        v = np.array(sigma*np.random.randn(3) + v0)
        while 1:
            r, v = push(v, r, dt, q, m)
            if r[0] <= 0 or r[0] >= right:
                storey.append(r[1])
                break
            if r[1] <= 0 or r[1] >= top:
                storex.append(r[0])
                break
        
    mp.pyplot.hist(storey, bins=[0, 1, 2, 3, 4, 5])
    """




