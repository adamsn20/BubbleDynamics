#!/usr/bin/env python
# coding: utf-8

# # Version 2 

# In[1]:


#-------------------------------------------------------------------------------------------------------------#
#------------------------------------------------Import Packages----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------#


import enum
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing._private.utils import tempdir
from matplotlib.animation import FuncAnimation #package to generate mp4 of the simulation
from IPython import display #package to display mp4
from scipy import integrate
from tqdm import tqdm
import copy
from scipy.interpolate import CubicSpline,interp1d
from scipy.integrate import quad
import pandas as pd
import math
import tkinter as tk
from tkinter import *
from tkinter import ttk
#all units in SI (Joule, N, m, s, W, mol, kg, K, etc.) unless noted.
#pressures are absolute unless noted.

#-------------------------------------------------------------------------------------------------------------#
#---------------------------------------------Define Global Variables-----------------------------------------#
#-------------------------------------------------------------------------------------------------------------#


#Global constants
gasConstant = 8.31446 #Joules/mol/K
refTemp = 298.15 #K
refPressure = 1E5 #N/m^2
refGamma = 1.4 #gamma of exterior air
airDensity = 1.22 #kg/m^3
airMW = 0.029 #kg/mol

#-------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------GUI Formatting----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------#

root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()

L1 = ttk.Label(frm, text="How many simulations do you want to run:",  justify="left").grid(column=0, row=0)
T1_Value = StringVar() 
T1 = ttk.Entry(frm, textvariable = T1_Value, justify = CENTER).grid(column=1, row=0)

L2 = ttk.Label(frm, text="What burn rate would you like to simulate:",  justify="left").grid(column=0, row=1)
T2_Value = DoubleVar() 
T2 = ttk.Entry(frm, textvariable = T2_Value, justify = CENTER).grid(column=1, row=1)

L3 = ttk.Label(frm, text="",  justify="left").grid(column=0, row=2)
L4 = ttk.Label(frm, text="Choose which graphs you want to print:",  justify="left").grid(column=0, row=3)

CB_Style = ttk.Style()
CB_Style.configure('TCheckbutton', width = 40, justify = tk.LEFT)

CB1_MPS = tk.IntVar()
CB2_MPSS = tk.IntVar()
CB3_HB_PA = tk.IntVar()
CB4_HB_PvT = tk.IntVar()
CB7_TNT= tk.IntVar()
CB8_BRM= tk.IntVar()
CB1 = ttk.Checkbutton(frm, text="Max Pressure of All Simulations", style = "TCheckbutton", variable = CB1_MPS).grid(column=0, row=4)
CB2 = ttk.Checkbutton(frm, text="Max Pressure of a Single Simulation", style = "TCheckbutton", variable = CB2_MPSS).grid(column=0, row=5)
CB3 = ttk.Checkbutton(frm, text="Heated Bubble - Position Animation", style = "TCheckbutton", variable = CB3_HB_PA).grid(column=0, row=6)
CB4 = ttk.Checkbutton(frm, text="Heated Bubble - Pressure vs. Time", style = "TCheckbutton", variable = CB4_HB_PvT).grid(column=0, row=7)
CB8 = ttk.Checkbutton(frm, text="Average Pressure vs. Distance", style = "TCheckbutton", variable = CB8_BRM).grid(column=0, row=8)
CB7 = ttk.Checkbutton(frm, text="TNT Equivalency", style = "TCheckbutton", variable = CB7_TNT).grid(column=0, row=9)


B1 = ttk.Button(frm, text="Submit", command=root.destroy).grid(column=2, row=10)

root.mainloop()

inputText1 = int(T1_Value.get())
inputText2 = int(T2_Value.get())

#-------------------------------------------------------------------------------------------------------------#
#----------------------------------------------Define Bubble Class--------------------------------------------#
#-------------------------------------------------------------------------------------------------------------#

class Bubble:
    """This class represents a bubble of gas."""
    #Class attributes
    epsln = 20 #Lennard Jones Potential epsilon of every bubble (purely repulsive)
    #Class method
    @staticmethod
    def interBubbleForce(sigma,r):     #Lennard Jones 
        return 24*Bubble.epsln/r*(sigma/r)**6 #softer repulsion (normally to the power 12 and no attraction)

    @staticmethod
    def EOS(moles,temperature,dia): #  Ideal Gas, return the pressure given number of moles, temperature, volume
        return moles*gasConstant*temperature/(np.pi/6*dia**3)

    @staticmethod
    def EOSDerivs(moles,b): #return dTdt given dnUdt and moles
        dTdt = (b.derivs['dnUdt'] - gasConstant/(b.gamma-1)*b.derivs['dndt'])*(b.gamma-1)/(gasConstant*moles)
        #assuming for now that the size of each bubble doesn't change which means this isn't isentropic
        
        #dVdt = (np.pi/6*self.size**3)/(1-self.gamma)*(self.derivs['dTdt']/self.temperature + self.derivs['dndt']/self.moles)
        #dPdt = moles*gasConstant*dTdt + gasConstant*temperature*b.derivs['dndt'] - pressure*dVdt
        
        return dTdt

    @classmethod
    def mach(cls, upstreamPressure, downstreamPressure, gamma): #See Perry's pages 6-23 (Eq.6-115 and 6-119)
        if (upstreamPressure >= downstreamPressure):
           ratio = upstreamPressure/downstreamPressure
           multiplier = -1.       
        else:
            ratio = downstreamPressure/upstreamPressure
            multiplier = 1.
        mach_abs = np.sqrt(2/(gamma-1)*(ratio**((gamma-1)/gamma)-1))
        return min(1.,mach_abs)*multiplier

    #Initializer / Instance attributes
    def __init__(self, gamma=1.4, Mw=0.029, pos=[0,0,0], temperature=refTemp, pressure=refPressure, size=0.01, dragC=0.5, heatPoint = [0,0,0], force = [0,0,0]): #<-- This is the original (I've added "heatPoint", the [x,y,z] array of the container's heatPoint and "force", the force vector of each bubble)    
        """Size is the diamter of the bubble."""
        
        self.force=force
        
        self.gamma = gamma #heat capacity ratio
        self.Mw = Mw #molecular weight
        self.pos = pos #position array
        self.temperature = temperature 
        self.pressure = pressure       
        self.size = size #diameter of the bubble
        self.dragC = dragC #drag coefficient, a sphere is 0.5 and a cube is 0.8
        self.heatPoint = heatPoint
        
        #set velocities, moles, and mass
        ar = np.random.rand(3)*2-1; mag = np.sqrt(np.dot(ar,ar)); ar = ar/mag
        self.vel = np.dot(np.sqrt(self.gamma*gasConstant*temperature/Mw),ar) #possible reduction in temperature for longer sim times 
        self.moles = (4/3*np.pi*(self.size/2)**3)*pressure/(gasConstant*temperature) #moles in the bubble
        self.mass = self.moles*self.Mw
        
        #initialize neighborhood array and force parameters
        self.neighbors = []
        self.posp = copy.copy(self.pos) #previous position upon update of the neighbors array
        
        #initialize derivative dictionary
        self.derivs = {'dndt':0,'dnUdt':0}
        
        #initialize Runge-Kutta variables
        self.posK = [0]*4; self.velK = [0]*4; self.nK = [0]*4; self.TK = [0]*4
        
        #set center for the bubble that is closest to the center; the center bubble is where the burning occurs, and totalenthalpy change for heated event
        self.center = False; self.heatedH = 0

    def dndt(self,moles,temperature,downstreamPressure,area):
        pressure = Bubble.EOS(moles=moles,temperature=temperature,dia=self.size); gamma=self.gamma; Mw=self.Mw
        MachNu = Bubble.mach(pressure,downstreamPressure,self.gamma) #See Equation 6-118 in Perry's
        if MachNu>0:
            #if the pressure of the bubble is less than the external pressure of the box; also assumes that this function isn't called when pressure is lower than downstream for interacting bubbles (same for dnUdt)
            pressure = refPressure; temperature = refTemp; gamma = refGamma; Mw = airMW
        return pressure*area*np.sqrt(gamma/(gasConstant*temperature*Mw))*MachNu*(1+(gamma-1)/2*MachNu**2)**((gamma+1)/(2-2*gamma))
    
    def dnUdt(self,dndt,temperature):
        gamma = self.gamma;
        if dndt>0:
            #if flow is from the exterior of the container volume; this function should not be called for interacting bubbles with lower pressure than their neighbor (same for dndt)
            gamma = refGamma; temperature = refTemp
        return dndt*gamma*gasConstant/(gamma-1)*(temperature-refTemp)

    def derivsCalc(self, pos, vel, moles, temperature, time, c, actual, heatPoint):
        
        relaxT = .001 #set desired relaxation period in seconds where the bubbles can orient themselves before heating begins
        
        #acceleration from the Container walls (Lennard Jones potential of walls) purely repulsive
        d2 = np.subtract(pos,c.dimensions) #distance from wall in x,y,z direction
        
        accelx = Container.wallForce(self.size,pos[0]) + Container.wallForce(self.size,d2[0]) #first term is for the walls at x=0,y=0,and z=0. The second term is for the walls at x=xlen,y=ylen, and z=zlen.
        accely = Container.wallForce(self.size,pos[1]) + Container.wallForce(self.size,d2[1])
        accelz = Container.wallForce(self.size,pos[2]) + Container.wallForce(self.size,d2[2])
        accel = np.divide([accelx,accely,accelz],moles*self.Mw)
        
        # for the center most bubble, add the addition of the burning after a relaxation period for bubbles to position themselves
        if time > relaxT: #relaxation period of 0.001 seconds
            if self.center:
                sigma = c.burnRateMultiplier*(3e-4); ave = c.burnRateMultiplier*(1e-3)
                self.derivs['dnUdt'] += self.heatedH/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((time-ave)/sigma)**2)        ######### REMEMBER THIS LOCATION FOR MASS ######
        #determine overlap with wall for venting through the wall (6 walls)
        r1 = pos[0]-c.dimensions[0]; r2 = pos[1]-c.dimensions[1]; r3 = pos[2]-c.dimensions[2]
        arp = (self.size/2)**2
        ventAreaWall = np.pi*(max([arp - r1**2,0]) +  max([arp - r2**2,0]) + max([arp - r3**2,0]) + max([arp - pos[0]**2,0]) + max([arp - pos[1]**2]) + max([arp - pos[2]**2,0]))
        if ventAreaWall > 0 and c.allowBubbleInteractions: #bubble interacts with wall
            dndtWall = self.dndt(moles=moles,temperature=temperature,downstreamPressure=refPressure,area=ventAreaWall)
            self.derivs['dndt'] += dndtWall
            self.derivs['dnUdt'] += self.dnUdt(dndtWall,temperature)
        #acceleration from contact with other bubbles using neighbors with buffer check    
        for nbub in self.neighbors:
            dist = np.linalg.norm(pos - nbub.pos) #distance between bubbles 
            msize = (self.size + nbub.size)/2 #average size (common mixing rule)
            self.force = Bubble.interBubbleForce(msize,dist)*np.subtract(pos,nbub.pos)/dist
            
            if self.center:    #for the center bubble, include a Proportionalcontroller to add a restraining force to keep it at the center.
                heatPointDist = np.linalg.norm(pos-heatPoint)
                self.force += (1750*heatPointDist)*np.subtract(heatPoint,pos)/heatPointDist #P-I Controller of the form u(t) = Kp*e(t) + Ki*∫(0->t)(e(𝜏)d𝜏) where Kp and Ki are non-negative coefficients        
                                #best coefficient so far: 1750 or 2000
            
            accel = np.add(accel,self.force/(moles*self.Mw))
            ventArea = np.pi/(4*dist**2)*(4*(dist*self.size/2)**2 - (dist**2 - nbub.size**2 + self.size**2)**2)
            if ventArea > 0 and c.allowBubbleInteractions: #bubbles interact
                pressure = Bubble.EOS(moles,temperature,self.size)
                if pressure > nbub.pressure: #this is used to not double up on the interactions with the neighborhood list
                    dndt_i_j = self.dndt(moles=moles,temperature=temperature,downstreamPressure=nbub.pressure,area=ventArea)
                    self.derivs['dndt'] += dndt_i_j
                    nbub.derivs['dndt'] += -dndt_i_j
                    dnUdt_i_j = self.dnUdt(dndt=dndt_i_j,temperature=temperature)
                    self.derivs['dnUdt'] += dnUdt_i_j
                    nbub.derivs['dnUdt'] += -dnUdt_i_j
        c.stat['pv'] += moles*self.Mw*np.dot(vel,vel)
        c.stat['fv'] += moles*self.Mw*np.dot(accel,vel)                    
        return accel
    
#-------------------------------------------------------------------------------------------------------------#
#---------------------------------------------Define Container Class------------------------------------------#
#-------------------------------------------------------------------------------------------------------------#    
    
class Container: 
    """ This is the container class where all of the bubbles are located."""
	#Class attributes
    #class method(s)
    @staticmethod
    def wallForce(sig,dis):
        return 24*Bubble.epsln/(dis*2)*(sig/(dis*2))**6 #this is assuming that the wall epsilon is the same as the bubble epsilon

    #Initializer / Instance attribute
    #def __init__(self, bubbleCount=64, bubbleDiameter=1.3472/4, xlen=1.3472, ylen=1.3472, zlen=1.3472, heatPoint = [.6736,.6736,.6736], finalIsoPressure=refPressure*8, allowBubbleInteractions=True, burnRateMultiplier=1): #<-- original conditions
    #def __init__(self, bubbleCount=4000, bubbleDiameter=2.3691, xlen=50, ylen=50, zlen=25, heatPoint = [50/2,50/2,0], finalIsoPressure=refPressure*8, allowBubbleInteractions=True, burnRateMultiplier=1):                   #<-- Literature conditions
    def __init__(self, bubbleCount=256, bubbleDiameter=2.3691, xlen=20, ylen=20, zlen=10, heatPoint = [10/2,10/2,0], finalIsoPressure=refPressure*8, allowBubbleInteractions=True, burnRateMultiplier=1):    
        #change xlen,ylen,zlen to just be an array of the container l, w, and h and change following code accordingly
        self.heatPoint = heatPoint
        self.bubbleCount = bubbleCount
        slotWidth = bubbleDiameter #bubble diameter
        bubVol = (np.pi/6*bubbleDiameter**3)
        self.size = slotWidth
        self.centerCoords=[xlen/2,ylen/2,zlen/2]
        self.dimensions = [xlen,ylen,zlen]
        self.vol = xlen*ylen*zlen
        self.rbuff = slotWidth*1.5
        self.allowBubbleInteractions = allowBubbleInteractions #this is to allow or not the interchange of gases between bubbles
        self.burnRateMultiplier = burnRateMultiplier #this is the multiplier to the rate that heat is released into the volume
        
        # initialize volume and placement with given bubbles
        self.bubbles = []; count = 0; flag = True
        maxbubbles = math.floor(xlen/bubbleDiameter)*math.floor(ylen/bubbleDiameter)*math.floor(zlen/bubbleDiameter) #calculates the max number of bubbles along each specified length of the container (bubbles placed side-by side and no partial bubbles). Then multiplies the number of bubbles to get the max bubbles in the container
        if bubbleCount > maxbubbles:
            raise ValueError("Too many bubbles or bubble diameter is to large!")
        
        for i in range(math.floor(xlen/bubbleDiameter)):
            for j in range(math.floor(ylen/bubbleDiameter)):
                for k in range(math.floor(zlen/bubbleDiameter)): 
                    pressure = refPressure #*2 if i==0 else refPressure
                    temperature = refTemp #*2 if i==0 else refTemp
                    bubble = Bubble(pos=np.multiply([i+0.5,j+0.5,k+0.5],slotWidth), temperature=temperature, pressure=pressure, size=slotWidth, heatPoint=self.heatPoint)
                    #if np.linalg.norm(bubble.pos - self.centerCoords) < self.rbuff and flag:
                     #       bubble.center = True; flag = False
                      #      bubble.heatedH = (finalIsoPressure-refPressure)*bubVol/(bubble.gamma-1)
                    self.bubbles.append(bubble)                   
                    count += 1
        
        for i in range(maxbubbles-bubbleCount):
            del self.bubbles[-1]
        
        heatedBub = min([[np.linalg.norm(bub.pos - heatPoint),ID] for ID,bub in enumerate(self.bubbles)])[1]
        self.heatedBub = heatedBub
        self.bubbles[heatedBub].center = True
        self.bubbles[heatedBub].heatedH = (finalIsoPressure-refPressure)*bubVol/(bubble.gamma-1)
        
        #initialize thermostat sums (maintains the temperature constant for the simulation)
        self.stat = {'pv':0,'fv':0}                        
        self.history = [] #initialize array 
        self.fnab() #initialize the neighborhood list and get initial interbubble forces

    def zero(self):
        for bub in self.bubbles: bub.derivs = {'dndt':0,'dnUdt':0,'dVdt':0} #reset bubble derivatives
        self.stat = {'pv':0,'fv':0} #zero sums for thermostat
        return None

    def fnab(self): #determine forces between neighbors and update neighbors array (all in the same double loop)
        upb = len(self.bubbles)
        for bub in self.bubbles:
            bub.neighbors = [] #reset neighbors array for each
            bub.posp = copy.copy(bub.pos) #reset previous position for each            
        for i,bubi in enumerate(self.bubbles):   
            for j in range(i+1,upb):
                dist = np.linalg.norm(bubi.pos - self.bubbles[j].pos) #distance between bubbles 
                if dist <= self.rbuff: 
                    bubi.neighbors.append(self.bubbles[j])
                    self.bubbles[j].neighbors.append(bubi)
        return None

    def copies(self): #create a dictionary housing a copy of the variables when this function is called
        snapshot = {'moles':[copy.copy(each.moles) for each in self.bubbles],
        'pressure':[copy.copy(each.pressure) for each in self.bubbles],'temperature':[copy.copy(each.temperature) for each in self.bubbles],
        'vel':[copy.copy(each.vel) for each in self.bubbles],'pos':[copy.copy(each.pos) for each in self.bubbles],
        'containerTemperature': sum([np.dot(bub.vel,bub.vel)*bub.Mw/gasConstant/bub.gamma for bub in self.bubbles])/len(self.bubbles), 'force':[copy.copy(each.force) for each in self.bubbles]}
        self.history.append(snapshot)
        return None
               
#-------------------------------------------------------------------------------------------------------------#
#-------------------------------------------Define Simulation Function----------------------------------------#
#-------------------------------------------------------------------------------------------------------------#

delt = 1E-4    #desired size of each time step
tSteps = 1000 #number of desired time steps for simulation

def simulate(container,dt=0.01,steps=1000): #dt and simtime in seconds (SI units)
    times = [0]
    container.copies()
    
    for j in tqdm(range(steps)):   
        container.zero() #zero derivate accumulators for all bubbles and zero thermostat sums    
        #Runge-Kutta k1's                
        for bub in container.bubbles: #this loop finds the accel for each bubble as well as accumulating the dnU/dt and dn/dt derivative properties of each bubble
            bub.velK[0] = np.multiply(bub.derivsCalc(pos=bub.pos,vel=bub.vel,c=container,moles=bub.moles,temperature=bub.temperature,time=times[-1],actual=True,heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            bub.velK[0] -= container.stat['fv']/container.stat['pv']*bub.vel*dt
            bub.posK[0] = np.multiply(bub.vel,dt)
            dTdt = Bubble.EOSDerivs(moles=bub.moles,b=bub)
            bub.nK[0] = bub.derivs['dndt']*dt
            bub.TK[0] = dTdt*dt
        container.zero() #zero derivate accumulators for all bubbles and zero thermostat sums         
        #Runge-Kutta k2's 
        for bub in container.bubbles:            
            bub.velK[1] = np.multiply(bub.derivsCalc(pos=np.add(bub.pos,0.5*bub.posK[0]),vel=np.add(bub.vel,0.5*bub.velK[0]),moles=bub.moles+0.5*bub.nK[0],temperature=bub.temperature+0.5*bub.TK[0],c=container,time=times[-1],actual=False,heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            bub.velK[1] -= container.stat['fv']/container.stat['pv']*np.add(bub.vel,0.5*bub.velK[0])*dt
            bub.posK[1] = np.add(bub.vel,bub.velK[0]*0.5)*dt
            dTdt = Bubble.EOSDerivs(moles=bub.moles+0.5*bub.nK[0],b=bub)
            bub.nK[1] = bub.derivs['dndt']*dt
            bub.TK[1] = dTdt*dt
        container.zero() #zero derivate accumulators for all bubbles and zero thermostat sums            
        #Runge-Kutta k3's
        for bub in container.bubbles:
            bub.velK[2] = np.multiply(bub.derivsCalc(pos=np.add(bub.pos,0.5*bub.posK[1]),vel=np.add(bub.vel,0.5*bub.velK[1]),moles=bub.moles+0.5*bub.nK[1],temperature=bub.temperature+0.5*bub.TK[1],c=container,time=times[-1],actual=False, heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            bub.velK[2] -= container.stat['fv']/container.stat['pv']*np.add(bub.vel,0.5*bub.velK[1])*dt
            bub.posK[2] = np.add(bub.vel,bub.velK[1]*0.5)*dt
            dTdt = Bubble.EOSDerivs(moles=bub.moles+0.5*bub.nK[1],b=bub)
            bub.nK[2] = bub.derivs['dndt']*dt
            bub.TK[2] = dTdt*dt
        container.zero() #zero derivate accumulators for all bubbles and zero thermostat sums
        #Runge-Kutta k4's
        for bub in container.bubbles:
            bub.velK[3] = np.multiply(bub.derivsCalc(pos=np.add(bub.pos,bub.posK[2]),vel=np.add(bub.vel,bub.velK[2]),moles=bub.moles+bub.nK[2],temperature=bub.temperature+bub.TK[2],c=container,time=times[-1],actual=False, heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            bub.velK[3] -= container.stat['fv']/container.stat['pv']*np.add(bub.vel,bub.velK[2])*dt
            bub.posK[3] = np.add(bub.vel,bub.velK[2])*dt
            dTdt = Bubble.EOSDerivs(moles=bub.moles+bub.nK[2],b=bub)
            bub.nK[3] = bub.derivs['dndt']*dt
            bub.TK[3] = dTdt*dt
            #Now update variables
            bub.vel += 1/6*(bub.velK[0]+2*bub.velK[1]+2*bub.velK[2]+bub.velK[3])
            bub.pos += 1/6*(bub.posK[0]+2*bub.posK[1]+2*bub.posK[2]+bub.posK[3]) 
            bub.temperature += 1/6*(bub.TK[0]+2*bub.TK[1]+2*bub.TK[2]+bub.TK[3])
            bub.moles += 1/6*(bub.nK[0]+2*bub.nK[1]+2*bub.nK[2]+bub.nK[3]) 
            bub.pressure = Bubble.EOS(moles=bub.moles,temperature=bub.temperature,dia=bub.size)        
        container.zero() #zero derivate accumulators for all bubbles and zero thermostat sums                         
        if max([np.linalg.norm(bub.pos - bub.posp) for bub in container.bubbles])>container.rbuff*0.4: 
            # print('Called fnab')
            container.fnab() #update neighborhood lists
            
            #The following are used as checkpoints to return specified values at different intervals as the simulations run; they are called whenever the neighborhood list updates.
            #dist_heatedBub = np.linalg.norm((container.history[j]['pos'][container.heatedBub])-container.heatPoint)
            #print(container.history[j]['pos'][container.heatedBub])  
            #print(container.history[j]['force'][container.heatedBub])    
            #print(f'heated bubble distance from heat point: {dist_heatedBub}')  
            
        #time and copy        
        times.append(dt+times[j])
        container.copies()
    return times

#-------------------------------------------------------------------------------------------------------------#
#--------------------------------------------Define Graphing Functions----------------------------------------#
#-------------------------------------------------------------------------------------------------------------#

# for the xth time step, find the pressure versus distance plot
def PvsPos(cont,timestep):
    pressure = []; spacebetween = []
    centerpos = cont.history[timestep]['pos'][cont.heatedBub]
    for each in range(len(cont.bubbles)):
        bubblepos = cont.history[timestep]['pos'][each]
        spacebetween.append(np.linalg.norm(centerpos-bubblepos))
        pressure.append(cont.history[timestep]['pressure'][each])
    pressure = np.array(pressure)/1e5 - 1                             #Pa to bar "-1" to change to gauge pressure
    arr = np.array([spacebetween,pressure])
    sarr = arr[:,arr[0].argsort()]
    cs = interp1d(sarr[0],sarr[1])
    return (sarr,cs)
    
def MaxPressure(mCont): 
    # equation to calculate the max pressures over all the simulations                       #is this the max pressure of each simulation or max of all of them combined?
    distances = np.linspace(0.001,graphDist,100)                                                   #will this range have to change depending on the size of the box?
    maxP = np.zeros(len(distances))
    # loop through each of the containers to find the max Pressure vs distance
    for j in range(len(mCont)):
        #loop over each time step and find the max pressure at a given distance
        for i in range(1,tSteps):                                                              
            out = PvsPos(mCont[j],i) #returns pressure vs distance and cubic spline
            Ps = out[1](distances) #get pressures at given distance from spline
            for i,p in enumerate(Ps):
                if p > maxP[i]:
                    maxP[i] = p #reset max pressure
    return maxP,distances

def GPressure(cont): 
    # prints a graph of the container for the max Pressure vs distance
    #loop over each time step and find the max pressure at a given distance
    distances = np.linspace(0.001,graphDist,100)                                                   #will this range have to change depending on the size of the box?
    maxP = np.zeros(len(distances))
    for i in range(1,tSteps):                                                                  
        out = PvsPos(cont,i) #returns pressure vs distance and cubic spline
        Ps = out[1](distances) #get pressures at given distance from spline
        for i,p in enumerate(Ps):
            if p > maxP[i]:
                maxP[i] = p #reset max pressure  

    plt.scatter(distances,maxP,label='bubble dynamics dropoff')
    plt.plot(distances,0.05/(distances**2),'r',label='squared distance dropoff model')
    plt.legend()
    plt.xlabel('distance'); plt.ylabel('Gauge Pressure')
    plt.ylim([0,2])
    plt.show()
    return

def singMaxPressure(cont):
#loop over each time step and find the max pressure at a given distance
    distances = np.linspace(0.001,graphDist,100)                                                  #will this range have to change depending on the size of the box?
    maxP = np.zeros(len(distances))
    for i in range(1,tSteps):                                                                 
        out = PvsPos(cont,i) #returns pressure vs distance and cubic spline
        Ps = out[1](distances) #get pressures at given distance from spline
        for i,p in enumerate(Ps):
            if p > maxP[i]:
                maxP[i] = p #reset max pressure  

    maxP = maxP*14.405 # multiply by 14.405 to convert from barg to psig

    plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.scatter(distances*3.281,maxP,label='bubble dynamics dropoff',s=75)
    plt.plot(distances*3.281,14.405*(0.05/(distances**2)),'r',label='squared distance dropoff model',linewidth='5')
    plt.legend()
    plt.xlabel('Distance (ft)',size=15); plt.ylabel('Pressure (psig)',size=15)
    plt.title('Max Pressure of Single Simulation')
    plt.ylim([0,40])
    plt.grid()
    
    plt.savefig('SingleMaxPressure')     #uncomment to save file externally

    return

def D3Graph(cont,position=21): 
    # obtain the location in x,y,z coordinates
    # at the bubble position determined by posititon
    # in the container determined by cont
    # center bubble is 21
    x = [each['pos'][position][0] for i,each in enumerate(cont.history)]
    y = [each['pos'][position][1] for i,each in enumerate(cont.history)]
    z = [each['pos'][position][2] for i,each in enumerate(cont.history)]

    # ax = plt.figure().add_subplot(projection='3d')
    # plt.plot(x, y, z)
    # plt.show()
    return x,y,z

def bubPressure(cont,times,bubbleNum=21):
    
    # returns the pressure of the bubble bubbleNum over time
    press_bubNum = [each['pressure'][bubbleNum]/6895 for i,each in enumerate(cont.history)] # divide by 6895 to change from Pa to psi
    
    plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.plot(times,press_bubNum,'firebrick',linewidth = '5')
    plt.grid()
    plt.ylim([14,62])     #I recently changed this from 26 to 28, as graph often went off of the graph                                                                
    plt.xlabel('Time (s)',size=15); plt.ylabel('Pressure (psi)',size=15)
    plt.title('Pressure Vs. Time for Heated Bubble')
    plt.savefig('PressureTime #'+ str(bubbleNum))     #uncomment to save to external file
    
    return

def AvePressure(mCont): 
    # plots a graph of the average max pressures of all the simulations.
    
    # equation to calculate the max pressures over all the simulations
    distances = np.linspace(0.001,graphDist,100)
    aveP = np.zeros(len(distances))
    # loop through each of the containers to find the max Pressure vs distance
    for j in range(len(mCont)):
        #loop over each time step and find the max pressure at a given distance
        maxP = np.zeros(len(distances))
        for i in range(1,tSteps):
            out = PvsPos(mCont[j],i) #returns pressure vs distance and cubic spline
            Ps = out[1](distances) #get pressures at given distance from spline
            for i,p in enumerate(Ps):
                if p > maxP[i]:
                    maxP[i] = p #reset max pressure
        aveP += maxP
    aveP = aveP/nSimulations  # Change this value for nSimulations if you update it above
    #print(f"Nsimulations: {nSimulations}")
    return aveP,distances

def PvsTatD(distance):     #later code so the repetitive interpolation is done only once, even if this function is called multiple times for various distances
    pressures = np.zeros(tSteps)     #set up array for pressures at each time step for the specified distance
    for simNum in range(len(multiCont)):     #loop over all the simulations to later average the pressures
        pres = np.zeros(len(multiCont[simNum].bubbles))
        dist = np.zeros(len(multiCont[simNum].bubbles))
        for time in range(tSteps):     #loop over all the timesteps to interpolate pressure vs. distance at each timestep, then pull the desired pressure value at the specified distance
            for bub in range(len(multiCont[simNum].bubbles)):     #loop over each bubble to find the given pressure and distance
                pres[bub] = multiCont[simNum].history[time]['pressure'][bub]
                dist[bub] = np.linalg.norm(multiCont[simNum].history[time]['pos'][multiCont[simNum].heatedBub]-multiCont[simNum].history[time]['pos'][bub])
            pres = ((np.array(pres))*1.45038E-4) - 14.5038    #convert from Pa to psi, subtract 14.5038 to convert to gauge pressure
            PvsD = np.array([dist,pres])
            sortPvsD = PvsD[:,PvsD[0].argsort()]
            interPvsD = interp1d(sortPvsD[0],sortPvsD[1])
            pressures[time] += interPvsD(distance)
    pressuresAve = np.array(pressures)/len(multiCont)     #average the pressures at each timestep for the specified distance
    
    plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.plot(times[0][:-1],pressuresAve,'firebrick', linewidth = '5')
    plt.grid()
    plt.ylim([0,1.5*max(pressuresAve)])                                                                     
    plt.xlabel('Time (s)',size=15); plt.ylabel('Pressure (psi)',size=15)
    plt.title(f'Pressure Vs. Time at {distance} m')
    plt.savefig(f'Pressure Vs Time at {round(distance)} m')     #uncomment to save to external file
    
    return

#-------------------------------------------------------------------------------------------------------------#
#------------------------------------------------Run Simulations----------------------------------------------#
#-------------------------------------Currently Running both BRM1 and BRM2------------------------------------#
#-------------------------------------------------------------------------------------------------------------#

# input number of simulations
nSimulations = inputText1                  #5 is okay for tests

multiCont = []
times = []


i = 0
for i in range(nSimulations):
    multiCont.append(Container(burnRateMultiplier=inputText2))
    times.append(simulate(multiCont[i],dt=delt,steps=tSteps))


#-------------------------------------------------------------------------------------------------------------#
#---------------------------------------------Call Graphing Functions-----------------------------------------#
#-------------------------------------------------------------------------------------------------------------#


graphDist = 0.5*max(multiCont[0].dimensions)     #to scale graphs with the customizable size of the container, we chose the distance to be graphed to as 50% of the longest container length. 50% was chosen to prevent trying to plot distances outside of the interpolator range in PvsPos(). Note that "distance" in plotting refers to distance from the center, heated bubble.



maxP,distances = MaxPressure(multiCont) # returns the distances and the max pressures of all simulations

# PLOT THE MAX PRESSURES OF ALL SIMULATIONS

maxP = (maxP)*14.504     # multiply pressure by 14.504 change from barg to psig

if CB1_MPS.get() == 1:
    plt.scatter(distances*3.281,maxP,label='bubble dynamics dropoff')    # multiply distances by 3.281 to convert from m to ft
    plt.plot(distances*3.281, 14.405*(0.05/(distances**2)),'r',label='squared distance dropoff model')
    plt.legend()
    plt.xlabel('Distance (ft)'); plt.ylabel('Pressure (psig)')
    plt.ylim([0,40])
    plt.title('Max Pressure of Simulations')

    plt.savefig('MaxPressureSimulations')     #uncomment to save to external file   

#-#

#PLOT THE MAX PRESSURE OF A SINGLE DEFINED SIMULATION
if CB2_MPSS.get() == 1:
    singMaxPressure(multiCont[0])
    plt.title('Max Pressure of Single Simulation')

#-#

# ANIMATION OF CENTER BUBBLE POSITION
if CB3_HB_PA.get() == 1:
    x,y,z = D3Graph(multiCont[0])
    x1,y1,z1 = D3Graph(multiCont[0],20)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    data = np.array([x,y,z])
    data1 = np.array([x1,y1,z1])

    def animate(t,data,line,line1):
        line.set_data(data[:2, :t])
        line.set_3d_properties(data[2,:t])
        line1.set_data(data1[:2, :t])
        line1.set_3d_properties(data1[2,:t])

    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
    line1, = ax.plot(data1[0, 0:1], data1[1, 0:1], data1[2, 0:1])

    # Setting the axes properties
    ax.set_xlim3d([0.0, 0.8])                                                  #experiment with what these axis limits need to be
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, 1.2])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 1.2])
    ax.set_zlabel('Z')

    ani = FuncAnimation(fig, animate, frames=len(x), fargs=(data,line,line1), interval = 1)

    ani.save('animation .gif',writer='pillow',fps=1000)

#-#

#PLOT PRESSURE VS. TIME FOR THE HEATED BUBBLE
if CB4_HB_PvT.get() ==1:
    simulationNum = 0 # value for the simulation number
    bubbleNum = multiCont[simulationNum].heatedBub # value of the bubble desired (21 for a 4x4 grid of bubbles)
    bubPressure(multiCont[simulationNum],times[simulationNum],bubbleNum)
    plt.title('Pressure Vs. Time for Heated Bubble')
    #-#

#PLOT AVERAGE PRESSURES OF SIMULATIONS AT EACH DISTANCE 

if CB8_BRM.get() == 1:
    aveP,distances = AvePressure(multiCont)

    aveP = aveP*14.504 # multiply by 14.504 to convert from barg to psig 

    plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.scatter(distances*3.281,aveP,label='bubble dynamics dropoff',s=75)
    plt.plot(distances*3.281,14.405*(0.05/(distances**2)),'r',label='squared distance dropoff model',linewidth='5')
    plt.legend()
    plt.grid()
    plt.xlabel('Distance (ft)',size=15); plt.ylabel('Pressure (psig)',size=15)
    plt.title('Average Pressures vs. Distance')
    plt.ylim([0,40])

    plt.savefig('AveragePressure')     #uncomment to save to external file

#-#

#PLOT TNT EQUIVALENCE

if CB7_TNT.get() == 1:
    
    aveP,distances = AvePressure(multiCont)
    aveP = aveP*14.504 # multiply by 14.504 to convert from barg to psig 

    for bubble in multiCont[0].bubbles:
        if bubble.center:
            heated_h_value = bubble.heatedH
            #print("heatedH of the center bubble:", heated_h_value)
            break  # If you only want the value from the center bubble

    mass = heated_h_value / 10_000 # 10 kJ/g                                                   #what is this 10_000 kJ/g
    # Z = R/W**(1/3)
    Z2 = (distances*3.281)/((mass/453.6)**(1/3)) # change the distance from m to ft by dividing by 3.281 and change mass from g to lb by dividing by 453.6

    # get points from spherical chart
    Z = np.array([0.1,1,5,10,50,100])
    U = np.array([10_000,1_000,30,7,0.7,0.3])

    plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.plot(Z,U,'firebrick',label='Standard TNT',linewidth = '5')
    plt.plot(Z2[1:],aveP[1:],'darkblue',label=f'Burn Rate Multiplier {inputText2}',linewidth = '5')
    #plt.plot(Z2[1:],aveP1[1:],'blue',label='Burn Rate Multiplier 0.5',linewidth = '5')     #should this be 'Burn Rate Multiplier 2'?
    plt.xlabel(r'$Z = R/W^{1/3}$ ($ft/lbm^{1/3}$)',size=15)
    plt.ylabel('$P_{so}$ (psig)',size=15)
    plt.title('TNT Equivalency')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    plt.savefig('TNT Equivalency.png')     #uncomment to save to external file

    plt.show()

#-#

#PLOT PRESSURE VS. TIME FOR SPECIFIED DISTANCES.
    
PvsTatD(distance=9.144)
#PvsTatD(distance=17.773)
#PvsTatD(distance=18.288)
#PvsTatD(distance=17.773)
#PvsTatD(distance=18.288)
#PvsTatD(distance=23.806)
#PvsTatD(distance=26.614)
#PvsTatD(distance=30.970)
#PvsTatD(distance=49.987)


# ## Things to Do

# - Change all the graphing to functions
# - Change burn rate/applied heating to being able to input a set pressure

# In[ ]:


PvsTatD(distance=0.5)   
PvsTatD(distance=1)
PvsTatD(distance=2)
PvsTatD(distance=9.144)


# In[56]:


print(inputText2) #print the most recent burnrate used


# In[31]:


bubPressure(multiCont[3],times[3],72)
plt.ylim(14.5,35)


# In[57]:


def bubPressureGauge(cont,times,bubbleNum=21):
    
    # returns the pressure of the bubble bubbleNum over time
    press_bubNum = [(each['pressure'][bubbleNum]/6895)-14.503 for i,each in enumerate(cont.history)] # divide by 6895 to change from Pa to psi subtract for gauge pressure
    
    plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.plot(times,press_bubNum,'firebrick',linewidth = '5')
    plt.grid()
    plt.ylim([0,35])     #I recently changed this from 26 to 28, as graph often went off of the graph                                                                
    plt.xlabel('Time (s)',size=15); plt.ylabel('Pressure (psi)',size=15)
    plt.title('Pressure Vs. Time for Heated Bubble')
    #plt.savefig('PressureTime #'+ str(bubbleNum))     #uncomment to save to external file
    
    print(max(press_bubNum))
    
    return


# In[70]:


numhold =0
bubPressureGauge(multiCont[numhold],times[numhold],72)


# In[81]:


aveP,distances = AvePressure(multiCont)

aveP = aveP*14.504 # multiply by 14.504 to convert from barg to psig 

plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
plt.scatter(distances*3.281,aveP,label='bubble dynamics dropoff',s=75)
#plt.plot(distances*3.281,14.405*(.5/(distances**2)),'r',label='squared distance dropoff model',linewidth='5')
#plt.legend()
plt.grid()
plt.xlabel('Distance (ft)',size=15); plt.ylabel('Pressure (psig)',size=15)
plt.title('Average Pressures vs. Distance')
plt.ylim([0,40])

    #plt.savefig('AveragePressure')     #uncomment to save to external file


# In[80]:


plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
plt.scatter(distances,aveP,label='bubble dynamics dropoff',s=75)
plt.plot(distances,14.405*(3/(distances**2)),'r',label='squared distance dropoff model',linewidth='5')
plt.legend()
plt.grid()
plt.xlabel('Distance (m)',size=15); plt.ylabel('Pressure (psig)',size=15)
plt.title('Average Pressures vs. Distance')
plt.ylim([0,40])


# In[6]:





# In[ ]:





# In[ ]:




