# bubbledynamics_streamlit.py
# Streamlit front-end for the GivenPressure_Bubble_Dynamics simulation
# Converted from GivenPressure_Bubble_Dynamics_NA_V1.ipynb (full notebook logic preserved)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import copy
import math
from tqdm import tqdm
import threading
import time
import io
from matplotlib.animation import FuncAnimation, PillowWriter

# --------------------------
# Global constants (same as notebook)
# --------------------------
gasConstant = 8.31446 # J/mol/K
refTemp = 298.15 # K
refPressure = 1E5 # Pa
refGamma = 1.4
airDensity = 1.22
airMW = 0.029

# --------------------------
# Bubble class (copied from notebook)
# --------------------------
class Bubble:
    """This class represents a bubble of gas."""
    epsln = 20 # Lennard Jones potential epsilon (repulsive)

    @staticmethod
    def interBubbleForce(sigma,r):
        return 24*Bubble.epsln/r*(sigma/r)**6

    @staticmethod
    def EOS(moles,temperature,dia):
        return moles*gasConstant*temperature/(np.pi/6*dia**3)

    @staticmethod
    def EOSDerivs(moles,b):
        # return dTdt given dnUdt and moles
        dTdt = (b.derivs['dnUdt'] - gasConstant/(b.gamma-1)*b.derivs['dndt'])*(b.gamma-1)/(gasConstant*moles)
        return dTdt

    @classmethod
    def mach(cls, upstreamPressure, downstreamPressure, gamma):
        if (upstreamPressure >= downstreamPressure):
           ratio = upstreamPressure/downstreamPressure
           multiplier = -1.
        else:
            ratio = downstreamPressure/upstreamPressure
            multiplier = 1.
        mach_abs = np.sqrt(2/(gamma-1)*(ratio**((gamma-1)/gamma)-1))
        return min(1.,mach_abs)*multiplier

    def __init__(self, gamma=1.4, Mw=0.029, pos=[0,0,0], temperature=refTemp, pressure=refPressure, size=0.01, dragC=0.5, heatPoint = [0,0,0], force = [0,0,0]):
        self.force=force
        self.gamma = gamma
        self.Mw = Mw
        self.pos = np.array(pos, dtype=float)
        self.temperature = temperature
        self.pressure = pressure
        self.size = size
        self.dragC = dragC
        self.heatPoint = np.array(heatPoint, dtype=float)

        # velocities, moles, mass
        ar = np.random.rand(3)*2-1
        mag = np.sqrt(np.dot(ar,ar))
        ar = ar/mag
        self.vel = np.dot(np.sqrt(self.gamma*gasConstant*temperature/Mw),ar)
        self.moles = (4/3*np.pi*(self.size/2)**3)*pressure/(gasConstant*temperature)
        self.mass = self.moles*self.Mw

        # neighborhood and bookkeeping
        self.neighbors = []
        self.posp = copy.copy(self.pos)
        self.derivs = {'dndt':0,'dnUdt':0}
        self.posK = [0]*4; self.velK = [0]*4; self.nK = [0]*4; self.TK = [0]*4
        self.center = False; self.heatedH = 0

    def dndt(self,moles,temperature,downstreamPressure,area):
        pressure = Bubble.EOS(moles=moles,temperature=temperature,dia=self.size)
        gamma=self.gamma; Mw=self.Mw
        MachNu = Bubble.mach(pressure,downstreamPressure,self.gamma)
        if MachNu>0:
            pressure = refPressure; temperature = refTemp; gamma = refGamma; Mw = airMW
        return pressure*area*np.sqrt(gamma/(gasConstant*temperature*Mw))*MachNu*(1+(gamma-1)/2*MachNu**2)**((gamma+1)/(2-2*gamma))

    def dnUdt(self,dndt,temperature):
        gamma = self.gamma
        if dndt>0:
            gamma = refGamma; temperature = refTemp
        return dndt*gamma*gasConstant/(gamma-1)*(temperature-refTemp)

    def derivsCalc(self, pos, vel, moles, temperature, time, c, actual, heatPoint):
        posRelaxT = 0.0025
        intRelaxT = posRelaxT+0.0025

        # container wall repulsion
        d2 = np.subtract(pos,c.dimensions)
        accelx = Container.wallForce(self.size,pos[0]) + Container.wallForce(self.size,d2[0])
        accely = Container.wallForce(self.size,pos[1]) + Container.wallForce(self.size,d2[1])
        accelz = Container.wallForce(self.size,pos[2]) + Container.wallForce(self.size,d2[2])
        accel = np.divide([accelx,accely,accelz],moles*self.Mw)

        # heated bubble heating event
        if time > posRelaxT:
            if self.center:
                sigma = c.burnRateMultiplier*(3e-4); ave = c.burnRateMultiplier*(1e-3)
                self.derivs['dnUdt'] += self.heatedH/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((time-ave)/sigma)**2)

        # venting through walls (approx)
        r1 = pos[0]-c.dimensions[0]; r2 = pos[1]-c.dimensions[1]; r3 = pos[2]-c.dimensions[2]
        arp = (self.size/2)**2
        ventAreaWall = np.pi*(max([arp - r1**2,0]) +  max([arp - r2**2,0]) + max([arp - r3**2,0]) + max([arp - pos[0]**2,0]) + max([arp - pos[1]**2,0]) + max([arp - pos[2]**2,0]))
        if ventAreaWall > 0 and c.allowBubbleInteractions:
            dndtWall = self.dndt(moles=moles,temperature=temperature,downstreamPressure=refPressure,area=ventAreaWall)
            self.derivs['dndt'] += dndtWall
            self.derivs['dnUdt'] += self.dnUdt(dndtWall,temperature)

        # interactions with neighbors
        for nbub in self.neighbors:
            dist = np.linalg.norm(pos - nbub.pos)
            if dist == 0:
                continue
            msize = (self.size + nbub.size)/2
            self.force = Bubble.interBubbleForce(msize,dist)*np.subtract(pos,nbub.pos)/dist

            if self.center:
                heatPointDist = np.linalg.norm(pos-heatPoint)
                if heatPointDist != 0:
                    self.force += (1750*heatPointDist)*np.subtract(heatPoint,pos)/heatPointDist

            accel = np.add(accel,self.force/(moles*self.Mw))
            ventArea = np.pi/(4*dist**2)*(4*(dist*self.size/2)**2 - (dist**2 - nbub.size**2 + self.size**2)**2)
            if ventArea > 0 and c.allowBubbleInteractions:
                pressure = Bubble.EOS(moles,temperature,self.size)
                if time > intRelaxT:
                    if pressure > nbub.pressure:
                        dndt_i_j = self.dndt(moles=moles,temperature=temperature,downstreamPressure=nbub.pressure,area=ventArea)
                        self.derivs['dndt'] += dndt_i_j
                        nbub.derivs['dndt'] += -dndt_i_j
                        dnUdt_i_j = self.dnUdt(dndt=dndt_i_j,temperature=temperature)
                        self.derivs['dnUdt'] += dnUdt_i_j
                        nbub.derivs['dnUdt'] += -dnUdt_i_j

        c.stat['pv'] += moles*self.Mw*np.dot(vel,vel)
        c.stat['fv'] += moles*self.Mw*np.dot(accel,vel)
        return accel

# --------------------------
# Container class (copied from notebook)
# --------------------------
class Container:
    @staticmethod
    def wallForce(sig,dis):
        return 24*Bubble.epsln/(dis*2)*(sig/(dis*2))**6

    def __init__(self, bubbleCount=1, bubbleDiameter=0.05, xlen=0.5, ylen=0.5, zlen=0.25, heatPoint = [0.25,0.25,0.025], finalIsoPressure=refPressure*8*2, allowBubbleInteractions=True, burnRateMultiplier=1):
        self.heatPoint = np.array(heatPoint, dtype=float)
        self.bubbleCount = bubbleCount
        slotWidth = bubbleDiameter
        bubVol = (np.pi/6*bubbleDiameter**3)
        self.size = slotWidth
        self.centerCoords=[xlen/2,ylen/2,zlen/2]
        self.dimensions = [xlen,ylen,zlen]
        self.vol = xlen*ylen*zlen
        self.rbuff = slotWidth*1.5
        self.allowBubbleInteractions = allowBubbleInteractions
        self.burnRateMultiplier = burnRateMultiplier

        # place bubbles in grid
        self.bubbles = []; count = 0; flag = True
        maxbubbles = math.floor(xlen/bubbleDiameter)*math.floor(ylen/bubbleDiameter)*math.floor(zlen/bubbleDiameter)
        if bubbleCount > maxbubbles:
            raise ValueError("Too many bubbles or bubble diameter is too large!")

        for i in range(math.floor(xlen/bubbleDiameter)):
            for j in range(math.floor(ylen/bubbleDiameter)):
                for k in range(math.floor(zlen/bubbleDiameter)):
                    pressure = refPressure
                    temperature = refTemp
                    bubble = Bubble(pos=np.multiply([i+0.5,j+0.5,k+0.5],slotWidth), temperature=temperature, pressure=pressure, size=slotWidth, heatPoint=self.heatPoint)
                    self.bubbles.append(bubble)
                    count += 1
                    if len(self.bubbles) >= bubbleCount:
                        break
                if len(self.bubbles) >= bubbleCount:
                    break
            if len(self.bubbles) >= bubbleCount:
                break

        heatedBub = min([[np.linalg.norm(bub.pos - heatPoint),ID] for ID,bub in enumerate(self.bubbles)])[1]
        self.heatedBub = heatedBub
        self.bubbles[heatedBub].center = True
        self.bubbles[heatedBub].heatedH = (finalIsoPressure-refPressure)*bubVol/(self.bubbles[heatedBub].gamma-1)

        self.stat = {'pv':0,'fv':0}
        self.history = []
        self.fnab()

    def zero(self):
        for bub in self.bubbles: bub.derivs = {'dndt':0,'dnUdt':0,'dVdt':0}
        self.stat = {'pv':0,'fv':0}
        return None

    def fnab(self):
        upb = len(self.bubbles)
        for bub in self.bubbles:
            bub.neighbors = []
            bub.posp = copy.copy(bub.pos)
        for i,bubi in enumerate(self.bubbles):
            for j in range(i+1,upb):
                dist = np.linalg.norm(bubi.pos - self.bubbles[j].pos)
                if dist <= self.rbuff:
                    bubi.neighbors.append(self.bubbles[j])
                    self.bubbles[j].neighbors.append(bubi)
        return None

    def copies(self):
        snapshot = {'moles':[copy.copy(each.moles) for each in self.bubbles],
        'pressure':[copy.copy(each.pressure) for each in self.bubbles],'temperature':[copy.copy(each.temperature) for each in self.bubbles],
        'vel':[copy.copy(each.vel) for each in self.bubbles],'pos':[copy.copy(each.pos) for each in self.bubbles],
        'containerTemperature': sum([np.dot(bub.vel,bub.vel)*bub.Mw/gasConstant/bub.gamma for bub in self.bubbles])/len(self.bubbles), 'force':[copy.copy(each.force) for each in self.bubbles]}
        self.history.append(snapshot)
        return None

# --------------------------
# Simulation function (Runge-Kutta) (copied)
# --------------------------
delt = 1E-4
tSteps = 1000

def simulate(container,dt=0.01,steps=1000, progress_callback=None, stop_event=None):
    times = [0]
    container.copies()

    for j in range(steps):
        if stop_event is not None and stop_event.is_set():
            # user requested stop
            break

        container.zero()
        # k1
        for bub in container.bubbles:
            bub.velK[0] = np.multiply(bub.derivsCalc(pos=bub.pos,vel=bub.vel,c=container,moles=bub.moles,temperature=bub.temperature,time=times[-1],actual=True,heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            if container.stat['pv'] != 0:
                bub.velK[0] -= container.stat['fv']/container.stat['pv']*bub.vel*dt
            bub.posK[0] = np.multiply(bub.vel,dt)
            dTdt = Bubble.EOSDerivs(moles=bub.moles,b=bub)
            bub.nK[0] = bub.derivs['dndt']*dt
            bub.TK[0] = dTdt*dt

        container.zero()
        # k2
        for bub in container.bubbles:
            bub.velK[1] = np.multiply(bub.derivsCalc(pos=np.add(bub.pos,0.5*bub.posK[0]),vel=np.add(bub.vel,0.5*bub.velK[0]),moles=bub.moles+0.5*bub.nK[0],temperature=bub.temperature+0.5*bub.TK[0],c=container,time=times[-1],actual=False,heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            if container.stat['pv'] != 0:
                bub.velK[1] -= container.stat['fv']/container.stat['pv']*np.add(bub.vel,0.5*bub.velK[0])*dt
            bub.posK[1] = np.add(bub.vel,bub.velK[0]*0.5)*dt
            dTdt = Bubble.EOSDerivs(moles=bub.moles+0.5*bub.nK[0],b=bub)
            bub.nK[1] = bub.derivs['dndt']*dt
            bub.TK[1] = dTdt*dt

        container.zero()
        # k3
        for bub in container.bubbles:
            bub.velK[2] = np.multiply(bub.derivsCalc(pos=np.add(bub.pos,0.5*bub.posK[1]),vel=np.add(bub.vel,0.5*bub.velK[1]),moles=bub.moles+0.5*bub.nK[1],temperature=bub.temperature+0.5*bub.TK[1],c=container,time=times[-1],actual=False, heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            if container.stat['pv'] != 0:
                bub.velK[2] -= container.stat['fv']/container.stat['pv']*np.add(bub.vel,0.5*bub.velK[1])*dt
            bub.posK[2] = np.add(bub.vel,bub.velK[1]*0.5)*dt
            dTdt = Bubble.EOSDerivs(moles=bub.moles+0.5*bub.nK[1],b=bub)
            bub.nK[2] = bub.derivs['dndt']*dt
            bub.TK[2] = dTdt*dt

        container.zero()
        # k4
        for bub in container.bubbles:
            bub.velK[3] = np.multiply(bub.derivsCalc(pos=np.add(bub.pos,bub.posK[2]),vel=np.add(bub.vel,bub.velK[2]),moles=bub.moles+bub.nK[2],temperature=bub.temperature+bub.TK[2],c=container,time=times[-1],actual=False, heatPoint=bub.heatPoint),dt)
        for bub in container.bubbles:
            if container.stat['pv'] != 0:
                bub.velK[3] -= container.stat['fv']/container.stat['pv']*np.add(bub.vel,bub.velK[2])*dt
            bub.posK[3] = np.add(bub.vel,bub.velK[2])*dt
            dTdt = Bubble.EOSDerivs(moles=bub.moles+bub.nK[2],b=bub)
            bub.nK[3] = bub.derivs['dndt']*dt
            bub.TK[3] = dTdt*dt

            bub.vel += 1/6*(bub.velK[0]+2*bub.velK[1]+2*bub.velK[2]+bub.velK[3])
            bub.pos += 1/6*(bub.posK[0]+2*bub.posK[1]+2*bub.posK[2]+bub.posK[3])
            bub.temperature += 1/6*(bub.TK[0]+2*bub.TK[1]+2*bub.TK[2]+bub.TK[3])
            bub.moles += 1/6*(bub.nK[0]+2*bub.nK[1]+2*bub.nK[2]+bub.nK[3])
            bub.pressure = Bubble.EOS(moles=bub.moles,temperature=bub.temperature,dia=bub.size)

        container.zero()
        if max([np.linalg.norm(bub.pos - bub.posp) for bub in container.bubbles])>container.rbuff*0.4:
            container.fnab()

        times.append(dt+times[j])
        container.copies()

        # call progress callback if provided
        if progress_callback is not None:
            progress_callback((j+1)/steps)

    return times

# --------------------------
# Graphing / analysis functions (copied)
# --------------------------
def PvsPos(cont,timestep):
    pressure = []; spacebetween = []
    centerpos = cont.history[timestep]['pos'][cont.heatedBub]
    for each in range(len(cont.bubbles)):
        bubblepos = cont.history[timestep]['pos'][each]
        spacebetween.append(np.linalg.norm(centerpos-bubblepos))
        pressure.append(cont.history[timestep]['pressure'][each])
    pressure = np.array(pressure)/1e5 - 1
    arr = np.array([spacebetween,pressure])
    sarr = arr[:,arr[0].argsort()]
    cs = interp1d(sarr[0],sarr[1], bounds_error=False, fill_value=(sarr[1,0], sarr[1,-1]))
    return (sarr,cs)

def MaxPressure(mCont, graphDist, tSteps_local):
    distances = np.linspace(0.001,graphDist,100)
    maxP = np.zeros(len(distances))
    for j in range(len(mCont)):
        for i in range(1,tSteps_local):
            out = PvsPos(mCont[j],i)
            Ps = out[1](distances)
            for idx,p in enumerate(Ps):
                if p > maxP[idx]:
                    maxP[idx] = p
    return maxP,distances

def GPressure(cont,graphDist,tSteps_local):
    distances = np.linspace(0.001,graphDist,100)
    maxP = np.zeros(len(distances))
    for i in range(1,tSteps_local):
        out = PvsPos(cont,i)
        Ps = out[1](distances)
        for idx,p in enumerate(Ps):
            if p > maxP[idx]:
                maxP[idx] = p
    plt.scatter(distances,maxP,label='bubble dynamics dropoff')
    plt.plot(distances,0.05/(distances**2),'r',label='squared distance dropoff model')
    plt.legend()
    plt.xlabel('distance'); plt.ylabel('Gauge Pressure')
    plt.ylim([0,2])
    plt.show()
    return

def singMaxPressure(cont,graphDist,tSteps_local):
    distances = np.linspace(0.001,graphDist,100)
    maxP = np.zeros(len(distances))
    for i in range(1,tSteps_local):
        out = PvsPos(cont,i)
        Ps = out[1](distances)
        for idx,p in enumerate(Ps):
            if p > maxP[idx]:
                maxP[idx] = p
    maxP = maxP*14.405
    fig = plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.scatter(distances*3.281,maxP,label='bubble dynamics dropoff',s=75)
    plt.plot(distances*3.281,14.405*(0.05/(distances**2)),'r',label='squared distance dropoff model',linewidth='5')
    plt.legend()
    plt.xlabel('Distance (ft)',size=15); plt.ylabel('Pressure (psig)',size=15)
    plt.title('Max Pressure of Single Simulation')
    plt.ylim([0,40])
    plt.grid()
    return fig

def D3Graph(cont,position=21):
    x = [each['pos'][position][0] for i,each in enumerate(cont.history)]
    y = [each['pos'][position][1] for i,each in enumerate(cont.history)]
    z = [each['pos'][position][2] for i,each in enumerate(cont.history)]
    return x,y,z

def bubPressure(cont,times,bubbleNum=21):
    press_bubNum = [each['pressure'][bubbleNum]/6895 for i,each in enumerate(cont.history)]
    fig = plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
    plt.plot(times,press_bubNum,'firebrick',linewidth = 2)
    plt.grid()
    plt.ylim([14,100])
    plt.xlabel('Time (s)',size=12); plt.ylabel('Pressure (psi)',size=12)
    plt.title('Pressure Vs. Time for Heated Bubble')
    return fig

def AvePressure(mCont, graphDist, tSteps_local):
    distances = np.linspace(0.001,graphDist,100)
    aveP = np.zeros(len(distances))
    for j in range(len(mCont)):
        maxP = np.zeros(len(distances))
        for i in range(1,tSteps_local):
            out = PvsPos(mCont[j],i)
            Ps = out[1](distances)
            for idx,p in enumerate(Ps):
                if p > maxP[idx]:
                    maxP[idx] = p
        aveP += maxP
    aveP = aveP/len(mCont)
    return aveP, distances

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(layout="wide", page_title="Bubble Dynamics Simulator")
st.title("Bubble Dynamics Simulator — Streamlit")

# Sidebar inputs (mirrors original Tk inputs)
st.sidebar.header("Simulation Controls")

n_simulations = st.sidebar.number_input("How many simulations do you want to run:", min_value=1, max_value=10, value=1, step=1)
maxP_input = st.sidebar.number_input("What is the max pressure (Psi):", value=10.0, step=1.0, format="%.3f")
bubbleDiameterSet = st.sidebar.number_input("What is the characteristic length of your container (m):", value=0.05, step=0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("Choose which graphs you want to print:")
CB1_MPS = st.sidebar.checkbox("Max Pressure of All Simulations", value=True)
CB2_MPSS = st.sidebar.checkbox("Max Pressure of a Single Simulation", value=False)
CB3_HB_PA = st.sidebar.checkbox("Heated Bubble - Position Animation", value=False)
CB4_HB_PvT = st.sidebar.checkbox("Heated Bubble - Pressure vs. Time", value=True)
CB8_BRM = st.sidebar.checkbox("Average Pressure vs. Distance", value=False)
CB7_TNT = st.sidebar.checkbox("TNT Equivalency", value=False)

# Derived values
xlenCalc = bubbleDiameterSet*10
ylenCalc = bubbleDiameterSet*10
zlenCalc = bubbleDiameterSet*5
maxBubbles = math.floor(xlenCalc/bubbleDiameterSet)*math.floor(ylenCalc/bubbleDiameterSet)*math.floor(zlenCalc/bubbleDiameterSet)
heatPointSet = [xlenCalc/2,ylenCalc/2,bubbleDiameterSet/2]

# compute inputText2 polynomial from original code
inputText2 = 2.4067243761E-10*maxP_input**5 - 1.1483532493E-07*maxP_input**4 + 2.0795527805E-05*maxP_input**3 - 1.7592839788E-03*maxP_input**2 + 7.6402081420E-02*maxP_input + 5.9694885897E-01

# simulation parameters
delt = 1E-4
tSteps_default = 1000

# place to show logs and progress
status_placeholder = st.empty()
progress_placeholder = st.empty()
plots_col = st.columns(2)

# run/stop controls using session_state to lock UI
if 'running' not in st.session_state:
    st.session_state.running = False
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'results' not in st.session_state:
    st.session_state.results = None

start_button = st.sidebar.button("Start Simulation")
stop_button = st.sidebar.button("Stop Simulation")

# disable inputs while running by greying via info message
if st.session_state.running:
    st.sidebar.info("Simulation in progress — inputs are locked.")

# function to run simulations (in main streamlit thread; it updates progress via callback)
def run_simulations_and_collect(nSimulations, bubbleDiameterSet, xlenCalc, ylenCalc, zlenCalc, inputText2, steps, progress_bar_callback):
    multiCont = []
    times_list = []
    for i in range(nSimulations):
        if st.session_state.stop_event.is_set():
            break
        cont = Container(bubbleCount=math.floor((xlenCalc/bubbleDiameterSet)*(ylenCalc/bubbleDiameterSet)*(zlenCalc/bubbleDiameterSet)),
                         bubbleDiameter=bubbleDiameterSet, xlen=xlenCalc, ylen=ylenCalc, zlen=zlenCalc,
                         heatPoint=heatPointSet, finalIsoPressure=refPressure*8*2, allowBubbleInteractions=True, burnRateMultiplier=inputText2)
        multiCont.append(cont)
        # simulate with small dt to match notebook behavior
        times = simulate(cont, dt=delt, steps=steps, progress_callback=lambda v: progress_bar_callback((len(multiCont)-1 + v)/nSimulations), stop_event=st.session_state.stop_event)
        times_list.append(times)
    return multiCont, times_list

# helper to update progress bar in Streamlit (0..1)
def make_progress_callback(pb):
    def cb(value):
        try:
            pb.progress(min(1.0, float(value)))
        except Exception:
            pass
    return cb

# When start button pressed
if start_button and not st.session_state.running:
    st.session_state.stop_event.clear()
    st.session_state.running = True
    status_placeholder.info("Starting simulation(s)...")
    progress_bar = progress_placeholder.progress(0.0)
    steps = tSteps_default

    # Because Streamlit reruns the script on interaction, we run the heavy loop here (blocking).
    # This is acceptable for Streamlit usage; the UI stays responsive to Stop clicks because we check stop_event.
    try:
        multiCont, times = run_simulations_and_collect(n_simulations, bubbleDiameterSet, xlenCalc, ylenCalc, zlenCalc, inputText2, steps, make_progress_callback(progress_bar))
        st.session_state.results = {'multiCont': multiCont, 'times': times}
        status_placeholder.success("Simulation(s) finished.")
    except Exception as e:
        st.exception(e)
        st.session_state.results = None
    finally:
        st.session_state.running = False
        progress_bar.empty()

# When stop button pressed
if stop_button:
    st.session_state.stop_event.set()
    status_placeholder.warning("Stop requested — finishing current step and halting...")

# When there are results, plot according to checkboxes
if st.session_state.results is not None and not st.session_state.running:
    multiCont = st.session_state.results['multiCont']
    times = st.session_state.results['times']
    graphDist = 0.5*max(multiCont[0].dimensions) if len(multiCont)>0 else 0.5*max([xlenCalc,ylenCalc,zlenCalc])

    # Max Pressure of All Simulations
    if CB1_MPS:
        maxP, distances = MaxPressure(multiCont, graphDist, tSteps_default)
        maxP_plot = (maxP)*14.504
        fig1 = plt.figure()
        plt.scatter(distances*3.281,maxP_plot,label='bubble dynamics dropoff')
        plt.plot(distances*3.281, 14.405*(0.05/(distances**2)),'r',label='squared distance dropoff model')
        plt.legend(); plt.xlabel('Distance (ft)'); plt.ylabel('Pressure (psig)'); plt.ylim([0,40]); plt.title('Max Pressure of Simulations')
        st.pyplot(fig1)

    # Max Pressure of Single Simulation
    if CB2_MPSS:
        fig2 = singMaxPressure(multiCont[0], graphDist, tSteps_default)
        st.pyplot(fig2)

    # Heated Bubble - Position Animation
    if CB3_HB_PA:
        # attempt to generate animation gif for the heated bubble movement
        try:
            x,y,z = D3Graph(multiCont[0], position=multiCont[0].heatedBub)
            data = np.array([x,y,z])
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
            ax.set_xlim3d([0.0, multiCont[0].dimensions[0]])
            ax.set_ylim3d([0.0, multiCont[0].dimensions[1]])
            ax.set_zlim3d([0.0, multiCont[0].dimensions[2]])
            def animate(t):
                line.set_data(data[:2, :t])
                line.set_3d_properties(data[2,:t])
                return line,
            ani = FuncAnimation(fig, animate, frames=len(x), interval=30)
            buf = io.BytesIO()
            ani.save(buf, writer=PillowWriter(fps=30))
            buf.seek(0)
            st.image(buf)
        except Exception as e:
            st.warning("Animation could not be created: " + str(e))

    # Heated Bubble Pressure vs Time
    if CB4_HB_PvT:
        bubbleNum = multiCont[0].heatedBub
        fig3 = bubPressure(multiCont[0], times[0], bubbleNum)
        st.pyplot(fig3)

    # Average Pressures vs Distance
    if CB8_BRM:
        aveP, distances = AvePressure(multiCont, graphDist, tSteps_default)
        aveP = aveP*14.504
        fig4 = plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
        plt.scatter(distances*3.281,aveP,label='bubble dynamics dropoff',s=75)
        plt.plot(distances*3.281,14.405*(0.05/(distances**2)),'r',label='squared distance dropoff model',linewidth='5')
        plt.legend(); plt.grid(); plt.xlabel('Distance (ft)'); plt.ylabel('Pressure (psig)'); plt.title('Average Pressures vs. Distance'); plt.ylim([0,40])
        st.pyplot(fig4)

    # TNT Equivalency
    if CB7_TNT:
        aveP, distances = AvePressure(multiCont, graphDist, tSteps_default)
        aveP = aveP*14.504
        heated_h_value = None
        for bubble in multiCont[0].bubbles:
            if bubble.center:
                heated_h_value = bubble.heatedH
                break
        if heated_h_value is None:
            st.warning("Could not find heated bubble")
        else:
            mass = heated_h_value / 10_000
            Z2 = (distances*3.281)/((mass/453.6)**(1/3))
            Z = np.array([0.1,1,5,10,50,100])
            U = np.array([10000,1000,30,7,0.7,0.3])
            fig5 = plt.figure(facecolor=(1, 1, 1),constrained_layout=True)
            plt.plot(Z,U,'firebrick',label='Standard TNT',linewidth = 2)
            plt.plot(Z2[1:],aveP[1:],'darkblue',label=f'Burn Rate Multiplier {inputText2}',linewidth = 2)
            plt.xlabel(r'$Z = R/W^{1/3}$ ($ft/lbm^{1/3}$)',size=12)
            plt.ylabel('$P_{so}$ (psig)',size=12)
            plt.title('TNT Equivalency'); plt.xscale('log'); plt.yscale('log'); plt.legend(); plt.grid()
            st.pyplot(fig5)
