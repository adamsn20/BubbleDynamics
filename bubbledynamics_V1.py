import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import copy, math

#---------------------------------------------------------------------------------------------#
#------------------------------------ Global Constants --------------------------------------#
#---------------------------------------------------------------------------------------------#
gasConstant = 8.31446
refTemp = 298.15
refPressure = 1E5
refGamma = 1.4
airDensity = 1.22
airMW = 0.029

#---------------------------------------------------------------------------------------------#
#------------------------------------ Bubble Class ------------------------------------------#
#---------------------------------------------------------------------------------------------#
class Bubble:
    epsln = 20

    @staticmethod
    def interBubbleForce(sigma, r):
        return 24 * Bubble.epsln / r * (sigma / r)**6

    @staticmethod
    def EOS(moles, temperature, dia):
        return moles * gasConstant * temperature / (np.pi / 6 * dia**3)

    @staticmethod
    def EOSDerivs(moles, b):
        dTdt = (b.derivs['dnUdt'] - gasConstant/(b.gamma-1)*b.derivs['dndt'])*(b.gamma-1)/(gasConstant*moles)
        return dTdt

    @classmethod
    def mach(cls, upstreamPressure, downstreamPressure, gamma):
        if upstreamPressure >= downstreamPressure:
            ratio = upstreamPressure / downstreamPressure
            multiplier = -1.
        else:
            ratio = downstreamPressure / upstreamPressure
            multiplier = 1.
        mach_abs = np.sqrt(2/(gamma-1)*(ratio**((gamma-1)/gamma)-1))
        return min(1., mach_abs) * multiplier

    def __init__(self, gamma=1.4, Mw=0.029, pos=[0,0,0], temperature=refTemp, pressure=refPressure, size=0.01, dragC=0.5, heatPoint=[0,0,0], force=[0,0,0]):
        self.force = force
        self.gamma = gamma
        self.Mw = Mw
        self.pos = np.array(pos)
        self.temperature = temperature
        self.pressure = pressure
        self.size = size
        self.dragC = dragC
        self.heatPoint = heatPoint

        ar = np.random.rand(3)*2-1
        ar /= np.linalg.norm(ar)
        self.vel = np.sqrt(self.gamma*gasConstant*temperature/Mw) * ar
        self.moles = (4/3*np.pi*(self.size/2)**3)*pressure/(gasConstant*temperature)
        self.mass = self.moles * self.Mw
        self.neighbors = []
        self.posp = copy.copy(self.pos)
        self.derivs = {'dndt':0, 'dnUdt':0}
        self.posK = [0]*4; self.velK = [0]*4; self.nK = [0]*4; self.TK = [0]*4
        self.center = False; self.heatedH = 0

    def dndt(self, moles, temperature, downstreamPressure, area):
        pressure = Bubble.EOS(moles, temperature, self.size)
        gamma = self.gamma
        Mw = self.Mw
        MachNu = Bubble.mach(pressure, downstreamPressure, gamma)
        if MachNu > 0:
            pressure = refPressure; temperature = refTemp; gamma = refGamma; Mw = airMW
        return pressure * area * np.sqrt(gamma/(gasConstant*temperature*Mw)) * MachNu * (1+(gamma-1)/2*MachNu**2)**((gamma+1)/(2-2*gamma))

    def dnUdt(self, dndt, temperature):
        gamma = self.gamma
        if dndt > 0:
            gamma = refGamma; temperature = refTemp
        return dndt * gamma * gasConstant / (gamma-1) * (temperature - refTemp)

    def derivsCalc(self, pos, vel, moles, temperature, time, c, actual, heatPoint):
        posRelaxT = 0.0025
        intRelaxT = posRelaxT + 0.0025

        d2 = np.subtract(pos, c.dimensions)
        accelx = Container.wallForce(self.size, pos[0]) + Container.wallForce(self.size, d2[0])
        accely = Container.wallForce(self.size, pos[1]) + Container.wallForce(self.size, d2[1])
        accelz = Container.wallForce(self.size, pos[2]) + Container.wallForce(self.size, d2[2])
        accel = np.divide([accelx, accely, accelz], moles * self.Mw)

        if time > posRelaxT:
            if self.center:
                sigma = c.burnRateMultiplier*(3e-4)
                ave = c.burnRateMultiplier*(1e-3)
                self.derivs['dnUdt'] += self.heatedH/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((time-ave)/sigma)**2)

        for nbub in self.neighbors:
            dist = np.linalg.norm(pos - nbub.pos)
            msize = (self.size + nbub.size)/2
            self.force = Bubble.interBubbleForce(msize, dist) * (pos - nbub.pos) / dist

            if self.center:
                heatPointDist = np.linalg.norm(pos - heatPoint)
                self.force += (1750 * heatPointDist) * (heatPoint - pos) / heatPointDist

            accel += self.force / (moles * self.Mw)
        return accel

#---------------------------------------------------------------------------------------------#
#------------------------------------ Container Class ----------------------------------------#
#---------------------------------------------------------------------------------------------#
class Container:
    @staticmethod
    def wallForce(sig, dis):
        return 24*Bubble.epsln/(dis*2)*(sig/(dis*2))**6

    def __init__(self, bubbleCount, bubbleDiameter, xlen, ylen, zlen, heatPoint, finalIsoPressure, allowBubbleInteractions=True, burnRateMultiplier=1):
        self.heatPoint = heatPoint
        self.bubbleCount = bubbleCount
        self.size = bubbleDiameter
        self.centerCoords = [xlen/2, ylen/2, zlen/2]
        self.dimensions = [xlen, ylen, zlen]
        self.vol = xlen*ylen*zlen
        self.rbuff = bubbleDiameter * 1.5
        self.allowBubbleInteractions = allowBubbleInteractions
        self.burnRateMultiplier = burnRateMultiplier

        self.bubbles = []
        for i in range(math.floor(xlen/bubbleDiameter)):
            for j in range(math.floor(ylen/bubbleDiameter)):
                for k in range(math.floor(zlen/bubbleDiameter)):
                    if len(self.bubbles) >= bubbleCount:
                        break
                    bubble = Bubble(pos=np.multiply([i+0.5,j+0.5,k+0.5], bubbleDiameter), heatPoint=self.heatPoint)
                    self.bubbles.append(bubble)
        heatedBub = min([[np.linalg.norm(bub.pos - heatPoint), ID] for ID, bub in enumerate(self.bubbles)])[1]
        self.heatedBub = heatedBub
        self.bubbles[heatedBub].center = True
        bubVol = (np.pi/6*bubbleDiameter**3)
        self.bubbles[heatedBub].heatedH = (finalIsoPressure - refPressure)*bubVol/(self.bubbles[heatedBub].gamma-1)
        self.stat = {'pv':0,'fv':0}
        self.history = []
        self.fnab()

    def zero(self):
        for bub in self.bubbles:
            bub.derivs = {'dndt':0, 'dnUdt':0}
        self.stat = {'pv':0,'fv':0}

    def fnab(self):
        upb = len(self.bubbles)
        for bub in self.bubbles:
            bub.neighbors = []
            bub.posp = copy.copy(bub.pos)
        for i, bubi in enumerate(self.bubbles):
            for j in range(i+1, upb):
                dist = np.linalg.norm(bubi.pos - self.bubbles[j].pos)
                if dist <= self.rbuff:
                    bubi.neighbors.append(self.bubbles[j])
                    self.bubbles[j].neighbors.append(bubi)

    def copies(self):
        snapshot = {
            'pressure':[copy.copy(each.pressure) for each in self.bubbles],
            'pos':[copy.copy(each.pos) for each in self.bubbles],
        }
        self.history.append(snapshot)

#---------------------------------------------------------------------------------------------#
#------------------------------------ Simulation Function ------------------------------------#
#---------------------------------------------------------------------------------------------#
def simulate(container, dt=1E-4, steps=500, stop_flag=None, progress_bar=None):
    times = [0]
    container.copies()
    for j in range(steps):
        if stop_flag and stop_flag.is_set():
            st.warning("Simulation stopped by user.")
            break
        container.zero()
        for bub in container.bubbles:
            accel = bub.derivsCalc(bub.pos, bub.vel, bub.moles, bub.temperature, times[-1], container, True, bub.heatPoint)
            bub.vel += accel * dt
            bub.pos += bub.vel * dt
        container.copies()
        times.append(dt + times[-1])
        if progress_bar:
            progress_bar.progress(j/steps)
    return times

#---------------------------------------------------------------------------------------------#
#------------------------------------ Streamlit Interface ------------------------------------#
#---------------------------------------------------------------------------------------------#
st.title("💧 Bubble Interaction Simulation")

with st.sidebar:
    st.header("Simulation Inputs")
    n_sim = st.number_input("Number of Simulations", min_value=1, max_value=10, value=1)
    maxP = st.number_input("Max Pressure (psi)", min_value=1.0, value=10.0)
    bubbleDiameterSet = st.number_input("Characteristic length of container (m)", min_value=0.01, value=0.05)
    run_btn = st.button("Run Simulation")
    stop_btn = st.button("Stop Simulation")

# Derived values
xlenCalc = bubbleDiameterSet*10
ylenCalc = bubbleDiameterSet*10
zlenCalc = bubbleDiameterSet*5
maxBubbles = math.floor(xlenCalc/bubbleDiameterSet)*math.floor(ylenCalc/bubbleDiameterSet)*math.floor(zlenCalc/bubbleDiameterSet)
heatPointSet = [xlenCalc/2, ylenCalc/2, bubbleDiameterSet/2]

if "stop_flag" not in st.session_state:
    import threading
    st.session_state.stop_flag = threading.Event()

if run_btn:
    st.session_state.stop_flag.clear()
    progress_bar = st.progress(0)
    cont = Container(bubbleCount=min(10, maxBubbles), bubbleDiameter=bubbleDiameterSet, xlen=xlenCalc, ylen=ylenCalc, zlen=zlenCalc,
                     heatPoint=heatPointSet, finalIsoPressure=refPressure*8*2)
    with st.spinner("Running simulation..."):
        times = simulate(cont, dt=1E-4, steps=300, stop_flag=st.session_state.stop_flag, progress_bar=progress_bar)
    st.success("Simulation complete!")

    pressures = [np.mean([p for p in snap['pressure']]) for snap in cont.history]
    plt.plot(times, pressures)
    plt.xlabel("Time (s)")
    plt.ylabel("Average Pressure (Pa)")
    st.pyplot(plt)

if stop_btn:
    st.session_state.stop_flag.set()
