import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Example Diffusion Simulator')
st.markdown('This is a simple example of a diffusion simulator. The user can \
select the number of random walks in the simulation using the slider below. The total number of \
    steps in each walk is fixed at 100.')

val = st.slider('Select a value for the number of random walks', 20, 200, 100)

# Create the simulation with the provided slider value
nsteps = 100; nwalks = val
sqdists = np.zeros((nwalks,nsteps))
for i in range(nwalks):
    #set parameters for 2-D random walk
    x = 0; y = 0; z = 0 #initial x,y, and z coordinates
    x_loc = [x]; y_loc = [y]; z_loc = [z]
    for j in range(nsteps):
        # for each step, the x or y coordinates change by 1 or -1 depending on the random number
        direction = np.random.choice([-1,0,1])
        if direction == -1:
            x += np.random.choice([-1,1])
        elif direction == 1:
            y += np.random.choice([-1,1])
        else:
            z += np.random.choice([-1,1])
        sqdists[i,j] = (x**2 + y**2 + z**2) #distance from origin
        x_loc.append(x); y_loc.append(y); z_loc.append(z)


#format the text to a smaller size
plt.rc('font', size=8)
#plot one of the simulation runs (the last one)
fig, ax = plt.subplots(1,2,figsize=(8,4))
ax[0].plot(x_loc,y_loc)
ax[0].plot(x_loc[0],y_loc[0],'go')
ax[0].plot(x_loc[-1],y_loc[-1],'ro')
ax[0].set_xlabel('x position'); ax[0].set_ylabel('y position')
ax[0].legend(['Random Walk','Start','End'])
ax[1].plot(z_loc,y_loc)
ax[1].plot(z_loc[0],y_loc[0],'go')
ax[1].plot(z_loc[-1],y_loc[-1],'ro')
ax[1].set_xlabel('z position'); ax[1].set_ylabel('y position')
ax[1].legend(['Random Walk','Start','End'])
st.pyplot(fig)


#calculate the average squared distance from the origin
avg_sqdist = np.mean(sqdists, axis=0)
df = pd.DataFrame({'Number of steps': range(1,nsteps+1),
                   'Average squared distance': avg_sqdist})

#fit linear model to the average squared distance using numpy with an intercept of 0
x = np.array([i for i in range(1,nsteps+1)])
y = avg_sqdist
x = x[:,np.newaxis] #change to column index
a, _, _, _ = np.linalg.lstsq(x, y,rcond=-1) #fit linear model with zero intercept and machine precision tolerance
df['Trend'] = a*x

#plot the average squared distance from the origin
fig, ax = plt.subplots()
ax.plot(df['Number of steps'], df['Average squared distance'], label='Average Squared Distance')
ax.plot(df['Number of steps'], df['Trend'], linestyle='--',label='Linear Fit')
ax.set_xlabel('Number of steps'); ax.set_ylabel('Average squared distance')
ax.legend()
ax.set_title('Average Squared Distance from Origin')
st.pyplot(fig)

st.write('The diffusion coefficient can be estimated from the slope of the line given a step size scaled to \
    the mean free path in that system. The slope of the line is:',a)
 
st.write('The code above is a simple example of a diffusion simulator. The user can select the number of random walks \
 in the simulation using the slider. The code generates a 3D random walk for the selected number of steps and plots \
    the walk in 2D.')

st.markdown('### Code')
st.write('Code used for the random walk. You can see that the direction is first picked from a random uniform distribution \
    and then the step is taken of size = 1 from the same distribution.)')
#showcode = st.checkbox('Show code') #recalculates the entire code when this changes state

st.code(f"""
# Create the simulation with the provided slider value
nsteps = 100; nwalks = val
sqdists = np.zeros((nwalks,nsteps))
for i in range(nwalks):
    #set parameters for 2-D random walk
    x = 0; y = 0; z = 0 #initial x,y, and z coordinates
    x_loc = [x]; y_loc = [y]; z_loc = [z]
    for j in range(nsteps):
        # for each step, the x, y, z coordinates change by 1 or -1 
        direction = np.random.choice([-1,0,1])
        if direction == -1:
            x += np.random.choice([-1,1])
        elif direction == 1:
            y += np.random.choice([-1,1])
        else:
            z += np.random.choice([-1,1])
        sqdists[i,j] = (x**2 + y**2 + z**2) #distance from origin
        x_loc.append(x); y_loc.append(y); z_loc.append(z)
""")
