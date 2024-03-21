import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Gravitational constant
G = 6.67*(10**-11) #Nm^2/kg^2

ms = 0.5e26  # kg
me = 1.0e26  # kg

#Define innitial positions as vectors (i,j,k)
rs = np.array((0, 1_000_000, 0)) #m
re = np.array((3_000_000, 0, 0)) #m

#Innitial velocities as vectors
vs1 = np.array((0, 20_000, 30_000)) #m/s
ve1 = np.array((10_000, 40_000, 10_000)) #m/s

#Define a state vector of position and velocity:
y=np.concatenate((rs, re, vs1, ve1))

def motion(t, y):
    """
    A function that takes in a time series t, and a state vector, y and returns the differential of the state vector (ydot)

    -----------------------------------------------------------------
    Inputs: 
    t: array
    Array of times to calculate the positions at

    y: array
    State vector - assumed to have the following structure:
    y[0:3] -> inital x,y,z starting location of body 1
    y[3:6] -> inital x,y,z starting location of body 2
    y[6:9] -> inital x,y,z starting velocity of body 1
    y[9:12] -> inital x,y,z starting velocity of body 2

    -----------------------------------------------------------------
    returns:
    ydot: array
    ydot[0:3] -> inital x,y,z starting velocity of body 1
    ydot[3:6] -> inital x,y,z starting velocity of body 2
    ydot[6:9] -> inital x,y,z starting acceleration of body 1
    ydot[9:12] -> inital x,y,z starting acceleration of body 2

    """
    #Extract information back out of the state vector:
    rs, re, vs1, ve1 = np.split(y, 4)

    #Define an empty array to populate later - that will be the derivative of the state vector.
    ydot = np.empty(len(y))

    #Populate it with the information that we already know:
    ydot[:3] = vs1
    ydot[3:6] = ve1

    #Distance between the two bodies:
    #(this is equivalent of doing the sum of squares square rooted)
    r = np.linalg.norm(re - rs) 

    #calculate the components of accelleration for the two masses in 
    #vectorised form. 
    #As this confused me innitially: 
    #
    # From newtons second law: F = m_1 * a_1 
    # And the force between two objects due to gravity is:
    # F = (G*m_1*m_2)/r^2
    #
    # Equate the force terms:
    # m_1 * a_1 = (G*m_1*m_2) / r^2
    # a_1 = (G*m_2) / r^2

    #Since acceleration is a vector quantity... a_x = a_1*unit vector in X direction etc
    # and a^ = a/|a|
    # therefore a_x = G*m_2*(X_2-X-1)/r^3 etc 

    #Calculate the acceleration of the two bodies
    asun0 = (G * me * (re-rs)) / r**3
    aearth0 = -(G * ms * (re-rs)) / r**3

    #Populate the derivate 
    ydot[6:9] = asun0
    ydot[9:12] = aearth0
    
    return ydot

t_0 = 0  #Start time (Seconds)
t_f = 480  # End time (Seconds)
#Define a time array of points to solve at 
t_points = np.linspace(t_0, t_f, 1000)

# Docs for scipy.integrate.solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
#Function numerically integrates a system of orginary differential equations given an initial value.
solved = scipy.integrate.solve_ivp(
    fun=motion, #RHS of the system - additional arguments need to be passed in with the args 
    t_span=[t_0, t_f], #Starts at t0 and continues untill it reaches tf
    y0=y, #Initial state of the system
    t_eval=t_points, #Times at which to solve the computated problem
    rtol=1e-9,
    atol=1e-12,
)


#Returns an object with different fields
ydot = solved.y.T #This is the transpose of values to the solution at time points t
# solved.t would be the time points - read docs for more

#Split the output into constituate components
Rsun, Rearth, Vsun, VEarth = np.split(ydot,4,axis=1)

#Find the centre of mass (the Barycenter)
barycenter = (ms*Rsun+me*Rearth)/(me+ms)

######################################### Animated 3D Plot
"""
#Define and plot the figure
fig = plt.figure(figsize=(17, 5))


# Create 3 subplots
ax1 = fig.add_subplot(131, projection="3d")
ax2 = fig.add_subplot(132, projection="3d")
ax3 = fig.add_subplot(133, projection="3d")

def update(frame):
    for ax in [ax1, ax2, ax3]:
        if ax == ax1:
            ax.cla()  # Clear the previous plot
            ax.set_xlabel("X (Meters)")
            ax.set_ylabel("Y (Meters)")
            ax.set_zlabel("Z (Meters)")
            
            # Plot Rsun, Rearth, and barycenter only up to the current frame
            ax.plot(Rsun[:frame, 0], Rsun[:frame, 1], Rsun[:frame, 2], label="Body 1", c='b')
            ax.plot(Rearth[:frame, 0], Rearth[:frame, 1], Rearth[:frame, 2], label="Body 2",c='g')
            ax.plot(barycenter[:frame, 0], barycenter[:frame, 1], barycenter[:frame, 2], label="Barycentre", c='r')
            ax.set_title("Motion relative to the innertial frame")
            ax.legend()

        elif ax == ax2:
            ax.cla()  # Clear the previous plot
            ax.set_xlabel("X (Meters)")
            ax.set_ylabel("Y (Meters)")
            ax.set_zlabel("Z (Meters)")
            
            # Plot Rsun, Rearth, and barycenter only up to the current frame
            ax.plot(Rearth[:frame, 0]- Rsun[:frame, 0], Rearth[:frame, 1] - Rsun[:frame, 1], Rearth[:frame, 2]  - Rsun[:frame, 2], label="Body 2",c='g')
            ax.plot(barycenter[:frame, 0] - Rsun[:frame, 0], barycenter[:frame, 1] - Rsun[: frame,1], barycenter[:frame, 2] - Rsun[:frame,2], label="Barycentre", c='r')
            ax.set_title("Motion relative to Body 1")
            ax.legend()
        
        else:
            ax.cla()  # Clear the previous plot
            ax.set_xlabel("X (Meters)")
            ax.set_ylabel("Y (Meters)")
            ax.set_zlabel("Z (Meters)")
            
            # Plot Rsun, Rearth, and barycenter only up to the current frame
            ax.plot(Rsun[:frame, 0] - barycenter[:frame, 0], Rsun[:frame, 1] - barycenter[:frame, 1], Rsun[:frame, 2] - barycenter[:frame, 2], label="Body 1",c='b')
            ax.plot(Rearth[:frame, 0] - barycenter[:frame, 0], Rearth[:frame, 1] - barycenter[:frame, 1], Rearth[:frame, 2] - barycenter[:frame, 2], label="Body 2",c='g')
            ax.set_title("Motion relative to the barycentre ")
            ax.legend()

    plt.draw()  # Update the plot

#Loop over the calculated points to make it look like a smooth graph
animation = FuncAnimation(fig, update, frames=len(Rsun), interval=1)
plt.show()
"""
######################################### 3D static Plot

#Define and plot the figure
fig = plt.figure(figsize=(17, 5))

# Create 3 subplots
ax1 = fig.add_subplot(131,projection="3d")
ax2 = fig.add_subplot(132,projection="3d")
ax3 = fig.add_subplot(133,projection="3d")

ax1.cla()  # Clear the previous plot
ax1.set_xlabel("X (Meters)")
ax1.set_ylabel("Y (Meters)")
ax1.set_zlabel("Z (Meters)")

# Plot Rsun, Rearth, and barycenter only up to the current frame
ax1.plot(Rsun[:, 0], Rsun[:, 1],Rsun[:,2],  label="Body 1",c='b')
ax1.plot(Rearth[:, 0], Rearth[:, 1],Rearth[:,2], label="Body 2",c='g')
ax1.plot(barycenter[:, 0], barycenter[:, 1], barycenter[:,2], label="Barycentre",c='r')
ax1.set_title("Motion relative to the innertial frame")
ax1.legend()

ax2.cla()  # Clear the previous plot
ax2.set_xlabel("X (Meters)")
ax2.set_ylabel("Y (Meters)")
ax3.set_zlabel("Z (Meters)")
# Plot Rsun, Rearth, and barycenter only up to the current frame
ax2.plot(Rearth[:, 0]- Rsun[:, 0], Rearth[:, 1] - Rsun[:, 1], Rearth[:,2] - Rsun[:,2], label="Body 2",c='g')
ax2.plot(barycenter[:, 0] - Rsun[:, 0], barycenter[:, 1] - Rsun[: ,1], barycenter[:,2] - Rsun[:,2], label="Barycentre",c='r')
ax2.set_title("Motion relative to Body 1")
ax2.legend()

ax3.cla()  # Clear the previous plot
ax3.set_xlabel("X (Meters)")
ax3.set_ylabel("Y (Meters)")
ax3.set_zlabel("Z (Meters)")
# Plot Rsun, Rearth, and barycenter only up to the current frame
ax3.plot(Rsun[:, 0] - barycenter[:, 0], Rsun[:, 1] - barycenter[:, 1], Rsun[:,2] - barycenter[:,2] ,label="Body 1",c='b')
ax3.plot(Rearth[:, 0] - barycenter[:, 0], Rearth[:, 1] - barycenter[:, 1], Rearth[:,2] - barycenter[:,2], label="Body 2",c='g')
ax3.set_title("Motion relative to the barycentre ")
ax3.legend()

plt.show()


######################################### 2D static Plot 
"""
#Define and plot the figure
fig = plt.figure(figsize=(17, 5))

# Create 3 subplots
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.cla()  # Clear the previous plot
ax1.set_xlabel("X (Meters)")
ax1.set_ylabel("Y (Meters)")
# Plot Rsun, Rearth, and barycenter only up to the current frame
ax1.plot(Rsun[:, 0], Rsun[:, 1],  label="Body 1",c='b')
ax1.plot(Rearth[:, 0], Rearth[:, 1], label="Body 2",c='g')
ax1.plot(barycenter[:, 0], barycenter[:, 1], label="Barycentre",c='r')
ax1.set_title("Motion relative to the innertial frame")
ax1.legend()

ax2.cla()  # Clear the previous plot
ax2.set_xlabel("X (Meters)")
ax2.set_ylabel("Y (Meters)")
# Plot Rsun, Rearth, and barycenter only up to the current frame
ax2.plot(Rearth[:, 0]- Rsun[:, 0], Rearth[:, 1] - Rsun[:, 1], label="Body 2",c='g')
ax2.plot(barycenter[:, 0] - Rsun[:, 0], barycenter[:, 1] - Rsun[: ,1],  label="Barycentre",c='r')
ax2.set_title("Motion relative to Body 1")
ax2.legend()

ax3.cla()  # Clear the previous plot
ax3.set_xlabel("X (Meters)")
ax3.set_ylabel("Y (Meters)")
# Plot Rsun, Rearth, and barycenter only up to the current frame
ax3.plot(Rsun[:, 0] - barycenter[:, 0], Rsun[:, 1] - barycenter[:, 1], label="Body 1",c='b')
ax3.plot(Rearth[:, 0] - barycenter[:, 0], Rearth[:, 1] - barycenter[:, 1],  label="Body 2",c='g')
ax3.set_title("Motion relative to the barycentre ")
ax3.legend()


plt.show()
"""