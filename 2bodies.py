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
)

#Returns an object with different fields
ydot = solved.y.T #This is the transpose of values to the solution at time points t
# solved.t would be the time points - read docs for more

#Split the output into constituate components
Rsun, Rearth, Vsun, VEarth = np.split(ydot,4,axis=1)

#Find the centre of mass (the Barycenter)
barycenter = (ms*Rsun+me*Rearth)/(me+ms)

#Define and plot the figure
fig = plt.figure(figsize=(14, 9))
#Make the plot 3d
ax = fig.add_subplot(111, projection="3d")
def update(frame):
    ax.cla()  # Clear the previous plot
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    ax.set_zlabel("Z (Meters)")
    
    # Plot Rsun, Rearth, and barycenter only up to the current frame
    ax.plot(Rsun[:frame, 0], Rsun[:frame, 1], Rsun[:frame, 2], label="Body 1")
    ax.plot(Rearth[:frame, 0], Rearth[:frame, 1], Rearth[:frame, 2], label="Body 2")
    ax.plot(barycenter[:frame, 0], barycenter[:frame, 1], barycenter[:frame, 2], label="Barycentre")
    
    ax.legend()
    plt.draw()  # Update the plot

#Loop over the calculated points to make it look like a smooth graph
animation = FuncAnimation(fig, update, frames=len(Rsun), interval=1)
plt.show()
