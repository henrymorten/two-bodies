import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from matplotlib.animation import FuncAnimation

plt.close("all")

#gravitational constant
G = 6.67e-11

#Hours to model over:
hours = 3600 #s

#Mass of the Earth
me = 5.97e24#Kg
#Radius of the Earth:
re = 6378e3 #m

#Mass of orbitig body:
m2 = 1000 #Kg

#Initial parameters of position and velocity
# r0 = np.array((8_000e3, 0, 6_000e3)) #m
# v0 = np.array((0,7_000,0)) #m/s

r0 = np.array((8_000e3, 0, 6_000e3)) #m
v0 = np.array((0,7_000,0)) #m/s

#nitial state vector y
y = np.concatenate((r0, v0)) 

#Start time (Seconds)
t_0 = 0
t_f = 5*hours  # End time (Seconds)

#Define a time array of points to solve at 
t_points = np.linspace(t_0, t_f, 20_000)

def diff(t, y):
    """
    A function that takes in a time series t, and a state vector, y and returns the differential of the state vector (ydot)

    -----------------------------------------------------------------
    Inputs: 
    t: array
    Array of times to calculate the positions at

    y: array
    State vector - assumed to have the following structure:
    y[0:2] -> inital x,y,z starting location of body 2
    y[3:5] -> inital x,y,z starting velocity of body 2

    -----------------------------------------------------------------
    returns:
    ydot: array
    ydot[0:2] -> inital x,y,z starting velocity of body 2
    ydot[3:5] -> inital x,y,z starting acceleration of body 2

    """
    #Extract Position and velocity from the state vector
    pos, vel = np.split(y, 2)

    #Calculate tha magnitude of the positional vector 
    r = np.linalg.norm(pos)

    #Calculate 3d component of acceleration
    a = (
    (-mu * pos) / r**3
    )

    #differential of the state vector - populate it with what we know
    ydot = np.concatenate((vel, a))
    return ydot

#Calculate the gravitational parameter (\mu)
mu = G*me

# Docs for scipy.integrate.solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
#Function numerically integrates a system of orginary differential equations given an initial value.
solved = scipy.integrate.solve_ivp(
    fun=diff, #RHS of the system - additional arguments need to be passed in with the args 
    t_span=[t_0, t_f], #Starts at t0 and continues untill it reaches tf
    t_eval=t_points,
    y0=y, #Initial state of the system
    vectorized=True,
    rtol=1e-9,
    atol=1e-12
)

#Returns an object with different fields
ydot = solved.y.T #This is the transpose of values to the solution at time points t
# solved.t would be the time points - read docs for more

#Extract position and velocity arrays from the differentiated state vector.
pos, vel = np.split(ydot, 2, axis=1)

######################################### Static 3D Plot


# Create a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
#With Radius of the earth
x = re * np.outer(np.cos(u), np.sin(v))
y = re * np.outer(np.sin(u), np.sin(v))
z = re * np.outer(np.ones(np.size(u)), np.cos(v))

# Create a new figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the orbit trajectory
ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='r', label='Orbit Trajectory')
#
#  Plot the sphere
ax.plot_surface(x, y, z,cmap=plt.cm.YlGnBu_r)


# Set an equal aspect ratio
ax.set_aspect('equal')

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Earth Sphere and Orbit Trajectory')
ax.legend()

plt.show()

######################################### Animated 3D Plot


"""fig = plt.figure()


# Create a subplot
ax = fig.add_subplot(111, projection="3d")

# Set fixed axis limits
ax.set_xlim(-2 * re, 2 * re)
ax.set_ylim(-2 * re, 2 * re)
ax.set_zlim(-2 * re, 2 * re)
# Create a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

#With Radius of the earth
x = re * np.outer(np.cos(u), np.sin(v))
y = re * np.outer(np.sin(u), np.sin(v))
z = re * np.outer(np.ones(np.size(u)), np.cos(v))

def update(frame):
    ax.cla()  # Clear the previous plot
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Y (Meters)")
    ax.set_zlabel("Z (Meters)")

    # Plot the sphere
    ax.plot_surface(x, y, z, color='b', alpha = 0.5)
            
    # Plot Rsun, Rearth, and barycenter only up to the current frame
    ax.plot(pos[:frame, 0], pos[:frame, 1], pos[:frame, 2], label="Orbit", c='r')
    ax.set_title("Relative 2 Body problem")
    ax.legend()
    ax.set_aspect('equal')
    plt.draw()  # Update the plot

#Loop over the calculated points to make it look like a smooth graph
animation = FuncAnimation(fig, update, frames=len(pos), interval=1)
plt.show()"""


