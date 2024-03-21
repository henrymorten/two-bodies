
# two-bodies
This is a simple model of the classical two body problem. Given innitial conditions - such as masses, location, and  velocity, it calculates the motion of the two bodies given that they are viewed as point particles - and the only things in existance. 

### Required packages: 
<ul>
  <li>Numpy: vectorised calculations and array manipulation</li>
  <li>Scipy: numerical integration </li>
  <li>Matplotlib (.pyplot and .animation): For data visualisation</li>
</ul>

There are different methods implemented in order to visualise the two body systems, comment/ un-comment to ajust to suit accordingly.
Examples from: Orbital Mechanics for Engineering Students 4th Edition, Howard D. Curtis  <br>

Update: <br>
Added additional code to "relative motion.py" to try and get around the matplotlib bug of the 3D sphere always being rendered infront of the trajectory - even if it wasn't physically there. This has apparantly been a known bug of 3+ years, still not fixed: https://matplotlib.org/2.2.2/mpl_toolkits/mplot3d/faq.html. <br>
To get the commented code working with Mayavi: https://pypi.org/project/mayavi/  - I installed:
<ul>
  <li> vtk (HAS TO BE VERSION 9.2 - I installed 9.2.4)
  <li> Mayavi
  <li>PyQt5
</ul>
All was installed using pip.