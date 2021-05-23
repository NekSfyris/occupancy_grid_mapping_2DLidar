from numpy import *
import numpy as np
from numpy.linalg import inv

'''
A 3D LIDAR unit is scanning a surface that is approximately planar, 
returning range, elevation and azimuth measurements. In order to estimate
the equation of the surface in parametric form (as a plane), we need to 
find a set of parameters that best fit the measurements. 

The sph_to_cat and estimate_params functions, 
which transform LIDAR measurements into a Cartesian coordinate frame 
and estimate the plane parameters, respectively. We assume that 
measurement noise is negligible.
'''


def sph_to_cart(epsilon, alpha, r):
  """
  Transform sensor readings to Cartesian coordinates in the sensor
  frame. The values of epsilon and alpha are given in radians, while 
  r is in metres. Epsilon is the elevation angle and alpha is the
  azimuth angle (i.e., in the x,y plane).
  """
  p = np.zeros(3)  # Position vector 
  
  """To degrees"""
  """It seems it is not needed though"""
  #epsilon = epsilon / 180 * pi
  #alpha = alpha / 180 * pi
  
  
  p[0]=r*np.cos(alpha)*np.cos(epsilon)
  p[1]=r*np.sin(alpha)*np.cos(epsilon)
  p[2]=r*np.sin(epsilon)
  
  return p
  
def estimate_params(P):
  """
  Estimate parameters from sensor readings in the Cartesian frame.
  Each row in the P matrix contains a single 3D point measurement;
  the matrix P has size n x 3 (for n points). The format is:
  
  P = [[x1, y1, z1],
       [x2, x2, z2], ...]
       
  where all coordinate values are in metres. Three parameters are
  required to fit the plane, a, b, and c, according to the equation
  
  z = a + bx + cy
  
  The function should retrn the parameters as a NumPy array of size
  three, in the order [a, b, c].
  """

  """
  P = np.asarray(P)
  A = np.ones((P.shape[0], 3))
  Z = np.zeros(P.shape[0])
  
  #take second and third column from P array
  A[:, 1] = copy(P[:, 0])
  A[:, 2] = copy(P[:, 1])
  
  Z = copy(P[:, 2])
  """

  P = array(P)
  
  param_est = zeros(3)
  
  ones_row = matrix(ones(len(P)))#num of rows
  ones_column = ones_row.T
  
  x_row = (P[:,0])
  x_column = matrix(x_row).T
  y_row = (P[:,1])
  y_column = matrix(y_row).T

  z_row = (P[:,2])
  z_column = matrix(z_row).T
  
  A = hstack(((ones_column),(x_column),(y_column)))
  
  B = z_column


  
  param_set = linalg.inv(A.T.dot(A)).dot(A.T).dot(B)

  param_est[0] = param_set[0,0]
  param_est[1] = param_set[1,0]
  param_est[2] = param_set[2,0]

  return param_est
  


P = [[1, 2, 3], [10, 3, 4], [3, 4, 15], [3, 4, 15]]
print("Point measurements:")
print(P)
x = estimate_params(P)
print("Point measurements:")
print(x)