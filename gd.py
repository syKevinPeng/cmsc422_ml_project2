"""
Generic implementation of gradient descent.
"""

from numpy import *
import util

def gd(func, grad, x0, numIter, stepSize):
    """
    Perform gradient descent on some function func, where grad(x)
    computes its gradient at position x.  Begin at position x0 and run
    for exactly numIter iterations.  Use stepSize/sqrt(t+1) as a
    step-size, where t is the iteration number.

    We return the final solution as well as the trajectory of function
    values.
    """
    
    # initialize current location
    x = x0

    # set up storage for trajectory of function values
    trajectory = zeros(numIter + 1)
    trajectory[0] = func(x)

    # begin iterations
    for iter in range(numIter):
        # compute the gradient at the current location
        g = grad(x)


        # compute the step size
        eta = stepSize/sqrt(iter + 1)


        # step in the direction of the gradient
        x = x - g*eta


        # record the trajectory
        trajectory[iter+1] = func(x)

    # return the solution
    return (x, trajectory)

# testing code
if __name__ == "__main__":
    # print(gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2))
    x, trajectory = gd(lambda x: 0.5*x^(4)-3*x^(2)-2*x+5, lambda x: 2*x^3-6*x-2, 10, 100, 1)
    print(trajectory)
