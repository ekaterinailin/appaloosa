import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

def linfit(x,y,x_err,y_err):

    # Define a function (quadratic in our case) to fit the data with.
    def lin_func(p, x):
        m, c = p
        return m*x+c

    # Create a model for fitting.
    lin_model = Model(lin_func)

    # Create a RealData object using our initiated data from above.
    data = RealData(x, y, sx=x_err, sy=y_err)

    # Set up ODR with the model and data.
    odr = ODR(data, lin_model, beta0=[0., 1.])

    # Run the regression.
    out = odr.run()

    # Use the in-built pprint method to give us results.
    #out.pprint()
    '''Beta: [ 1.01781493  0.48498006]
    Beta Std Error: [ 0.00390799  0.03660941]
    Beta Covariance: [[ 0.00241322 -0.01420883]
     [-0.01420883  0.21177597]]
    Residual Variance: 0.00632861634898189
    Inverse Condition #: 0.4195196193536024
    Reason(s) for Halting:
      Sum of squares convergence'''

    #x_fit = np.linspace(x[0], x[-1], 1000)
    #y_fit = lin_func(out.beta, x_fit)

    #plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='None', marker='x')
    #plt.plot(x_fit, y_fit)

   # plt.show()
    return out.beta, out.sd_beta