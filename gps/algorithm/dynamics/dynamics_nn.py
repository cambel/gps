""" This file defines neural net for dynamics estimation. """
import numpy as np

from gps.algorithm.dynamics.dynamics import Dynamics


class DynamicsLR(Dynamics):
    """ Dynamics with neural net, with constant prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.Fm = None
        self.fv = None
        self.dyn_covar = None

    def update_prior(self, sample):
        """ Update dynamics prior. """
        # Nothing to do - constant prior.
        pass

    def get_prior(self):
        """ Return the dynamics prior, or None if constant prior. """
        return None

    def fit(self, X, U):
        """ Fit dynamics. """
        
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics wih least squares regression.
        for t in range(T - 1):
            # xux = [X(t)[N,dX], U(t)[N,dU], X(t+1)[N,dX]] for t
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            # xux_mean = [X_mean[dX], U_mean[dU], X_mean[dX]]
            xux_mean = np.mean(xux, axis=0)
            empsig = (xux - xux_mean).T.dot(xux - xux_mean) / N
            sigma = 0.5 * (empsig + empsig.T)
            sigma[it, it] += self._hyperparams['regularization']

            Fm = np.linalg.solve(sigma[it, it], sigma[it, ip]).T
            # print "Fm", Fm.shape
            fv = xux_mean[ip] - Fm.dot(xux_mean[it])

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5 * (dyn_covar + dyn_covar.T)
        return self.Fm, self.fv, self.dyn_covar
