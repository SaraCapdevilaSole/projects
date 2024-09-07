from scipy.integrate import solve_ivp, odeint

import numpy as np
from tqdm import tqdm 
import logging

logging.basicConfig(level=logging.INFO)

no_ODE_y, VERBOSE = True, False

class BaseODESolver(object):
    def __init__(self, ode) -> None:
        self.ode = ode

    def solve(self, x0, dt, T, n_parcellations):
        raise NotImplementedError("solve method must be implemented in subclasses")

    @staticmethod
    def _initialise_solver(x0, dt, T, n_parcellations):
        X = np.zeros([int(np.ceil(T/dt) + 1), len(x0), n_parcellations])
        # initialise specifying parcellation values or not
        try:
            # no parcellation values specified i.e. np.shape(x0) == [len(x0),]
            X[0, :, :] = x0.reshape(1, len(x0), 1) # make shape compatible w/ broadcasting
        except ValueError:
            # parcellation values specified i.e. np.shape(x0) == [len(x0), n_parcellations]
            X[0, :, :] = x0
        except AttributeError:
            # Not an array e.g. a list
            logging.error("Convert x0 to an array!")
        return X


class EulerSolver(BaseODESolver):
    def solve(self, x0, dt, T, n_parcellations):
        logging.info(f'ODEs initialised.\nSolving using the Euler Method.')
        X = self._initialise_solver(x0, dt, T, n_parcellations)
        # print(X[0,-1,:])
        for i,t in tqdm(enumerate(np.arange(0, T-dt, dt))):
            # solve for i time, : all functions, : all parcellations
            dX = self.ode(t, X[i, :, :])
            X[i+1,:,:] = X[i,:,:] + dt*dX
            if no_ODE_y:
                X[i+1,-1,:] = dX[-1] # if not ODE on y
            if VERBOSE:
                logging.info(X[i+1,-1,:])
        logging.info("Finishing solving ODE")
        return X
    
class HeunsSolver(BaseODESolver):
    def solve(self, x0, dt, T, n_parcellations):
        logging.info(f'ODEs initialised.\nSolving using the Heun\'s Method.')
        X = self._initialise_solver(x0, dt, T, n_parcellations)
        for i, t in tqdm(enumerate(np.arange(0, T - dt, dt))):
            # solve for i time, : all functions, : all parcellations
            dX = self.ode(t, X[i, :, :])
            dX_predicted = self.ode(t + dt, X[i, :, :] + dt * dX)
            X[i + 1, :, :] = X[i, :, :] + 0.5 * dt * (dX + dX_predicted)
            if no_ODE_y:
                X[i + 1, -1, :] = 0.5 * (dX_predicted[-1] + dX[-1])  # if not ODE on y
            if VERBOSE:
                logging.info(X[i + 1, -1, :])
        logging.info("Finishing solving ODE")
        return X
    

class ODEIntSolver(BaseODESolver):
    def solve(self, x0, dt, T, n_parcellations):
        # TODO
        return solve_ivp(self.ode, [0, T], x0.reshape(-1), method='RK45', t_eval=np.arange(0, T, dt))
        
        
class RK4Solver(BaseODESolver):
    def solve(self, x0, dt, T, n_parcellations):
        X = self._initialise_solver(x0, dt, T, n_parcellations)
        for i,t in enumerate(np.arange(0, T-dt, dt)):
            k1 = self.ode(t, X[i,:,:])
            k2 = self.ode(t+dt/2, X[i,:,:] + k1*dt/2)
            k3 = self.ode(t+dt/2 + 1e-64, X[i,:,:] + k2*dt/2) # 1e-64 so that next data file chosen
            k4 = self.ode(t+dt, X[i,:,:] + k3*dt)
            dX = (k1 + 2*k2 + 2*k3 + k4)/6
            X[i+1,:,:] = X[i,:,:] + dt*dX
            if no_ODE_y:
                X[i+1,-1,:] = dX[-1]
            if VERBOSE:
                logging.info(X[i+1,-1,:])
        return X


