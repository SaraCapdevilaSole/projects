import os
import meshio
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

no_ODE_y = True

class Balloon:
    def __init__(self, t_array, data):
        self.kappa = 0.65  # signal decay rate (s^-1)
        self.gamma = 0.41  # flow-dependent elimination rate (s^-1)
        self.tau = 0.98    # haemodynamic transit time (s)
        self.alpha = 0.32  # Grubb’s exponent

        if no_ODE_y:
            # Parameters from Deco: https://github.com/decolab/cb-neuromod/blob/6fa5abb8226e2c15330b187bfadc1924dec84918/functions/BOLD.m#L47
            # self.rho = 0.34 
            # self.k1 = self.rho * 7    
            # self.k2 = 2
            # self.k3 = 2 * self.rho - 0.2
            # self.V0 = 0.02   

            # THIS WOKRS AS EXPECTED!
            self.rho = 0.4
            self.k1 = 4.3*40.3*self.rho*0.04
            self.k2 = 25*self.rho*0.04
            self.k3 = 1
            self.V0 = 0.04     
        else:
            # Parameters from Nature paper
            self.rho = 0.34    # resting oxygen extraction fraction
            self.k1 = 3.72     # 3T fMRI parameter
            self.k2 = 0.53     # 3T fMRI parameter
            self.k3 = 0.53     # 3T fMRI parameter
            self.V0 = 0.02     # resting blood volume fraction

        self.t_array = t_array
        # self.data_path_f: callable = lambda f: data_path + f'_{f}.vtu'
        self.data = self._format_data(data)

    # def _load_data(self):
    #     logging.info(f'Loading files from path: {self.data_path_f('file_#')}')
    #     return {
    #         t: self._load_file(self.data_path_f(t)) for t in tqdm(range(0,len(self.t_array)))
    #     }

    def _format_data(self, data):
        logging.info(f'Formating input files')
        try:
            return {
                t: phi for t, phi in enumerate(data)
            }
        except ValueError:
            logging.error('Data should contain time and phi_e information')
        except TypeError:
            logging.error('Time array should be element 0, and phi_e element 1')

    # @staticmethod
    # def _load_file(data_path):
    #     mesh = meshio.read(data_path)
    #     mesh_phi_e = mesh.point_data['phi_e']
    #     return mesh_phi_e

    def get_n_parcellations(self) -> int:
        return len(self.data[next(iter(self.data))])
    
    def _time_to_file(self, t_i):
        try:
            return np.where(np.isclose(self.t_array, t_i, atol=10**(-5)))[0][0]
            # return np.where(self.t_array == t_i)[0][0]
        except IndexError:
            print("Approximating data file with closest match in time!")
            return np.abs(self.t_array - t_i).argmin()

    def ode_system(self, t, x: np.array):
        # x = [z,f,v,q,y]
        # x size: [num vertices, 5] 
        function_names = ['_dz_dt', '_df_dt', '_dv_dt', '_dq_dt', '_dy_dt']
        derivatives = [getattr(self, func)(t, x) for func in function_names]
        return np.array(derivatives)

    def _dz_dt(self, t, x):
        N_t = self.data[self._time_to_file(t)]
        return N_t - self.kappa * x[0] - self.gamma * (x[1] - 1)
    
    def _df_dt(self, t, x):
        return x[0]
    
    def _dv_dt(self, t, x):
        return 1/self.tau * (
            x[1] - x[2]**(1/self.alpha)
        )
    
    def _dq_dt(self, t, x):
        if no_ODE_y:
            exp_term = x[2]**(1/self.alpha) * x[3] / x[2] # deco
        else: 
            exp_term = x[2]**(1/self.alpha - 1) # nature paper

        return 1/self.tau * (
            x[1] / self.rho * 
                (1 - (1-self.rho)**(1/x[1]))
                - exp_term
        )
    
    def _dy_dt(self, t, x):
        return self.V0 * (
              self.k1 * (1 - x[3]) 
            + self.k2 * (1 - x[3]/x[2])  
            + self.k3 * (1 - x[2])
        )