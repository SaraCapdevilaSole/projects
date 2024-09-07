import os

class BaseConfig:
    VALID_HEMIS = ['rh', 'lh', 'both', None]
    def __init__(self, hemi, gamma, path_msh, path_txt, output_phi_e, pinns_input_file, dt, r, T, noise, sampling_nom=0.01, bold_file=None):
        self.T = T
        self.hemi = hemi
        self.gamma = gamma
        self.path_msh = path_msh
        self.path_txt = path_txt
        self.output_phi_e = output_phi_e
        self.pinns_input_file = pinns_input_file
        self.dt = dt
        self.r = r
        self.noise = noise
        self.freq = int(sampling_nom/dt) #int(0.01/dt)
        self.bold_file = bold_file
        self._check_dir(output_phi_e)
    
    @staticmethod
    def _check_dir(dir):
        os.makedirs(dir, exist_ok=True)

    def _calculate_parcellations(self):
        assert self.hemi in self.VALID_HEMIS, f"hemi type {self.hemi} not accepted"
        if self.hemi is None:
            return -1
        elif self.hemi != 'both':
            return 51
        else:
            return 101


class BrainConfig(BaseConfig):
    def __init__(self, T=1, dt=1e-4, sampling_nom=0.001, noise=0.3, size=30, XN=100, TN=10, extra=None, mesh_type='inflated', hemi='lh', prop_to_keep=0.1): #, factor=2): 
        self.NAME = mesh_type 
        _dir = f'./data/helmholtz_3d_{self.NAME}_mesh'
        _extra = f"size={size}_dt={dt}_sn={sampling_nom}_T={T}_XN={XN}_TN={TN}"
        _extra += f'_Q={noise}' if noise is not None else ''
        if extra is not None:
            _extra += extra

        self.prop_to_keep = prop_to_keep
        # self.factor = factor

        params = {'gamma':116,'r':30}
        # params = {'gamma':10,'r':0.001}
        
        if params['gamma'] == 10:
            print("USING FAKE PARAMETERS!!!")

        self.path_func = lambda s: f'data/input_data/{self.NAME}_meshes/{hemi}_{self.NAME}_{s}k.msh'
        output_phi = os.path.join(_dir, _extra)
        self._dir = _dir
        self.size = size
        self.XN = XN
        self.TN = TN
        self.solver_params = {'ksp_type': 'gmres', 'pc_type': 'ilu', 'ksp_rtol': 1e-12}

        pinns_dir = f'./data/formatted_data_{self.NAME}'
        self._check_dir(pinns_dir)
        super().__init__(
            gamma=params['gamma'],
            r=params['r'],
            path_txt=None,
            path_msh=self.path_func(s=size),
            output_phi_e=output_phi,
            pinns_input_file=f'{pinns_dir}/PINN-NFT_data_{_extra}.npy',
            bold_file=f'./data/bold_data/bold_data_{_extra}.npy',
            dt=dt,
            T=T,
            noise=noise, 
            hemi=hemi,
            sampling_nom=sampling_nom,
        )

        self.parcellations = self._calculate_parcellations()


class SphereConfig(BaseConfig):
    NAME = 'sphere'
    def __init__(self, T=1, dt=1e-4, sampling_nom=0.001, noise=0.3, size=30, XN=100, TN=10, radius=100, extra=None): 
        self.NAME = f'unit_' + self.NAME if radius==1 else self.NAME + f'_r={radius}'
        _dir = f'./data/helmholtz_3d_{self.NAME}'
        _extra = f"size={size}_dt={dt}_sn={sampling_nom}_T={T}_XN={XN}_TN={TN}"
        _extra += f'_Q={noise}' if noise is not None else ''
        if extra is not None:
            _extra += extra

        params_r = {
            100: {'gamma':116,'r':30}, 
            # 100: {'gamma':10,'r':0.001},
            1: {'gamma':10,'r':0.001}
        }
        assert radius in params_r.keys(), f"Unknown PDE parameters with radius: {radius}"
        params = params_r[radius]

        if params['gamma'] == 10:
            print("USING FAKE PARAMETERS!!!")

        output_phi = os.path.join(_dir, _extra)
        
        self.radius = radius
        self._dir = _dir
        self.size = size
        self.XN = XN
        self.TN = TN
        self.solver_params = {'ksp_type': 'gmres', 'pc_type': 'ilu', 'ksp_rtol': 1e-12}

        pinns_dir = f'./data/formatted_data_{self.NAME}'
        self._check_dir(pinns_dir)
        super().__init__(
            gamma=params['gamma'],
            r=params['r'],
            path_txt=None,
            path_msh=None,
            output_phi_e=output_phi,
            pinns_input_file=f'{pinns_dir}/PINN-NFT_data_{_extra}.npy',
            bold_file=f'./data/bold_data/bold_data_{_extra}.npy',
            dt=dt,
            T=T,
            noise=noise, 
            hemi=None,
            sampling_nom=sampling_nom,
        )

        self.parcellations = self._calculate_parcellations()


class SquareConfig(BaseConfig):
    def __init__(self, T=1, dt=1e-6, sampling_nom=1e-4, noise=0.3, size=50, XN=100, TN=10, extra=None): 
        _dir = './data/helmholtz_2d_square'
        _extra = f"size={size}_dt={dt}_sn={sampling_nom}_T={T}_XN={XN}_TN={TN}"
        _extra += f'_Q={noise}' if noise is not None else ''
        if extra is not None:
            _extra += extra
        output_phi = os.path.join(_dir, _extra)

        params = {'gamma':10,'r':0.001}
    
        self._dir = _dir
        self.size = size
        self.XN = XN
        self.TN = TN
        self.solver_params = {'ksp_type': 'gmres', 'pc_type': 'ilu', 'ksp_rtol': 1e-12}

        pinns_dir = './data/formatted_data_square'
        self._check_dir(pinns_dir)
        super().__init__(
            gamma=params['gamma'],
            r=params['r'],
            path_txt=None,
            path_msh=None,
            output_phi_e=output_phi,
            pinns_input_file=f'{pinns_dir}/PINN-NFT_data_{_extra}.npy',
            bold_file=f'./data/bold_data/bold_data_{_extra}.npy',
            dt=dt,
            T=T,
            noise=noise, 
            hemi=None,
            sampling_nom=sampling_nom,
        )

        self.parcellations = self._calculate_parcellations()
