from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn
from flax import linen as nn

from utils import find_idx, space_time_signal, get_idxs, generate_temporal_signal

class NFT3D(ForwardIVP): 
    def __init__(self, config, u_ref, t_star, u0, coords, gamma, r, L_star):
        super().__init__(config)

        self.coords = coords
        self.t_star = t_star
        self.L_star = L_star
        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        self.gamma = gamma
        self.r = r
        # self.noise = noise #
        self.noise = config.data.noise

        self.mesh_size = u_ref.shape[1]
        self.u0 = u0
        self.radius = config.data.radius
        self.sequation_args = config.data.spatial_equation_args
        self.params_sigmoid = config.data.params_sigmoid
        self.model_terms = config.data.model_terms['terms']
        self.idxs = get_idxs(self.model_terms)
        self.max_idx = max(self.idxs.values()) if self.idxs else None
        self.u_ref = u_ref 
        self.surface_loss = config.surface_loss

        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0, 0)), (None, 0, None, None, None))
        self.Q_pred_fn = vmap(self.Q_net, (None, None, 0, 0, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, 0))
        self.rs_pred_fn = self.rs_net 
        self.gamma_pred_fn = self.gamma_net 
        self.grad_pred_fn = self.grad_coeff_net 
        self.tgrad_pred_fn = self.tgrad_coeff_net 
        self.alpha_pred_fn = self.alpha_net 
        self.max_f_pred_fn = self.max_f_net 
        self.min_f_pred_fn = self.min_f_net 
        self.step_f_pred_fn = self.step_f_net 
        self.mult_f_pred_fn = self.mult_f_net 
        self.mr_pred_fn = self.mr_net 
        self.ms_pred_fn = self.ms_net 
        self.v_pred_fn = self.v_net 
        self.freq_denom_pred_fn = self.freq_denom_net 
        self.noise_pred_fn = self.noise_net 

    def neural_net(self, params, t, x, y, z):
        w = jnp.stack([t, x, y, z])
        output = self.state.apply_fn(params, w)
        return output

    def u_net(self, params, t, x, y, z):
        output = self.neural_net(params, t, x, y, z)
        return output[0]
    
    def Q_net(self, params, x, y, z):
        """spatial noise"""
        output = self.neural_net(params, 0.0, x, y, z)
        return output[self.idxs['Qs']]
    
    def rs_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        rs = output[self.idxs['rs']] 
        # rs = rs**2 
        rs = self.sigmoid(rs, **self.params_sigmoid['rs'])
        return rs
    
    def gamma_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        gamma = output[self.idxs['gamma']]
        # gamma = gamma**2
        gamma = self.sigmoid(gamma, **self.params_sigmoid['gamma_s'])
        return gamma

    def grad_coeff_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        grad_coeff = output[self.idxs['grad']]
        return grad_coeff

    def tgrad_coeff_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        tgrad_coeff = output[self.idxs['tgrad']]
        return tgrad_coeff
###
    def max_f_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        max_f = output[self.idxs['max_f']]**2
        return max_f

    def min_f_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        min_f = output[self.idxs['min_f']]**2
        return min_f

    def step_f_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        step_f = output[self.idxs['step_f']]**2
        return step_f

    def mult_f_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        mult_f = output[self.idxs['mult_f']]**2
        return mult_f

    def alpha_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        alpha = output[self.idxs['alpha']] 
        return alpha

    def mr_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        mr = output[self.idxs['mr']]
        return mr

    def ms_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        ms = output[self.idxs['ms']]
        ms = self.sigmoid(ms, **self.params_sigmoid['ms'])
        return ms

    def v_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        v = output[self.idxs['v']]
        return v

    def freq_denom_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        freq_denom = output[self.idxs['freq_denom']]
        return freq_denom

    def noise_net(self, params):
        output = self.neural_net(params, 0.0, 0.0, 0.0, 0.0)
        noise = output[self.idxs['noise']]
        return noise
    
    @staticmethod
    def sigmoid(x, xl, xh, fac=10):
        """Clip output x in range (xl, xh)"""
        return xl + (xh - xl) * (1 + jnp.exp(- x/fac))**(-1)

    def r_net(self, params, t, x, y, z):
        u = self.u_net(params, t, x, y, z)

        noise = self.noise
        sequation_args = dict(self.sequation_args)

        mr = self.mr_net(params) if self.model_terms['mr'] else sequation_args['mult'][0]
        ms = self.ms_net(params) if self.model_terms['ms'] else sequation_args['mult'][1]
        sequation_args['mult'] = (mr, ms)

        if self.model_terms['alpha']:
            sequation_args['alpha'] = self.alpha_net(params)
        if self.model_terms['v']:
            sequation_args['v'] = self.v_net(params)
        if self.model_terms['freq_denom']:
            sequation_args['freq_denom'] = self.freq_denom_net(params)
        if self.model_terms['noise']:
            noise = self.noise_net(params)
        
        if self.model_terms['Qs']:
            min_f = self.min_f_net(params) if self.model_terms['min_f'] else 2
            max_f = self.max_f_net(params) if self.model_terms['max_f'] else 130
            step_f = self.step_f_net(params) if self.model_terms['step_f'] else 5
            mult_f = self.mult_f_net(params) if self.model_terms['mult_f'] else 2
            # Q = self.Q_net(params, t, x, y, z) if self.model_terms['Qs'] else Q
            Q_time = generate_temporal_signal(t, sequation_args['alpha'], min_f, max_f, step_f, mult_f)
            Q_space = self.Q_net(params, x, y, z) # Find spatial noise
            Q = Q_time * Q_space
        else:
            Q = space_time_signal(t, x * self.L_star, y * self.L_star, z * self.L_star, noise, self.radius, **sequation_args)

        rs = self.rs_net(params) / self.L_star if self.model_terms['rs'] else self.r
        gamma = self.gamma_net(params) if self.model_terms['gamma'] else self.gamma

        if self.model_terms['grad']:
            u_x = grad(self.u_net, argnums=2)(params, t, x, y, z)
            u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
            u_z = grad(self.u_net, argnums=4)(params, t, x, y, z)
            u_grad = u_x + u_y + u_z
            grad_coeff = self.grad_coeff_net(params)
            grad_term = grad_coeff * u_grad
        else:
            grad_term = 0

        u_t = grad(self.u_net, argnums=1)(params, t, x, y, z)

        u_hessian = hessian(self.u_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)

        u_tt = u_hessian[0][0]
        u_xx = u_hessian[1][1]
        u_yy = u_hessian[2][2]
        u_zz = u_hessian[3][3]

        if self.model_terms['tgrad']:
            u_tx = u_hessian[0][1]
            u_ty = u_hessian[0][2]
            u_tz = u_hessian[0][3]
            u_tgrad = u_tx + u_ty + u_tz
            tgrad_coeff = self.tgrad_coeff_net(params)
            tgrad_term = tgrad_coeff * u_tgrad
        else:
            tgrad_term = 0
        
        u_laplacian = u_xx + u_yy + u_zz

        i_gamma = 1 / gamma

        ru = (i_gamma**2) * u_tt + 2 * i_gamma * u_t + u - (rs ** 2) * u_laplacian + grad_term + tgrad_term - Q
        
        # u_hess = jnp.array([[u_hessian[i][j] for j in range(1, 3)] for i in range(1, 3)])
        # rH = - 2 * self.H[parcell] * jnp.dot(u_grad, self.n[parcell, :]) 
        # rn = - jnp.dot(self.n[parcell, :], jnp.dot(u_hess, self.n[parcell, :]))
        #return ru, rH, rn

        return ru
    
    def ru_net(self, params, t, x, y, z):
        ru, _, _ = self.r_net(params, t, x, y, z)
        return ru
    
    def rH_net(self, params, t, x, y, z):
        _, rH, _ = self.r_net(params, t, x, y, z)
        return rH
    
    def rn_net(self, params, t, x, y, z):
        _, _, rn = self.r_net(params, t, x, y, z)
        return rn
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        "Compute residuals and weights for causal training"
        t_sorted = batch[:, 0].sort()
        ru_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2], batch[:, 3]
        )
        # Split residuals into chunks
        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        # rH_pred = rH_pred.reshape(self.num_chunks, -1)
        # rn_pred = rn_pred.reshape(self.num_chunks, -1)

        l_ru = jnp.mean(ru_pred**2, axis=1)
        # l_rH = jnp.mean(rH_pred**2, axis=1)
        # l_rn = jnp.mean(rn_pred**2, axis=1)

        w_ru = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_ru)))
        # w_rH = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_rH)))
        # w_rn = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_rn)))

        # w = jnp.vstack([w_ru, w_rH, w_rn])
        # w = w.min(0)
        return l_ru, w_ru #l_ru, l_rH, l_rn, w

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # ic_batch = batch["ic"]
        # data_batch = batch["data"]
        res_batch = batch["res"]

        # ic_coords_batch, u0_batch = ic_batch
        # t_batch, coords_batch, u_batch = data_batch

        # Initial condition loss
        u0_pred = vmap(self.u_net, (None, None, 0, 0, 0))(
            params, self.t0, self.coords[:,0], self.coords[:,1], self.coords[:,2]
        )
        ics_loss = jnp.mean((self.u0 - u0_pred) ** 2)
        # u0_pred = vmap(self.u_net, (None, None, 0, 0, 0))(
        #     params, self.t0, ic_coords_batch[:,0], ic_coords_batch[:,1], ic_coords_batch[:,2]
        # )
        # ics_loss = jnp.mean((u0_batch - u0_pred) ** 2)

        # Data loss
        u_pred = self.u_pred_fn(
            params, self.t_star, self.coords[:, 0], self.coords[:, 1], self.coords[:,2]
        )
        u_loss = jnp.sqrt(jnp.mean((self.u_ref - u_pred) ** 2))
        # u_pred = self.u_pred_fn(
        #     params, t_batch, coords_batch[:, 0], coords_batch[:, 1], coords_batch[:,2]
        # )
        # u_loss = jnp.sqrt(jnp.mean((u_batch - u_pred) ** 2))

        # Residual loss
        if self.config.weighting.use_causal == True:
            # ru_l, rH_l, rn_l, w = self.res_and_w(params, batch)
            ru_l, w = self.res_and_w(params, res_batch)
            ru_loss = jnp.mean(ru_l * w)
            # rH_loss = jnp.mean(rH_l * w)
            # rn_loss = jnp.mean(rn_l * w)
        else:
            ru_pred = self.r_pred_fn(
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]
            )
            ru_loss = jnp.mean((ru_pred) ** 2)
            # rH_loss = jnp.mean((rH_pred) ** 2)
            # rn_loss = jnp.mean((rn_pred) ** 2)

        if self.surface_loss:
            loss_dict = {
                "ics": ics_loss,
                "data": u_loss,
                "res": ru_loss,
                # "H": rH_loss,
                # "n": rn_loss,
            }
        else:
            loss_dict = {
                "ics": ics_loss, 
                "data": u_loss, 
                "res": ru_loss
            }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref):
        u_pred = self.u_pred_fn(params, t, coords[:, 0], coords[:, 1], coords[:, 2])
        u_error = jnp.sqrt(jnp.mean((u_pred - u_ref) ** 2))
        return u_error

    @partial(jit, static_argnums=(0,))
    def compute_Q_error(self, params, t, coords, Q_ref):
        Q_pred = self.Q_pred_fn(params, t, coords[:, 0], coords[:, 1], coords[:, 2])
        Q_error = jnp.sqrt(jnp.mean((Q_pred - Q_ref) ** 2))
        return Q_error

    @partial(jit, static_argnums=(0,))
    def compute_rs_error(self, params):
        rs_pred = self.rs_pred_fn(params) / self.L_star
        rs_error = jnp.sqrt(jnp.mean((rs_pred - self.r) ** 2))
        return rs_error

    @partial(jit, static_argnums=(0,))
    def compute_gamma_error(self, params):
        gamma_pred = self.gamma_pred_fn(params)
        gamma_error = jnp.sqrt(jnp.mean(((gamma_pred - self.gamma)) ** 2))
        return gamma_error


class NFTEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, t, coords, u_ref):
        u_error = self.model.compute_l2_error(
            params, t, coords, u_ref
        )
        self.log_dict["u_error"] = u_error
        pass

    def log_preds(self, params):
        not_constant = {'Qs'}
        model_terms = self.config.data.model_terms['terms']

        # Logging active terms
        for term, is_active in model_terms.items():
            if is_active and term not in not_constant:
                self._log_prediction(term, params)

    def _log_prediction(self, term, params):
        log_key = f'{term}_pred'
        fn_name = f'{term}_pred_fn'
        self.log_dict[log_key] = getattr(self.model, fn_name)(params)     

    def __call__(self, state, batch, t, coords, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, t, coords, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
