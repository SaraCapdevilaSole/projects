from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn

from utils import find_idx, space_time_signal

class NFT3D(ForwardIVP): 
    def __init__(self, config, u_ref, t_star, u0, coords, gamma, r, L_star, curv, normals):
        super().__init__(config)

        self.coords = coords
        self.L_star = L_star
        self.t_star = t_star
        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        self.gamma = gamma
        self.r = r
        self.noise = config.data.noise

        self.mesh_size = u_ref.shape[1]
        self.u0 = u0

        self.radius = config.data.radius
        self.sequation_args = config.data.spatial_equation_args
        self.u_ref = u_ref 
        self.surface_loss = config.surface_loss
        self.curv = curv
        self.normals = normals

        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0, 0)), (None, 0, None, None, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, 0, 0, 0, 0, 0))

    def u_net(self, params, t, x, y, z):
        if self.surface_loss:
            w = jnp.stack([t, x, y, z])
        else:
            w = jnp.stack([t, x, y, z])
        u = self.state.apply_fn(params, w)
        return u[0]

    def r_net(self, params, t, x, y, z, H, n_x, n_y, n_z):
        u = self.u_net(params, t, x, y, z)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y, z)

        u_hessian = hessian(self.u_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)

        u_tt = u_hessian[0][0]
        u_xx = u_hessian[1][1]
        u_yy = u_hessian[2][2]
        u_zz = u_hessian[3][3]
        
        u_laplacian = u_xx + u_yy + u_zz

        i_gamma = 1 / self.gamma

        Q = space_time_signal(t, x * self.L_star, y * self.L_star, z * self.L_star, self.noise, self.radius, **self.sequation_args)

        ru = (i_gamma**2) * u_tt + 2 * i_gamma * u_t + u - (self.r ** 2) * u_laplacian - Q
        
        if self.surface_loss:
            u_x = grad(self.u_net, argnums=2)(params, t, x, y, z)
            u_y = grad(self.u_net, argnums=3)(params, t, x, y, z)
            u_z = grad(self.u_net, argnums=4)(params, t, x, y, z)
            u_grad = jnp.array([u_x, u_y, u_z])
            u_hess = jnp.array([[u_hessian[i][j] for j in range(1, 4)] for i in range(1, 4)])
            n_vec = jnp.array([n_x, n_y, n_z])
            rH = - 2 * H * jnp.dot(u_grad, n_vec) 
            rn = - jnp.dot(n_vec, jnp.dot(u_hess, n_vec))
            return ru, rH, rn
        else:
            return ru, None, None
    
    def ru_net(self, params, t, x, y, z, H, n_x, n_y, n_z):
        ru, _, _ = self.r_net(params, t, x, y, z, H, n_x, n_y, n_z)
        return ru
    
    def rH_net(self, params, t, x, y, z, H, n_x, n_y, n_z):
        _, rH, _ = self.r_net(params, t, x, y, z, H, n_x, n_y, n_z)
        return rH
    
    def rn_net(self, params, t, x, y, z, H, n_x, n_y, n_z):
        _, _, rn = self.r_net(params, t, x, y, z, H, n_x, n_y, n_z)
        return rn
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch, curv, normals):
        "Compute residuals and weights for causal training"
        t_sorted = batch[:, 0].sort()
        ru_pred, rH_pred, rn_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2], batch[:, 3], curv, normals[:, 0], normals[:, 1], normals[:, 2]
        )
        # Split residuals into chunks
        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        l_ru = jnp.mean(ru_pred**2, axis=1)
        w_ru = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_ru)))
        
        if self.surface_loss:
            rH_pred = rH_pred.reshape(self.num_chunks, -1)
            rn_pred = rn_pred.reshape(self.num_chunks, -1)

            l_rH = jnp.mean(rH_pred**2, axis=1)
            l_rn = jnp.mean(rn_pred**2, axis=1)

            w_rH = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_rH)))
            w_rn = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_rn)))

            w = jnp.vstack([w_ru, w_rH, w_rn])
            w = w.min(0)
            return l_ru, l_rH, l_rn, w
        else:
            return l_ru, None, None, w_ru

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        if self.surface_loss:
            res_batch = batch[0]
            curv, normals = batch[1], batch[2]
        else:
            res_batch = batch
            curv, normals = None, None

        # Initial condition loss
        u0_pred = vmap(self.u_net, (None, None, 0, 0, 0))(
            params, self.t0, self.coords[:,0], self.coords[:,1], self.coords[:,2]
        ) 
        ics_loss = jnp.mean((self.u0 - u0_pred) ** 2)

        u_pred = self.u_pred_fn(
            params, self.t_star, self.coords[:, 0], self.coords[:, 1], self.coords[:,2]
        )
        u_loss = jnp.sqrt(jnp.mean((self.u_ref - u_pred) ** 2))

        # Residual loss
        if self.config.weighting.use_causal:
            ru_l, rH_l, rn_l, w = self.res_and_w(params, res_batch, curv, normals)
            ru_loss = jnp.mean(ru_l * w)
            if self.surface_loss:
                rH_loss = jnp.mean(rH_l * w)
                rn_loss = jnp.mean(rn_l * w)
        else:
            ru_pred, rH_pred, rn_pred = self.r_pred_fn(
                params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3], curv, normals[:, 0], normals[:, 1], normals[:, 2]
            )
            ru_loss = jnp.mean((ru_pred) ** 2)
            if self.surface_loss:
                rH_loss = jnp.mean((rH_pred) ** 2)
                rn_loss = jnp.mean((rn_pred) ** 2)

        loss_dict = {
            "ics": ics_loss, 
            "data": u_loss, 
            "res": ru_loss
        }
        if self.surface_loss:
            loss_dict["H"] = rH_loss
            loss_dict["n"] = rn_loss
            
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref):
        u_pred = self.u_pred_fn(params, t, coords[:, 0], coords[:, 1], coords[:, 2])
        u_error = jnp.sqrt(jnp.mean((u_pred - u_ref) ** 2))
        return u_error

class NFTEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    # def log_errors(self, params, t, coords, u_ref):
    #     u_error = self.model.compute_l2_error(
    #         params, t, coords, u_ref
    #     )
    #     self.log_dict["u_error"] = u_error

    def __call__(self, state, batch): #, t, coords, u_ref):
        self.log_dict = super().__call__(state, batch)

        # if self.config.logging.log_errors:
        #     self.log_errors(state.params, t, coords, u_ref)

        return self.log_dict
