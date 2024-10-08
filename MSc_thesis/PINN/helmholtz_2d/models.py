from functools import partial

import jax.numpy as jnp
import jax
from jax import lax, jit, grad, vmap, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn

from utils import find_idx, space_time_signal

class NFT3D(ForwardIVP): 
    def __init__(self, config, u_ref, t_star, coords, gamma, r, Qs, u_t, u_tt):
        super().__init__(config)

        # Assigning variables
        self.coords = coords
        self.t_star = t_star
        self.t0 = t_star[0]
        self.t1 = t_star[-1]
        self.Qs = Qs
        self.gamma = gamma
        self.r = r

        # Reference and constraints
        self.u0 = u_ref[0, :] 
        self.u_ref = u_ref # No sufficient constraints without boundary conditions
        self.u_t_ref = u_t
        self.u_tt_ref = u_tt

        # Access configuration
        self.noise = config.data.noise
        self.surface_loss = config.surface_loss
        self.sobolev_loss = config.data.sobolev_loss

        # Precomputing function mappings
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))
 
    def u_net(self, params, t, x, y):
        """Neural network approximation of the solution u(t, x, y)."""
        w = jnp.stack([t, x, y])
        u = self.state.apply_fn(params, w)
        return u[0]
    
    def u_t_net(self, params, t, x, y):
        """Gradient of u with respect to time, u_t."""
        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        return u_t
    
    def u_hessian_net(self, params, t, x, y):
        """Second-order derivatives of u."""
        u_hessian = hessian(self.u_net, argnums=(1, 2, 3))(params, t, x, y)
        return u_hessian
    
    def u_tt_net(self, params, t, x, y):
        """Second-order time derivative of u, u_tt."""
        return self.u_hessian_net(params, t, x, y)[0][0]

    def r_net(self, params, t, x, y):
        """Residual computation for the governing equation."""
        u = self.u_net(params, t, x, y)
        u_t = self.u_t_net(params, t, x, y)
        u_hessian = self.u_hessian_net(params, t, x, y)

        u_tt = u_hessian[0][0]
        u_xx = u_hessian[1][1]
        u_yy = u_hessian[2][2]
        u_laplacian = u_xx + u_yy 

        i_gamma = 1 / self.gamma
        Q = space_time_signal(t, x, y, self.noise)

        # Nonlinear residual equation
        ru = (i_gamma**2) * u_tt + 2 * i_gamma * u_t + u - (self.r ** 2) * u_laplacian - Q

        return ru
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        "Compute residuals and weights for causal training"
        t_sorted = batch[:, 0].sort()
        ru_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2]
        )

        # Split residuals into chunks
        ru_pred = ru_pred.reshape(self.num_chunks, -1)

        # Compute the residual loss and weight
        l_ru = jnp.mean(ru_pred**2, axis=1)
        w_ru = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l_ru)))

        return l_ru, w_ru 

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        """Compute losses for initial conditions, data, and residuals."""
        # Initial condition loss (t=0)
        u0_pred = vmap(self.u_net, (None, None, 0, 0))(
            params, self.t0, self.coords[:,0], self.coords[:,1]
        ) 
        ics_loss = jnp.mean((self.u0 - u0_pred) ** 2)

        # Data loss over the domain
        u_pred = self.u_pred_fn(
            params, self.t_star, self.coords[:, 0], self.coords[:, 1]
        )
        u_loss = jnp.sqrt(jnp.mean((self.u_ref - u_pred) ** 2))

        # Residual loss
        if self.config.weighting.use_causal == True:
            ru_l, w = self.res_and_w(params, batch)
            ru_loss = jnp.mean(ru_l * w)
        else:
            ru_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            ru_loss = jnp.mean((ru_pred) ** 2)

        if self.surface_loss:
            raise NotImplementedError
        
        elif self.sobolev_loss:
            key = jax.random.PRNGKey(0)
            key, subkey = jax.random.split(key)
            time_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=len(self.t_star))
            time = self.t_star[time_idx]
            coords_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=self.coords.shape[0])
            coords_0 = self.coords[coords_idx, 0]
            coords_1 = self.coords[coords_idx, 1]

            u_t_pred = self.u_t_net(params, time, coords_0, coords_1)
            u_t_loss = jnp.sqrt(jnp.mean((self.u_t_ref[time_idx, coords_idx] - u_t_pred) ** 2))
            
            u_tt_pred = self.u_tt_net(params, time, coords_0, coords_1)
            u_tt_loss = jnp.sqrt(jnp.mean((self.u_tt_ref[time_idx, coords_idx] - u_tt_pred) ** 2))

            loss_dict = {
                "ics": ics_loss, 
                "data": u_loss, 
                "res": ru_loss, 
                "u_t": u_t_loss,
                "u_tt": u_tt_loss
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
        """Placeholder for NTK diagonal computation (Not Implemented)."""
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref):
        """Compute L2 error between predicted and reference solution."""
        u_pred = self.u_pred_fn(params, t, coords[:, 0], coords[:, 1])
        u_error = jnp.sqrt(jnp.mean((u_pred - u_ref) ** 2))
        return u_error


class NFTEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, t, coords, u_ref):
        u_error = self.model.compute_l2_error(
            params, t, coords, u_ref
        )
        self.log_dict["u_error"] = u_error

    def __call__(self, state, batch, t, coords, u_ref):
        """Perform evaluation and log errors."""
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, t, coords, u_ref)

        return self.log_dict
