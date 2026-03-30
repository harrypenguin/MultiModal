import torch
import torch.nn as nn
import torch.nn.functional as F


class RedshiftFlow(nn.Module):
    """Conditional flow matching network for scalar redshift inference.

    Learns a velocity field dz/dt conditioned on encoder features, enabling
    sampling from the posterior p(z | features) via ODE integration.

    Uses optimal-transport conditional flow matching (OT-CFM): trains on
    linear interpolation paths from noise to target, regressing the velocity.

    Args:
        context_dim: dimension of the conditioning vector (pooled encoder features)
        hidden_dim: hidden layer width
        num_layers: number of hidden layers
    """

    def __init__(self, context_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        # Input: [t (1), z (1), context (context_dim)] -> velocity (1)
        layers = [nn.Linear(1 + 1 + context_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def velocity(self, t, z, context):
        """Predict dz/dt at flow time t, redshift value z, given context.

        Args:
            t: (B,) flow time in [0, 1]
            z: (B,) current redshift value
            context: (B, context_dim) conditioning features

        Returns:
            (B,) predicted velocity dz/dt
        """
        inp = torch.cat([t.unsqueeze(-1), z.unsqueeze(-1), context], dim=-1)
        return self.net(inp).squeeze(-1)

    def flow_matching_loss(self, context, z_target):
        """OT-CFM training loss: regress velocity along linear interpolation path.

        Args:
            context: (B, context_dim) conditioning features
            z_target: (B,) ground-truth redshift values

        Returns:
            scalar MSE loss on the velocity field
        """
        B = z_target.shape[0]
        t = torch.rand(B, device=z_target.device, dtype=z_target.dtype)
        x0 = torch.randn_like(z_target)
        # Linear interpolation: z_t = (1-t)*x0 + t*z_target
        zt = (1 - t) * x0 + t * z_target
        # Target velocity for OT path: v* = z_target - x0
        target_v = z_target - x0
        pred_v = self.velocity(t, zt, context)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, context, num_steps=50):
        """Sample z by Euler-integrating the learned velocity field.

        Note: this method is non-differentiable (used at inference).
        For training with inferred z, use sample_differentiable instead.

        Args:
            context: (B, context_dim) conditioning features
            num_steps: number of Euler integration steps

        Returns:
            (B,) sampled redshift values
        """
        z = torch.randn(context.shape[0], device=context.device, dtype=context.dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((context.shape[0],), i * dt, device=context.device, dtype=context.dtype)
            z = z + self.velocity(t, z, context) * dt
        return z

    def sample_differentiable(self, context, num_steps=50):
        """Sample z with gradient flow through the velocity field.

        Creates a computation graph through all Euler steps so that
        reconstruction loss can backpropagate through z_inferred to
        update the flow network parameters.

        Args:
            context: (B, context_dim) conditioning features
            num_steps: number of Euler integration steps

        Returns:
            (B,) sampled redshift values (with gradients)
        """
        z = torch.randn(context.shape[0], device=context.device, dtype=context.dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((context.shape[0],), i * dt, device=context.device, dtype=context.dtype)
            z = z + self.velocity(t, z, context) * dt
        return z
