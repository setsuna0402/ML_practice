"""
PINN model for primordial H/He chemistry.

Draft version:
- No final normalization scheme is imposed yet.
- The model uses a hard initial condition:
      y(t) = y0 + t_scaled * NN(...)
  so y(t=0) = y0 exactly.
- State variables currently predicted:
      HI fraction, HeI fraction, HeII fraction, specific internal energy
- HII and HeIII can be reconstructed from conservation.

The input layout is defined explicitly in INPUT_NAMES below so that the
normalization strategy can be changed later without changing the network.
"""

import torch
import torch.nn as nn


INPUT_NAMES = (
    "time",
    "density",
    "HI0",
    "HeI0",
    "HeII0",
    "u0",
    "Gamma_HI",
    "Gamma_HeI",
    "Gamma_HeII",
    "pi_HI",
    "pi_HeI",
    "pi_HeII",
)

OUTPUT_NAMES = (
    "HI",
    "HeI",
    "HeII",
    "u",
)


class IdentityNormalizer(nn.Module):
    """
    Placeholder normalizer.

    Replace this later with log scaling / affine scaling once the final
    parameter ranges are decided.
    """
    def forward(self, x):
        return x


class ChemistryPINN(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        n_hidden_layers=4,
        activation=nn.Tanh,
        normalizer=None,
        time_scale=1.0,
    ):
        super().__init__()

        self.input_dim = len(INPUT_NAMES)
        self.output_dim = len(OUTPUT_NAMES)

        self.normalizer = normalizer or IdentityNormalizer()
        self.time_scale = float(time_scale)

        layers = [nn.Linear(self.input_dim, hidden_dim), activation()]

        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]

        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (N, 12)
            Columns follow INPUT_NAMES.

        Returns
        -------
        y : Tensor, shape (N, 4)
            [HI, HeI, HeII, u]

        Notes
        -----
        This is deliberately a simple hard-IC formulation for the first
        draft. It does NOT yet force fractions into [0, 1]. We can change
        the output parameterization after deciding whether log-fraction,
        logit, or another constrained representation is preferable.
        """
        x_norm = self.normalizer(x)

        raw = self.net(x_norm)

        t = x[:, 0:1]
        y0 = torch.stack(
            (
                x[:, 2],  # HI0
                x[:, 3],  # HeI0
                x[:, 4],  # HeII0
                x[:, 5],  # u0
            ),
            dim=1,
        )

        # Hard initial condition:
        # at t = 0, correction = 0 exactly.
        tau = t / self.time_scale
        y = y0 + tau * raw

        return y


def time_derivative(model, x):
    """
    Compute dy/dt with autograd.

    x must have requires_grad=True.

    Returns
    -------
    dydt : Tensor, shape (N, 4)
    """
    if not x.requires_grad:
        raise ValueError("x.requires_grad must be True for PINN derivatives.")

    y = model(x)

    derivatives = []

    for i in range(y.shape[1]):
        grad_i = torch.autograd.grad(
            y[:, i].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0]

        derivatives.append(grad_i)

    return torch.stack(derivatives, dim=1)


def reconstruct_fractions(y):
    """
    Reconstruct dependent species fractions.

    Input
    -----
    y[..., 0] = HI
    y[..., 1] = HeI
    y[..., 2] = HeII

    Returns
    -------
    dict of tensors:
      HI, HII, HeI, HeII, HeIII

    No clipping is applied in this draft. Unphysical values should remain
    visible during development.
    """
    HI = y[..., 0]
    HeI = y[..., 1]
    HeII = y[..., 2]

    HII = 1.0 - HI
    HeIII = 1.0 - HeI - HeII

    return {
        "HI": HI,
        "HII": HII,
        "HeI": HeI,
        "HeII": HeII,
        "HeIII": HeIII,
    }
