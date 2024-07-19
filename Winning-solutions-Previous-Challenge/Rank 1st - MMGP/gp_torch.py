#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#

import numpy as np
import torch
import torch.nn as nn

class GaussianProcessRegressor(nn.Module):
    def __init__(
        self,
        length_scale=1.0,
        noise_scale=1.0,
        amplitude_scale=1.0,
    ):
        super().__init__()
        if isinstance(length_scale, float):
            length_scale = np.array([length_scale])
        elif isinstance(length_scale, np.ndarray):
            assert length_scale.ndim == 1
        else:
            raise TypeError()

        self.register_parameter(
            "length_scale_",
            param=nn.Parameter(torch.Tensor(np.log(length_scale)), requires_grad=True),
        )
        self.register_parameter(
            "noise_scale_",
            param=nn.Parameter(torch.tensor(np.log(noise_scale)), requires_grad=True),
        )
        self.register_parameter(
            "amplitude_scale_",
            param=nn.Parameter(
                torch.tensor(np.log(amplitude_scale)), requires_grad=True
            ),
        )

        self.nll = None

    def forward(self, x):
        alpha = self.alpha
        k = self.Kxy(self.X, x)
        mu = k.T.mm(alpha)
        return mu

    def log_marginal_likelihood(self, X, y):
        D = X.shape[1]
        K = self.Kxx(X)
        L = torch.linalg.cholesky(K)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        marginal_likelihood = (
            -0.5 * y.T.mm(alpha)
            - torch.log(torch.diag(L)).sum()
            - D * 0.5 * np.log(2 * np.pi)
        )
        self.L = L
        self.alpha = alpha
        self.K = K
        return marginal_likelihood

    def Kxx(self, X):
        param = self.length_scale_.exp().sqrt()
        sqdist = torch.cdist(X / param[None], X / param[None]) ** 2

        res = self.amplitude_scale_.exp() * torch.exp(-0.5 * sqdist) + self.noise_scale_.exp() * torch.eye(len(X)).type_as(X)

        return res

    def Kxy(self, X, Z):
        param = self.length_scale_.exp().sqrt()
        sqdist = torch.cdist(X / param[None], Z / param[None]) ** 2
        res = self.amplitude_scale_.exp() * torch.exp(-0.5 * sqdist)

        return res

    def fit(self, X, y, opt, num_steps):
        assert X.shape[1] == len(self.length_scale_)
        self.y = y
        self.X = X

        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

        self.train()
        nll_hist = []
        for it in range(num_steps):
            opt.zero_grad()
            try:
                nll = -self.log_marginal_likelihood(self.X, self.y).sum()
            except torch.linalg.LinAlgError:
                break
            nll.backward()
            opt.step()
            if it%10==0 and it<1000:
                scheduler.step()
            nll_hist.append(nll.item())
        return nll_hist