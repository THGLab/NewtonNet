import torch
from torch import nn


def get_representation_by_string(cutoff, cutoff_network='poly', radial_network='bessel', n_basis=20):
    representations = {}
    representations['norm'] = ScaledNorm(r=cutoff)

    if cutoff_network == 'poly':
        representations['cutoff'] = PolynomialCutoff(p=9)
    elif cutoff_network == 'cos':
        representations['cutoff'] = CosineCutoff()
    else:
        raise NotImplementedError(f'The cutoff function {cutoff_network} is unknown.')
    
    if radial_network == 'bessel':
        representations['radial'] = RadialBesselLayer(n_basis=n_basis)
    else:
        raise NotImplementedError(f'The radial function {radial_network} is unknown.')
    
    return representations


class ScaledNorm(nn.Module):
    '''
    Compute scaled norm of interatomic distances.
    Based on Johannes Klicpera, Janek Grob, Stephan Gunnemann. Directional Message Passing for Molecular Graphs. ICLR 2020.

    Parameters:
        r (float): cutoff radius.
    '''
    def __init__(self, r, **kwargs):
        super().__init__()
        self.r = r

    def forward(self, disp):
        """Compute scaled norm.

        Args:
            disp (torch.Tensor): values of interatomic distance vectors.

        Returns:
            torch.Tensor: values of scaled norm.

        """
        # Compute values of scaled norm
        dist = torch.norm(disp, dim=-1, keepdim=True)
        dir = disp / dist
        dist = dist / self.r

        return dist, dir

class PolynomialCutoff(nn.Module):
    '''
    Compute polynomial cutoff function.
    Based on Johannes Klicpera, Janek Grob, Stephan Gunnemann. Directional Message Passing for Molecular Graphs. ICLR 2020.

    Parameters:
        p (int): degree of polynomial. Default: 9.

    Notes:
        y = 1 - 0.5 * (p + 1) * (p + 2) * x^p + p * (p + 2) * x^(p + 1) - 0.5 * p * (p + 1) * x^(p + 2)
        y(0) = 1
        y(1) = 0
    '''
    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, dist):
        """Compute cutoff.

        Args:
            dist (torch.Tensor): values of scaled interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 1 \
            - 0.5 * (self.p + 1) * (self.p + 2) * dist.pow(self.p) \
            + self.p * (self.p + 2) * dist.pow(self.p + 1) \
            - 0.5 * self.p * (self.p + 1) * dist.pow(self.p + 2)

        return cutoffs


class CosineCutoff(nn.Module):
    '''
    Compute Behler cosine cutoff function.
    Copied from: https://github.com/atomistic-machine-learning/schnetpack under the MIT License.

    Notes:
        y = 0.5 * (1 + cos(pi * x))
        y(0) = 1
        y(1) = 0
    '''
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, dist):
        """Compute cutoff.

        Args:
            dist (torch.Tensor): values of scaled interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(dist * np.pi) + 1.0)

        return cutoffs


class RadialBesselLayer(nn.Module):
    '''
    Radial Bessel functions based on the work by DimeNet: https://github.com/klicperajo/dimenet

    Parameters:
        n_basis (int): Total number of radial functions. Default: 16.

    Notes:
        y = sin(pi * r) / (pi * r)
    '''

    def __init__(self, n_basis):
        super().__init__()
        self.n_basis = n_basis
        self.frequencies = nn.Parameter(torch.arange(1, self.n_basis + 1) * torch.pi, requires_grad=False)
        self.epsilon = 1.0e-8

    def forward(self, dist):
        '''
        Compute smeared-gaussian distance values.

        Arguments:
            dist (torch.Tensor): interatomic distance values of shape (batch_size, n_atoms, n_atoms).

        Returns:
            torch.Tensor: edge embedding of shape (batch_size, n_atoms, n_atoms, n_radials).
        '''
        out = torch.sin(self.frequencies * dist) / dist #/ self.frequencies

        return out
