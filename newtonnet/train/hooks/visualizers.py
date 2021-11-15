import os
import torch
import torch.nn.functional as F

import numpy as np
import plotly
import matplotlib.pyplot as plt
import plotly.express as px


class VisHook(object):
    """
    Base class for visualization hooks.
    """

    def create_grid(self, ndim, length, size):
        half_length = length / 2.
        offsets = torch.linspace(-half_length, half_length, size)
        if ndim == 3:
            grids = torch.meshgrid([offsets, offsets, offsets])
        elif ndim == 2:
            grids = torch.meshgrid([offsets, offsets])

        grid = torch.stack(grids, dim=-1)  # shape: G,G,G,3

        grid = grid.data.cpu().numpy()
        # grid = grid.reshape(-1, 2)
        return grid

    def set_output(self, show=True, path2save=None):
        """

        Parameters
        ----------
        show: bool
            If True, show the plot in the console.
        path: str
            path to the folder to save the figure.

        Returns
        -------

        """
        self.show = show
        self.path2save = path2save
        if path2save is not None:
            if not os.path.exists(path2save):
                os.makedirs(path2save)


    def get_2d_rot_mat(self, theta):
        """
        Affine transformation matrix for counter clockwise rotation around origin.

        Parameters
        ----------
        theta: float
            degree in float


        Returns
        -------

        """
        theta = np.radians(theta)
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

    def rot_grid(self, x, theta, dtype, device):
        rot_mat = self.get_2d_rot_mat(theta)
        rot_mat = rot_mat[None, ...].type(dtype).repeat(x.shape[0],1,1)
        grid = F.affine_grid(rot_mat, x.size()).type(dtype)
        grid = grid.to(device)
        x = F.grid_sample(x, grid)
        return x


class Vis3DVoxelCartesian(VisHook):
    def __init__(self, grid_size, grid_length):
        self.grid = self.create_grid(3, grid_length, grid_size)

    def run(self, input):
        assert input.shape == self.grid.shape


class VizMolVectors3D(VisHook):

    def __init__(self):
        pass

    def run(self, R, Z, F, dF, dR):
        """
        Parameters
        ----------
        R: ndarray
            shape: A,3
        Z: ndarray
            shape: A,
        F: ndarray
            shape: A,3
        dF: ndarray
            shape: A,3
        dR: ndarray
            shape: A,3

        """
        if isinstance(R, torch.Tensor):
            R = R.data.cpu().numpy()
            Z = Z.data.cpu().numpy()
            F = F.data.cpu().numpy()
            dF= dF.data.cpu().numpy()
            dR= dR.data.cpu().numpy()

        # move R
        R_mean = np.mean(R, axis=0)
        R = R - R_mean
        # xlim, ylim, zlim = np.max(R, axis=0)

        # normalize vectors
        fmax = np.linalg.norm(F, axis=1).max()
        fmax += 1e-8
        F = F / fmax
        # dF = dF / fmax
        # dR = dR / fmax

        F *= 2
        dF *= 4

        # fig = plt.figure()
        ax = plt.figure().add_subplot(projection='3d')

        # cpk coloring
        atom_color = {1: 'gray', 8: 'r', 6: 'k', 7: 'b', 16: 'y'}

        ax.quiver(R[:, 0], R[:, 1], R[:, 2], F[:, 0], F[:, 1], F[:, 2], color='g', **{'linewidth':1.5})  # , scale=21
        ax.quiver(R[:, 0], R[:, 1], R[:, 2], dF[:, 0], dF[:, 1], dF[:, 2], color='b',**{'linewidth':1.5})  # , scale=21
        if dR is not None and dR.ndim == dF.ndim:
            ax.quiver(R[:, 0], R[:, 1], R[:, 2], dR[:, 0], dR[:, 1], dR[:, 2], color='r')  # , scale=21
        ax.scatter(R[:, 0], R[:, 1], R[:, 2], color=[atom_color[z] for z in Z], s=250, alpha=0.8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Hide grid lines
        ax.grid(False)
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        # ax.set_zlim(-5, 5)

        if self.show:
            plt.show()

        if self.path2save is not None:
            plt.savefig(os.path.join(self.path2save, "mol_3d_vectors.png"), dpi=300)
            plt.savefig(os.path.join(self.path2save, "mol_3d_vectors.eps"), dpi=300)



