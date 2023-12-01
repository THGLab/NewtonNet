import torch
import torch.nn as nn

class EnergyForceLoss(nn.Module):
    def __init__(
            self, 
            w_energy: float = 1.0,
            w_force: float = 0.0,
            w_f_mag: float = 0.0,
            w_f_dir: float = 0.0,
            wf_decay: float = 0.0,
            ):
        super(EnergyForceLoss, self).__init__()
        self.w_energy = w_energy
        self.w_force = w_force
        self.w_f_mag = w_f_mag
        self.w_f_dir = w_f_dir
        self.wf_decay = torch.tensor(wf_decay, dtype=torch.float)

    def forward(self, preds, batch_data):

        # compute the mean squared error on the energies
        if self.w_energy > 0:
            diff_energy = preds['E'] - batch_data['E']
            err_sq_energy = torch.mean(diff_energy ** 2)
            err_sq = self.w_energy * err_sq_energy

        # compute the mean squared error on the forces
        if self.w_force > 0:
            diff_forces = preds['F'] - batch_data['F']
            err_sq_forces = torch.mean(diff_forces ** 2)
            err_sq = err_sq + self.w_force * err_sq_forces

        # compute the mean square error on the force magnitudes
        if self.w_f_mag > 0:
            diff_forces = torch.norm(preds['F'], p=2, dim=-1) - torch.norm(batch_data['F'], p=2, dim=-1)
            err_sq_mag_forces = torch.mean(diff_forces ** 2)
            err_sq = err_sq + self.w_f_mag * err_sq_mag_forces

        # compute the mean square error on the hidden force directions
        if self.w_f_dir > 0:
            cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            direction_diff = 1 - cos(preds['F_latent'], batch_data['F'])
            # direction_diff = direction_diff * torch.norm(batch_data["F"], p=2, dim=-1)
            direction_loss = torch.mean(direction_diff)
            err_sq = err_sq + self.w_f_dir * direction_loss

        return err_sq
    
    def force_loss_decay(self):
        self.w_force = self.w_force * torch.exp(-self.wf_decay)