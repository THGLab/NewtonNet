import numpy as np
from ase.calculators.calculator import Calculator

import torch
from torch_geometric.nn import radius_graph


##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLAseCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'hessian']

    ### Constructor ###
    def __init__(
            self, 
            model_path: str,
            properties: list = ['energy', 'forces'], 
            disagreement: str = 'std',
            device: str | list[str] = 'cpu',
            script: bool = False,
            trace_n_atoms: int = 0,
            **kwargs,
            ):
        """
        Constructor for MLAseCalculator for NewtonNet models

        Parameters:
            model_path (str): The path to the model.
            settings_path (str): The path to the settings.
            properties (list): The properties to be predicted. Default: ['energy', 'forces'].
            disagreement (str): The type of disagreement to be calculated. Default: 'std'.
            device (str): The device for the calculator. Default: 'cpu'.
            script (bool): Whether to script the model. Default: False.
            trace_n_atoms (int): The number of atoms to be traced. Default: 0.
        """
        Calculator.__init__(self, **kwargs)

        if type(device) is list:
            self.device = [torch.device(item) for item in device]
        else:
            self.device = [torch.device(device)]

        self.models = []
        self.properties = properties
        self.dtype = None
        if type(model_path) is not list:
            model_path = [model_path]
        for model in model_path:
            model = torch.load(model, map_location=self.device[0])
            # for key in self.properties:
            #     if key in model.output_layers.keys():
            #         continue
            #     output_layer = get_output_by_string(key)
            #     model.output_layers.update({key: output_layer})
            #     if isinstance(output_layer, FirstDerivativeProperty):
            #         model.embedding_layer.requires_dr = True
            #     if isinstance(output_layer, SecondDerivativeProperty):
            #         assert output_layer.dependent_property in model.output_layers.keys(), f'cannot find dependent property {output_layer.dependent_property}'
            #         model.output_layers[output_layer.dependent_property].requires_dr = True
            self.dtype = next(model.named_parameters())[1].dtype
            self.models.append(model)
        
        self.script = script
        self.trace_n_atoms = trace_n_atoms
        self.disagreement = disagreement
        

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=None):
        super().calculate(atoms, self.properties, system_changes)
        preds = {}
        if 'energy' in self.properties:
            preds['energy'] = np.zeros(len(self.models))
        if 'forces' in self.properties:
            preds['forces'] = np.zeros((len(self.models), len(atoms), 3))
        z = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device[0])
        pos = torch.tensor(atoms.positions, dtype=torch.float, device=self.device[0])
        try:
            edge_index = torch.tensor(atoms.edge_index, dtype=torch.long, device=self.device[0])
            disp = torch.tensor(atoms.disp, dtype=torch.float, device=self.device[0])
            # print('using precomputed edge_index')
        except AttributeError:
            # edge_index = torch.tensor(
            #     np.stack(neighbor_list('ij', atoms, cutoff=float(self.models[0].embedding_layer.norm.r))), 
            #     dtype=torch.long, device=self.device[0])
            edge_index = radius_graph(
                pos,
                self.models[0].embedding_layer.norm.r,
                max_num_neighbors=1024,
            )
            disp = pos[edge_index[0]] - pos[edge_index[1]]
            # print('using radius_graph')
        batch = torch.zeros_like(z, dtype=torch.long, device=self.device[0])
        for model_, model in enumerate(self.models):
            pred = model(z, disp, edge_index, batch)
            if 'energy' in self.properties:
                preds['energy'][model_] = pred.energy.cpu().detach().numpy()
            if 'forces' in self.properties:
                preds['forces'][model_] = pred.gradient_force.cpu().detach().numpy()
            del pred

        self.results['outlier'] = self.q_test(preds['energy'])
        for key in self.properties:
            self.results[key] = self.remove_outlier(preds[key], self.results['outlier']).mean(axis=0)

            if self.disagreement == 'std':
                self.results[key + '_disagreement'] = preds[key].std(axis=0).max()
            elif self.disagreement == 'std_outlierremoval':
                self.results[key + '_disagreement'] = self.remove_outlier(preds[key], self.results['outlier']).std(axis=0).max()
            elif self.disagreement == 'range':
                self.results[key + '_disagreement'] = (preds[key].max(axis=0) - preds[key].min(axis=0)).max()
            elif self.disagreement == 'values':
                self.results[key + '_disagreement'] = preds[key]
            del preds[key]

    def remove_outlier(self, data, idx):
        if idx is None:
            return data
        else:
            return np.delete(data, idx, axis=0)

    def q_test(self, data):
        '''
        Dixon's Q test for outlier detection

        Parameters:
            data (1d array): The data to be tested.

        Returns:
            idx (int): The index to be filtered out.
        '''
        if len(data) < 3:
            idx = None
        else:
            q_ref = { 3: 0.970,  4: 0.829,  5: 0.710, 
                      6: 0.625,  7: 0.568,  8: 0.526,  9: 0.493, 10: 0.466, 
                     11: 0.444, 12: 0.426, 13: 0.410, 14: 0.396, 15: 0.384, 
                     16: 0.374, 17: 0.365, 18: 0.356, 19: 0.349, 20: 0.342, 
                     21: 0.337, 22: 0.331, 23: 0.326, 24: 0.321, 25: 0.317, 
                     26: 0.312, 27: 0.308, 28: 0.305, 29: 0.301, 30: 0.290}.get(len(self.models))  # 95% confidence interval
            sorted_data = np.sort(data, axis=0)
            q_stat_min = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
            q_stat_max = (sorted_data[-1] - sorted_data[-2]) / (sorted_data[-1] - sorted_data[0])
            if q_stat_min > q_ref:
                idx = np.argmin(data)
            elif q_stat_max > q_ref:
                idx = np.argmax(data)
            else:
                idx = None
        return idx
    