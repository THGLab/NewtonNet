import numpy as np
from ase.io import read
from newtonnet.utils.ase_interface import MLAseCalculator


CHARGE_FIELD_NAMES = {'charge', 'q', 'charges', 'initial_charges', 'mbi_charges'}

def read_dft_charges(xyz_path, frame_index=0):
    """Parse per-atom charges from extxyz column, bypassing ASE parser."""
    with open(xyz_path) as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                return None, None
            n = int(line.strip())
            comment = f.readline()
            atom_lines = [f.readline() for _ in range(n)]
            if i == frame_index:
                props_str = [p for p in comment.split() if p.startswith('Properties=')]
                if not props_str:
                    return None, None
                fields = props_str[0].split('=', 1)[1].split(':')
                col = 0
                field_name = None
                for j in range(0, len(fields) - 2, 3):
                    name, typ, count = fields[j], fields[j+1], int(fields[j+2])
                    if name in CHARGE_FIELD_NAMES:
                        field_name = name
                        break
                    col += count
                if field_name is None:
                    return None, None
                charges = np.array([float(l.split()[col]) for l in atom_lines])
                return charges, field_name
            i += 1


xyz_path = 'electrolyte_data/test/raw/electrolyte_test.xyz'
atoms = read(xyz_path, index=0)
dft, field_name = read_dft_charges(xyz_path, frame_index=0)

calc = MLAseCalculator(
    model_path='electrolyte_model_les/training_1/models/best_model.pt',
    properties=['charges', 'energy', 'forces'],
    precision='double',
    device='cuda',
)
atoms.calc = calc
atoms.get_potential_energy()

predicted = atoms.calc.results['charges']
symbols = atoms.get_chemical_symbols()

if dft is not None:
    print(f"DFT charge field: '{field_name}'")
    print(f"{'Atom':<6} {'Symbol':<8} {'Predicted':>12} {'DFT':>10}")
    print("-" * 38)
    for i, (sym, q_pred, q_dft) in enumerate(zip(symbols, predicted, dft)):
        print(f"{i:<6} {sym:<8} {q_pred:>12.4f} {q_dft:>10.4f}")
else:
    print(f"{'Atom':<6} {'Symbol':<8} {'Predicted':>12}")
    print("-" * 28)
    for i, (sym, q) in enumerate(zip(symbols, predicted)):
        print(f"{i:<6} {sym:<8} {q:>12.4f}")
