import torch
import pyxtal
from eval_utils import load_model, get_crystals_list, lattices_to_params_shape
from pathlib import Path
from pyxtal.symmetry import Group
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from p_tqdm import p_map
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
import json

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

rev_chemical_symbols = {ch:i for i,ch in enumerate(chemical_symbols)}

def get_data_from_syminfo(spacegroup_number, wyckoff_letters, atom_types = None):
    
    g = Group(spacegroup_number)
    ops_tot = []
    anchor_index = []
    num_atoms = 0
    if atom_types is not None:
        assert len(wyckoff_letters) == len(atom_types)
        atom_numbers = []
    for idx in range(len(wyckoff_letters)):
        letter = wyckoff_letters[idx][-1] # 'a' for '1a'
        ops = g[letter].ops
        for op in ops:
            ops_tot.append(op.affine_matrix)
            anchor_index.append(num_atoms)
            if atom_types is not None:
                atom_numbers.append(rev_chemical_symbols[atom_types[idx]])
        num_atoms += len(ops)
    data = Data(
        spacegroup = torch.LongTensor([spacegroup_number]),
        ops = torch.FloatTensor(ops_tot),
        anchor_index = torch.LongTensor(anchor_index),
        num_nodes = num_atoms,
        num_atoms = num_atoms,
    )
    data.ops_inv = torch.linalg.pinv(data.ops[:,:3,:3])
    if atom_types is not None:
        data.atom_types = torch.LongTensor(atom_numbers)
    else:
        data.atom_types = torch.zeros(num_atoms)
    return data

class CustomDataset(Dataset):

    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list


    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def diffusion(loader, model, step_lr):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms
    )

def get_pymatgen(crystal_array):
    frac_coords = crystal_array['frac_coords']
    atom_types = crystal_array['atom_types']
    lengths = crystal_array['lengths']
    angles = crystal_array['angles']
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths.tolist() + angles.tolist())),
            species=atom_types, coords=frac_coords, coords_are_cartesian=False)
        return structure
    except:
        return None

def construct_dataset_from_syminfo(spacegroup_number, wyckoff_letters, atom_types = None):

    data = get_data_from_syminfo(spacegroup_number, wyckoff_letters, atom_types)
    dataset = CustomDataset([data])
    return dataset

def construct_dataset_from_json(json_file):

    with open(json_file, 'r') as f:
        json_list = json.load(f)

    data_list = []

    for idx, json_data in tqdm(enumerate(json_list)):
        try:
            data = get_data_from_syminfo(**json_data)
            data_list.append(data)
        except:
            print(f"Parsing Json with index {idx} failed. Skipped.")

    print(f"Collected {len(data_list)} queries.")

    dataset = CustomDataset(data_list)

    return dataset

def generate_structures_from_dataset(model_path, dataset, batch_size = 128, step_lr = 1e-5):

    model_path = Path(model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)

    if torch.cuda.is_available():
        model.to('cuda')
    loader = DataLoader(dataset, batch_size = min(batch_size, len(dataset)))
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(loader, model, step_lr)
    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    structure_list = p_map(get_pymatgen, crystal_list)
    return structure_list


    