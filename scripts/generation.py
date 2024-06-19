import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import pdb

import os


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

class SampleDataset(Dataset):

    def __init__(self, dataset, total_num, seed = 9999):
        super().__init__()
        self.total_num = total_num
        self.dataset = dataset
        self.is_carbon = dataset == 'carbon'
        self.seed = seed
        np.random.seed(self.seed)
        self.indexes = np.random.choice(len(self.dataset), total_num)


    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        idx = self.indexes[index]
        data = self.dataset[idx]
        return data


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, loaders, cfg = load_model(
        model_path, load_data=True, testing=False)


    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')


    train_loader, val_loader = loaders

    train_set = train_loader.dataset

    test_set = SampleDataset(train_set, args.batch_size * args.num_batches_to_samples, seed = args.seed)
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms) = diffusion(test_loader, model, args.step_lr)

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
    }, model_path / gen_out_name)
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--seed', default=9999, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)
