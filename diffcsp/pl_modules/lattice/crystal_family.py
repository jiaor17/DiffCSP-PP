import torch
import torch.nn as nn
import numpy as np
import math
from diffcsp.pl_modules.lattice.matrix import logm, expm, sqrtm

class CrystalFamily(nn.Module):

    def __init__(self):

        super(CrystalFamily, self).__init__()

        basis = self.get_basis()
        masks, biass = self.get_spacegroup_constraints()
        family = self.get_family_idx()

        self.register_buffer('basis', basis)
        self.register_buffer('masks', masks)
        self.register_buffer('biass', biass)
        self.register_buffer('family', family)

    def get_basis(self):

        basis = torch.FloatTensor([
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ])

        # Normalize
        basis = basis / basis.norm(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)

        return basis

    def get_spacegroup_constraint(self, spacegroup):

        mask = torch.ones(6)
        bias = torch.zeros(6)

        if 195 <= spacegroup <= 230:
            pos = [0,1,2,3,4]
            mask[pos] = 0.

        elif 143 <= spacegroup <= 194:
            pos = [0,1,2,3]
            mask[pos] = 0.
            bias[0] = -0.25 * np.log(3) * np.sqrt(2)

        elif 75 <= spacegroup <= 142:
            pos = [0,1,2,3]
            mask[pos] = 0.

        elif 16 <= spacegroup <= 74:
            pos = [0,1,2]
            mask[pos] = 0.

        elif 3 <= spacegroup <= 15:
            pos = [0,2]
            mask[pos] = 0.

        elif 0 <= spacegroup <= 2:
            pass
            
        return mask, bias

    def get_spacegroup_constraints(self):

        masks, biass = [], []

        for i in range(231):
            mask, bias = self.get_spacegroup_constraint(i)
            masks.append(mask.unsqueeze(0))
            biass.append(bias.unsqueeze(0))

        return torch.cat(masks, dim = 0), torch.cat(biass, dim = 0)

    def get_family_idx(self):

        family = []
        for spacegroup in range(231):
            if 195 <= spacegroup <= 230:
                family.append(6)

            elif 143 <= spacegroup <= 194:
                family.append(5)

            elif 75 <= spacegroup <= 142:
                family.append(4)

            elif 16 <= spacegroup <= 74:
                family.append(3)

            elif 3 <= spacegroup <= 15:
                family.append(2)

            elif 0 <= spacegroup <= 2:
                family.append(1)
        return torch.LongTensor(family)

    def de_so3(self, L):

        # L: B * 3 * 3

        LLT = L @ L.transpose(-1,-2)
        L_sym = sqrtm(LLT)
        return L_sym

    def v2m(self, vec):

        batch_size, dims = vec.shape
        if dims == 6:
            basis = self.basis
        elif dims == 5:
            basis = self.basis[:-1]
        log_mat = torch.einsum('bk, kij -> bij', vec, basis)
        mat = expm(log_mat)
        return mat

    def m2v(self, mat):

        # mat: B * 3 * 3

        log_mat = logm(mat)
        vec = torch.einsum('bij, kij -> bk', log_mat, self.basis)
        return vec

    def proj_k_to_spacegroup(self, vec, spacegroup):

        batch_size, dims = vec.shape
        if dims == 6:
            masks = self.masks[spacegroup, :] # B * 6
            biass = self.biass[spacegroup, :] # B * 6  
        elif dims == 5:
            # - volume
            masks = self.masks[spacegroup, :-1] # B * 5
            biass = self.biass[spacegroup, :-1] # B * 5            
        return vec * masks + biass



        