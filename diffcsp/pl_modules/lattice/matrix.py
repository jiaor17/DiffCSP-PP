import torch
import torch.linalg as linalg
import pdb

def logm(A):
    det = torch.det(A)
    mask = ~(det > 0)
    b = mask.sum()
    A[mask] = torch.eye(3).unsqueeze(0).to(A).expand(b, -1, -1)
    eigenvalues, eigenvectors = linalg.eig(A)
    return torch.einsum("bij,bj,bjk->bik", eigenvectors, eigenvalues.log(), torch.linalg.inv(eigenvectors)).real

def expm(A):
    return torch.matrix_exp(A)

def sqrtm(A):
    det = torch.det(A)
    mask = ~(det > 0)
    b = mask.sum()
    A[mask] = torch.eye(3).unsqueeze(0).to(A).expand(b, -1, -1)    
    eigenvalues, eigenvectors = linalg.eig(A)
    return torch.einsum("bij,bj,bjk->bik", eigenvectors, eigenvalues.sqrt(), torch.linalg.inv(eigenvectors)).real