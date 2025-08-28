import torch
from utils import MolecularMetrics
import torch.nn.functional as F
from sparse_molecular_dataset import SparseMolecularDataset
import numpy as np
import rdkit
from rdkit import Chem
from pysmiles import read_smiles
import string
import random

data = SparseMolecularDataset()
data.load("qm9_5k.sparsedataset")

b_dim = data.bond_num_types
m_dim = data.atom_num_types

def gradient_penalty(y, x):
        weight = torch.ones(y.size())
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

def label2onehot(labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim])
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

def sample_z(batch_size):
        return np.random.normal(0, 1, size=(batch_size, 32))

def postprocess(inputs, method = "softmax", temperature=1.):
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]
    
def reward(mols,metric ="qed,unique,validity,diversity"):
        rr = 1.
        for m in ('logp,sas,qed,unique' if metric == 'all' else metric).split(','):

            if m == 'np':
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == 'logp':
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == 'sas':
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == 'qed':
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == 'novelty':
                rr *= MolecularMetrics.novel_scores(mols, data)
            elif m == 'dc':
                rr *= MolecularMetrics.drugcandidate_scores(mols, data)
            elif m == 'unique':
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == 'diversity':
                rr *= MolecularMetrics.diversity_scores(mols, data)
            elif m == 'validity':
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise RuntimeError('{} is not defined as a metric'.format(m))

        return rr.reshape(-1, 1)
    
def get_reward(n_hat, e_hat, method = "soft_gumbel"):
        (edges_hard, nodes_hard) = postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        rewards = torch.from_numpy(reward(mols))
        return rewards
    
def get_gen_mols( n_hat, e_hat):
        (edges_hard, nodes_hard) = postprocess((e_hat, n_hat))
        edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
        mols = [data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                for e_, n_ in zip(edges_hard, nodes_hard)]
        return mols

def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'NP': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'QED': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'Solute': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        'SA': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
        'diverse': MolecularMetrics.diversity_scores(mols, data),
        'drugcand': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

    m1 = {'valid': MolecularMetrics.valid_total_score(mols) * 100,
          'unique': MolecularMetrics.unique_total_score(mols) * 100,
          'novel': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1

def random_string(string_len=3):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_len))

def save_mol_img(mols, f_name='tmp.png', is_test=False):
    orig_f_name = f_name
    for a_mol in mols:
        try:
            if Chem.MolToSmiles(a_mol) is not None:
                print('Generating molecule')

                if is_test:
                    f_name = orig_f_name
                    f_split = f_name.split('.')
                    f_split[-1] = random_string() + '.' + f_split[-1]
                    f_name = ''.join(f_split)

                rdkit.Chem.Draw.MolToFile(a_mol, f_name)
                a_smi = Chem.MolToSmiles(a_mol)
                mol_graph = read_smiles(a_smi)

                break

                # if not is_test:
                #     break
        except:
            continue