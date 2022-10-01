from rdkit import Chem
import numpy as np
import torch
from torch import zeros

class BasicMolecularMetrics(object):
    def __init__(self, atom_dict, bond_dict, training_dataset = None):
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict

        if training_dataset is not None:
            print("Processing dataset to Smiles")
            self.dataset_smiles_list = []
            for graph in training_dataset:
                n_nodes = graph['n_nodes']
                X = torch.argmax(graph["node_features"][:n_nodes], dim=-1)
                A = graph["adj"][:n_nodes, :n_nodes]
                E = torch.argmax(graph["edge_features"][:n_nodes, :n_nodes], dim=-1)
                print(E)
                mol = self.build_molecule(X, A, E)
                smiles = self.toSmiles(mol)
                if smiles is not None:
                    self.dataset_smiles_list.append(smiles)
            print("Done!")

    def build_molecule(self, X, A, E):
        assert len(X.shape) == 1
        assert len(A.shape) == 2
        assert len(E.shape) == 2
        mol = Chem.RWMol()
        for atom in X:
            a = Chem.Atom(self.atom_dict[atom.item()])
            mol.AddAtom(a)

        all_bonds = torch.nonzero(torch.tril(A, diagonal=-1))
        for bond in all_bonds:
            mol.AddBond(bond[0].item(), bond[1].item(), self.bond_dict[E[bond[0], bond[1]].item()])

        return mol

    def toSmiles(self, mol):
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return Chem.MolToSmiles(mol)

    def compute_validity(self, generated):
        """ generated: list of triplets (X, A, E)"""
        valid = []

        for graph in generated:
            mol = self.build_molecule(*graph)
            smiles = self.toSmiles(mol)
            if smiles is not None:
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def plot(self, valid, filename):
        for i, mol in enumerate(valid):
            svg = Chem.Draw.MolsToGridImage(valid, molsPerRow=1, useSVG=True)
            filename += f'_{i}.svg'
            with open(filename, 'w') as f:
                f.write(svg)

    def evaluate(self, generated, filename=None):
        valid, validity = self.compute_validity(generated)
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if filename is not None:
            self.plot(valid)
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                novel, novelty = self.compute_novelty(unique)
                print(f"Uniqueness over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
                novel = None
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
            novel = None
        return [validity, uniqueness, novelty], unique
