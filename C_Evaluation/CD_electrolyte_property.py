from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit import DataStructs

from tqdm import trange, tqdm

import pandas as pd
import numpy as np
import csv


Electrolyte_SMILES_path = 'DataA_InputData/electrolyte.txt'
QM_structure_path = 'DataA_InputData/gdb9.sdf'
QM_property_path = 'DataA_InputData/gdb9.sdf.csv'


def write_property():
    with open(Electrolyte_SMILES_path, 'r') as txt_component_smiles:
        component_smiles = txt_component_smiles.readlines()
        ELE_structure_file = [Chem.MolFromSmiles(cs[:-1]) for cs in component_smiles]

    QM_structure_file = Chem.SDMolSupplier(QM_structure_path, removeHs=True)
    QM_property_file = pd.read_csv(QM_property_path)
    property_list = np.zeros((len(ELE_structure_file), 4))

    for i in range(len(ELE_structure_file)):
        ele_mol = ELE_structure_file[i]
        for j in trange(len(QM_structure_file)):
            qm_mol = QM_structure_file[j]
            sim = 0
            try:
                fp_ele = AllChem.GetMorganFingerprintAsBitVect(ele_mol, 2)
                fp_qm = AllChem.GetMorganFingerprintAsBitVect(qm_mol, 2)
                sim = DataStructs.DiceSimilarity(fp_ele, fp_qm)
            except:
                pass
            if sim >= 0.99:
                property_list[i] = [0, QM_property_file['alpha'][j], QM_property_file['homo'][j], QM_property_file['lumo'][j]]
                break

    property_list_list = property_list.tolist()
    for i in range(len(ELE_structure_file)):
        property_list_list[i][0] = component_smiles[i]

    with open('DataD_OutputData/Electrolyte_properties.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES', 'Alpha', 'HOMO', 'LUMO'])
        writer.writerows(property_list_list)


if __name__ == '__main__':
    write_property()
