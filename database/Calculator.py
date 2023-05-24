from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromXYZFile
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np


def mol_from_xyz(xyz):
    m = MolFromXYZFile(xyz)

    try:
        Chem.SanitizeMol(m)
    except:
        assert "Molecule cannot be sanitized"

    return m


def auto_correct_descs(descs):
    descs = np.nan_to_num(descs)
    descs[descs > 10 ** 5] = 0
    descs[descs < -10 ** 5] = 0
    return descs


class RDKitDescriptors:
    def __init__(self, auto_correct=True, dict_mode=True):
        self.desc_list = [desc_name[0] for desc_name in Descriptors.descList]
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self.desc_list)
        self.desc_list = ["RDKit_desc_" +
                          desc_name for desc_name in self.desc_list]
        self.auto_correct = auto_correct
        self.dict_mode = dict_mode

    def _desc_calc(self, xyz):
        m = mol_from_xyz(xyz)
        return self.calculator.CalcDescriptors(m)

    # calc descriptors
    def calc(self, xyz):
        """
        xyz: xyz file path
        dict_mode: if true, return type will be a dict, otherwise, list
        """

        # calculate newly
        descs = self._desc_calc(xyz)

        if self.auto_correct:
            descs = auto_correct_descs(descs)

        desc_dict = {k: v for k, v in zip(self.desc_list, descs)}

        if self.dict_mode:
            return desc_dict
        else:
            return descs

    # calc descriptors from xyz list
    def calc_list(self, ls, pandas_mode=True):
        """
        ls: list of xyz
        pandas_mode: if true, return type will be dataframe, otherwise list
        """
        temp_mode = self.dict_mode
        self.dict_mode = False
        res_list = [self.calc(i) for i in ls]
        self.dict_mode = temp_mode

        if pandas_mode:
            df = pd.DataFrame(res_list)
            df.columns = self.desc_list
            return df

        return res_list

    def auto_calc(self, arg, pandas_mode=True):
        if type(arg) is str:
            return self.calc(arg)
        elif type(arg) is list:
            return self.calc_list(arg, pandas_mode=pandas_mode)
        else:
            assert False, ("unexpected type: ", type(arg))


class AutoDescriptor:
    def __init__(self, calculators=[RDKitDescriptors()]):
        self.calculators = calculators
        # self.descriptor_names = list(self.__call__("give some base XYZ file here to make this work").columns)
        # self.descriptor_names.remove("XYZ")

    def __call__(self, xyz_list):
        if type(xyz_list) is str:
            xyz_list = [xyz_list]
        elif type(xyz_list) is list:
            pass
        else:
            assert False, ("unexpected type: ", type(xyz_list))

        integ_pd = pd.DataFrame()
        integ_pd["XYZ"] = xyz_list
        for num, calculator in enumerate(self.calculators):
            df = (calculator.auto_calc(xyz_list))
            integ_pd = pd.concat([integ_pd, df], axis=1)

        return integ_pd
