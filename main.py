from A_DataPreperation.AU_util import *
from B_Train.BA_train import train_AREVGA_ZINC
from D_Optimization.DA_sample import *


if __name__ == '__main__':
    prepare_ZINC_dataset_npz()
    prepare_pre_train_dataset_npz()
    prepare_Augment_dataset_npz()
    train_AREVGA_ZINC()
    bias = torch.tensor([1.1, 1, 0.9, 1, 1.12, 1.1, 1]).to(dtype)
    condition = [5, 2, 5, 3.5, 4, 5, 1]
    random_search(bias=bias, burte_force_link=False)
