"""MNIST dataset utilities for federated learning."""


from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST


def load_datasets(  # pylint: disable=too-many-arguments
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between the
        clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    datasets, testset = _partition_data(num_clients, iid, balance, seed)
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)


def _download_data() -> Tuple[Dataset, Dataset]:
    """Downloads (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # # OLD, for FL
    # trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    # testset = MNIST("./dataset", train=False, download=True, transform=transform)

    # NEW, for nuScenes
    # TODO: Add solution for loading img_tensor_list etc. Convert to trainset and testset. 
    trainset, testset = _loadData()

    return trainset, testset


def _partition_data(
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    trainset, testset = _download_data()
#     trainset = Subset(trainset, list(range(0, len(trainset)//1, 1))) #Daniel new line, less data selected
#     testset = Subset(testset, list(range(0, len(testset)//1, 1))) #Daniel new line, less data selected
    
    # print("Hej från _partition_data i dataset.py")

    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * num_clients
    if iid:
        if balance:
            print("IID and balanced: in total 1728, 864 from Boston and 864 from Singapore. Distributed randomly between clients.")
            balanced_trainset = Subset(trainset, range(319,len(trainset)-1))
            partition_size = int(len(balanced_trainset) / num_clients)
            lengths = [partition_size] * num_clients
            print(f"Length of train set = {len(balanced_trainset)}")
            datasets = random_split(balanced_trainset, lengths, torch.Generator().manual_seed(seed))
            
        else: 
            print("IID and not balanced: in total 2048, 1183 from Boston and 865 from Singapore. Distributed randomly between clients. ")
            datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    
    else:
        if balance:
            print("non-IID and balanced: in total 1728, 864 from Boston and 864 from Singapore. Distributed city-wise between clients.")
            balanced_trainset = Subset(trainset, range(319,len(trainset)-1))
            partition_size = int(len(balanced_trainset) / num_clients)
            lengths = [partition_size] * num_clients
            print(f"Length of train set = {len(balanced_trainset)}")
            
            indices = list(range(len(balanced_trainset)))
            start = 0
            datasets = []
            for i in range(num_clients):
                end = start + partition_size
                subset_indices = indices[start:end]
                subset = Subset(balanced_trainset, subset_indices)
                print(f"Lenght of subset = {len(subset)}")
                datasets.append(subset)
                start = end
            print(f"Lenght of datasets = {len(datasets)}")
            
        else:
            print("non-IID and not balanced: in total 2048, 1183 from Boston and 865 from Singapore. Distributed city-wise between clients.")
            print("But in practice: in total 2016, 1152 from Boston and 864 from Singapore. Distributed city-wise between clients.")
            boston_trainset = Subset(trainset, range(0,1152))
            singapore_trainset = Subset(trainset, range(2048-864,len(trainset)))
            print(f"Length of Boston_trainset = {len(boston_trainset)}")
            print(f" Length ofSingapore_trainset = {len(singapore_trainset)}")
            
            half_of_clients = int(num_clients/2)
                                     
            partition_size_boston = int(len(boston_trainset)/half_of_clients)
            lengths_boston = [partition_size_boston] * half_of_clients
            datasets_boston = random_split(boston_trainset, lengths_boston, torch.Generator().manual_seed(seed))
                                     
            partition_size_singapore = int(len(singapore_trainset)/half_of_clients)
            lengths_singapore = [partition_size_singapore] * half_of_clients
            datasets_singapore = random_split(singapore_trainset, lengths_singapore, torch.Generator().manual_seed(seed))
                                
            # datasets = datasets_boston + datasets_singapore
            datasets = []
            for subset in datasets_boston:
                print(f"Lenght of Boston subset = {len(subset)}")
                datasets.append(subset)
            for subset in datasets_singapore:
                print(f"Lenght of Singapore subset = {len(subset)}")
                datasets.append(subset)
        
    return datasets, testset



def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in class_counts:
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


######################## FOR NUSCENES #######################################

#################################################################################################################################
# Define your custom dataset class that inherits from torch.utils.data.Dataset
class NuscenesDataset(Dataset):
    def __init__(self, image_data, agent_state_data, ground_truth_data):
        self.image_data = image_data
        self.agent_state_data = agent_state_data
        self.ground_truth_data = ground_truth_data
        
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, index):
        image_data_item = self.image_data[index]
        agent_state_data_item = self.agent_state_data[index]
        ground_truth_data_item = self.ground_truth_data[index]
        
        return image_data_item, agent_state_data_item, ground_truth_data_item

def _loadData():
    ################################################################################################################################################
    # Load data
    version = "v1.0-trainval" # v1.0-mini, v1.0-trainval
    seconds_of_history_used = 2.0 # 2.0
    sequences_per_instance = "one_sequences_per_instance" # one_sequences_per_instance, all_sequences_per_instance
#       Non-sorted
#     train_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_img_tensor_list.pt")
#     train_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_agent_state_vector_list.pt")
#     train_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_future_xy_local_list.pt")

#     val_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_img_tensor_list.pt")
#     val_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_agent_state_vector_list.pt")
#     val_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_future_xy_local_list.pt")

#      Sorted
    train_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B=1321_S=865_train_img_tensor_list.pt")
    train_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B=1321_S=865_train_agent_state_vector_list.pt")
    train_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B=1321_S=865_train_future_xy_local_list.pt")

    val_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B=349_S=254_val_img_tensor_list.pt")
    val_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B=349_S=254_val_agent_state_vector_list.pt")
    val_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B=349_S=254_val_future_xy_local_list.pt")

    scale_factor = 1/2
    # Squeeze for correct dimensions
    for i, train_img_tensor in enumerate(train_img_tensor_list):
        dummy = torch.nn.functional.interpolate(train_img_tensor, scale_factor=scale_factor, mode='bilinear')
        train_img_tensor_list[i] = torch.squeeze(dummy, dim=0)
        train_agent_state_vector_list[i] = torch.squeeze(train_agent_state_vector_list[i], dim=0)
        
    for j, val_img_tensor in enumerate(val_img_tensor_list):
        dummy = torch.nn.functional.interpolate(val_img_tensor, scale_factor=scale_factor, mode='bilinear')
        val_img_tensor_list[j] = torch.squeeze(dummy, dim=0)
        val_agent_state_vector_list[j] = torch.squeeze(val_agent_state_vector_list[j], dim=0)

        
    ################################################################################################################################################

    # For testing
    train_short_size = 2048
    short_train_img_tensor_list = train_img_tensor_list[-train_short_size:]
    short_train_agent_state_vector_list = train_agent_state_vector_list[-train_short_size:]
    short_train_future_xy_local_list = train_future_xy_local_list[-train_short_size:]
    val_short_size = 512
    short_val_img_tensor_list = val_img_tensor_list[-val_short_size:]
    short_val_agent_state_vector_list = val_agent_state_vector_list[-val_short_size:]
    short_val_future_xy_local_list = val_future_xy_local_list[-val_short_size:]


    # Prints
    train_num_datapoints = len(train_img_tensor_list)
    print(f"train_num_datapoints whole dataset = {train_num_datapoints}")
    short_train_num_datapoints = len(short_train_img_tensor_list)
    print(f"train_num_datapoints short = {short_train_num_datapoints}")
    # print(f"train_img_tensor_list[0] = {train_img_tensor_list[0].size()}")
    # print(f"train_agent_state_vector_list[0] = {train_agent_state_vector_list[0].size()}")
    # print(f"train_future_xy_local_list[0] = {train_future_xy_local_list[0].size()}\n")
    val_num_datapoints = len(val_img_tensor_list)
    print(f"val_num_datapoints whole dataset = {val_num_datapoints}")
    short_val_num_datapoints = len(short_val_img_tensor_list)
    print(f"val_num_datapoints short = {short_val_num_datapoints}")
    # print(f"val_img_tensor_list[0] = {val_img_tensor_list[0].size()}")
    # print(f"val_agent_state_vector_list[0] = {val_agent_state_vector_list[0].size()}")
    # print(f"val_future_xy_local_list[0] = {val_future_xy_local_list[0].size()}\n")


    # Variables
#     batch_size = 10
#     shuffle = True # Set to True if you want to shuffle the data in the dataloader
    # num_modes = 64 # 2206, 415, 64 (match with eps_traj_set)
    # eps_traj_set = 8 # 2, 4, 8 (match with num_modes)
    # learning_rate = 1e-4 # From Covernet paper: fixed learning rate of 1e−4
    # num_epochs = 4998
    

    # Define datasets
    train_dataset = NuscenesDataset(train_img_tensor_list, train_agent_state_vector_list, train_future_xy_local_list)
    train_shortDataset = NuscenesDataset(short_train_img_tensor_list, short_train_agent_state_vector_list, short_train_future_xy_local_list)
    val_dataset = NuscenesDataset(val_img_tensor_list, val_agent_state_vector_list, val_future_xy_local_list)
    val_shortDataset = NuscenesDataset(short_val_img_tensor_list, short_val_agent_state_vector_list, short_val_future_xy_local_list)

#     # Instantiate dataloaders
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     train_shortDataloader = DataLoader(train_shortDataset, batch_size=batch_size, shuffle=shuffle)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
#     val_shortDataloader = DataLoader(val_shortDataset, batch_size=batch_size, shuffle=shuffle)

    return train_shortDataset, val_shortDataset