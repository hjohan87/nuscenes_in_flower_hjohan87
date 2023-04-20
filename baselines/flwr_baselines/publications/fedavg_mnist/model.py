# nuScenes dev-kit.
# Code written by Freddy Boulton, Tung Phan 2020.
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f

# from nuscenes.prediction.models.backbone import calculate_backbone_feature_dim
from backbone import calculate_backbone_feature_dim

from torch.utils.data import DataLoader # From FL original model

import pickle # For new train version below

# Number of entries in Agent State Vector
ASV_DIM = 3

from backbone import ResNetBackbone # EXTRA TMP
# class CoverNet(nn.Module): 
class Net(nn.Module): # EXTRA TMP
    """ Implementation of CoverNet https://arxiv.org/pdf/1911.10298.pdf """

    # def __init__(self, backbone: nn.Module, num_modes: int,
    #              n_hidden_layers: List[int] = None,
    #              input_shape: Tuple[int, int, int] = (3, 500, 500)):
    # EXTRA TMP:
    def __init__(self, backbone: nn.Module = ResNetBackbone('resnet50'), num_modes: int = 64,
                 n_hidden_layers: List[int] = None,
                 input_shape: Tuple[int, int, int] = (3, 500, 500)):
        """
        Inits Covernet.
        :param backbone: Backbone model. Typically ResNetBackBone or MobileNetBackbone
        :param num_modes: Number of modes in the lattice
        :param n_hidden_layers: List of dimensions in the fully connected layers after the backbones.
            If None, set to [4096]
        :param input_shape: Shape of image input. Used to determine the dimensionality of the feature
            vector after the CNN backbone.
        """

        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(f"Param n_hidden_layers must be a list. Received {type(n_hidden_layers)}")

        super().__init__()

        if not n_hidden_layers:
            n_hidden_layers = [4096]

        self.backbone = backbone

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        n_hidden_layers = [backbone_feature_dim + ASV_DIM] + n_hidden_layers + [num_modes]

        linear_layers = [nn.Linear(in_dim, out_dim)
                         for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]

        self.head = nn.ModuleList(linear_layers)
        self.relu = nn.ReLU()

    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:
        """
        :param image_tensor: Tensor of images in the batch.
        :param agent_state_vector: Tensor of agent state vectors in the batch
        :return: Logits for the batch.
        """

        backbone_features = self.backbone(image_tensor)

        logits = torch.cat([backbone_features, agent_state_vector], dim=1)

        for linear in self.head:
            logits = self.relu(linear(logits))

        return logits


def mean_pointwise_l2_distance(lattice: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Computes the index of the closest trajectory in the lattice as measured by l1 distance.
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
    return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()


class ConstantLatticeLoss:
    """
    Computes the loss for a constant lattice CoverNet model.
    """

    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                 similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(lattice)
        self.similarity_func = similarity_function

    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):

            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth)
            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_logits.device)
            classification_loss = f.cross_entropy(logit.unsqueeze(0), label)

            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()

# Below from FL original model

def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    """
    # # OLD, from FL
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # net.train()
    # for _ in range(epochs):
    #     net = _training_loop(net, trainloader, device, criterion, optimizer)

    # NEW, from nuScenes
    # Lattice and similarity function
    with open('data/sets/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl', 'rb') as f:
        latticeData = pickle.load(f)
    lattice = np.array(latticeData) # a numpy array of shape [num_modes, n_timesteps, state_dim]
    similarity_function = mean_pointwise_l2_distance  # You can also define your own similarity function
    criterion = ConstantLatticeLoss(lattice, similarity_function)
    lr = 1e-4 # From Covernet paper: fixed learning rate of 1e−4
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # Replace <YOUR_LEARNING_RATE> with your desired learning rate
    net.train()
    for _ in range(epochs):
        net = _training_loop(net, trainloader, device, criterion, optimizer)


def _training_loop(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    # # OLD, from FL
    # for images, labels in trainloader:
    #     images, labels = images.to(device), labels.to(device)
    #     optimizer.zero_grad()
    #     loss = criterion(net(images), labels)
    #     loss.backward()
    #     optimizer.step()
    # return net
    
    # NEW, from nuScenes
    for image_tensor, agent_state_vector, ground_truth_trajectory in trainloader:
        # Get batch data
        # image_tensor, agent_state_vector, ground_truth_trajectory = batch
        image_tensor = image_tensor.to(device)
        agent_state_vector = agent_state_vector.to(device)
        ground_truth_trajectory = ground_truth_trajectory.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = net(image_tensor, agent_state_vector)

        # Compute loss
        loss = criterion(logits, ground_truth_trajectory)

        # Backward pass
        loss.backward()
        optimizer.step()

        # # Print loss for this batch
        # print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}")
    return net


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    # # OLD, for FL
    # criterion = torch.nn.CrossEntropyLoss()
    # correct, total, loss = 0, 0, 0.0
    # net.eval()
    # with torch.no_grad():
    #     for images, labels in testloader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)
    #         loss += criterion(outputs, labels).item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # if len(testloader.dataset) == 0:
    #     raise ValueError("Testloader can't be 0, exiting...")
    # loss /= len(testloader.dataset)
    # accuracy = correct / total

    # # print(f"total: {total}, correct: {correct}") #Daniel
    # return loss, accuracy


    # NEW, for nuScenes
    with open('data/sets/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl', 'rb') as f:
        latticeData = pickle.load(f)
    lattice = np.array(latticeData) # a numpy array of shape [num_modes, n_timesteps, state_dim]
    lattice = torch.Tensor(lattice).to(device)
    similarity_function = mean_pointwise_l2_distance  # You can also define your own similarity function
    # print("Hej från test i model.py")
    criterion = ConstantLatticeLoss(lattice, similarity_function)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for image_tensor, agent_state_vector, ground_truth_trajectory in testloader:
            image_tensor = image_tensor.to(device)
            agent_state_vector = agent_state_vector.to(device)
            ground_truth_trajectory = ground_truth_trajectory.to(device)
            logits = net(image_tensor, agent_state_vector)
            loss += criterion(logits, ground_truth_trajectory)#.item()
#             print("lattice shape:", lattice.shape)
#             print("ground_truth_trajectory shape:", ground_truth_trajectory.shape)
            total += ground_truth_trajectory.size(0)
            _, predicted = torch.max(logits, 1)
            for index, ground_truth in enumerate(ground_truth_trajectory):
                closest_lattice_trajectory = similarity_function(lattice, ground_truth)
                print("predicted[index]:", predicted[index])
                print("closest_lattice_trajectory:", closest_lattice_trajectory)
                correct += (predicted[index] == closest_lattice_trajectory).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
#     print(" len(testloader.dataset)", len(testloader.dataset))
    # Convert the tensor to a float
    loss = loss.item()
    accuracy = correct / total

    # print(f"total: {total}, correct: {correct}") #Daniel
    return loss, accuracy
    