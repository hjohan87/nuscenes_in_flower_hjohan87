"""Contains utility functions for CNN FL on MNIST."""


from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.common import Metrics
from flwr.common.typing import NDArrays, Scalar
from flwr.server.history import History
from torch.utils.data import DataLoader

# Hannes
from flwr.common.parameter import ndarrays_to_parameters # Hannes, for parameters fix Hannes_FedOpt
from flwr.common.typing import NDArrays, Scalar, Parameters # Hannes la till parameters, for parameters fix Hannes_FedOpt
# from tempfile import TemporaryFile # Hannes la till parameters, for parameters fix Hannes_FedOpt

from flwr_baselines.publications.fedavg_mnist import model

def plot_metric_from_history_NEW(
    hist: History,
    save_plot_path: Path,
    expected_maximum: float,
    suffix: Optional[str] = "",
    graphLabel: Optional[str] = "neverReached", # Hannes
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    expected_maximum : float
        The expected maximum accuracy from the original paper.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """

    #Hannes
    # Centralized
    centralized_accuracy_dict = hist.metrics_centralized
    rounds_centralized, centralized_accuracy = zip(*centralized_accuracy_dict["accuracy"])
    # centralized_accuracy_and_rounds = hist.metrics_centralized["accuracy"]
    values_and_rounds_loss = hist.losses_centralized
    # centralized_accuracy = [acc[1] for acc in centralized_accuracy_and_rounds]
    centralized_loss = [loss[1] for loss in values_and_rounds_loss]
    # rounds_centralized = [round[0] for round in values_and_rounds_loss]
    
    # Distributed
    distributed_accuracy_dict = hist.metrics_distributed
    rounds_distributed, distributed_accuracy = zip(*distributed_accuracy_dict["accuracy"])
    # distributed_accuracy_and_rounds = hist.metrics_distributed["accuracy"]
    values_and_rounds_loss = hist.losses_distributed
    # distributed_accuracy = [acc[1] for acc in distributed_accuracy_and_rounds]
    distributed_loss = [loss[1] for loss in values_and_rounds_loss]
    # rounds_distributed = [round[0] for round in values_and_rounds_loss]
    
    thislabel = graphLabel
    # y_loss = plt.ylim([0, 1.4])
    # y_acc = plt.ylim([0, 1])

    fig, axs = plt.subplots(2, 2, sharey='row', sharex='col')
    axs[0, 0].plot(rounds_distributed, distributed_loss, label=thislabel)
    axs[0, 0].legend()
    axs[0, 0].grid(axis='y', color='0.95')
    axs[0, 0].set_title('Distributed loss')
    axs[1, 0].plot(rounds_distributed, distributed_accuracy, label=thislabel)
    axs[1, 0].legend()
    axs[1, 0].grid(axis='y', color='0.95')
    axs[1, 0].set_title('Distributed accuracy')
    axs[0, 1].plot(rounds_centralized[1:], centralized_loss[1:], label=thislabel) # To skip 0 value (before training)
    # axs[0, 1].plot(rounds_centralized, centralized_loss, label=thislabel)
    axs[0, 1].legend()
    axs[0, 1].grid(axis='y', color='0.95')
    axs[0, 1].set_title('Centralized loss')
    axs[1, 1].plot(rounds_centralized[1:], centralized_accuracy[1:], label=thislabel)
    # axs[1, 1].plot(rounds_centralized, centralized_accuracy, label=thislabel) # To skip 0 value (before training)
    axs[1, 1].legend()
    axs[1, 1].grid(axis='y', color='0.95')
    axs[1, 1].set_title('Centralized accuracy')

    for ax in axs.flat:
        ax.set(xlabel='Rounds (epochs)')#, ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # fig = plt.figure()
    # axis = fig.add_subplot(111)
    # plt.plot(np.asarray(rounds), np.asarray(values), label=graphLabel)
    # # plt.plot(np.asarray(rounds), np.asarray(values_accuracy), label=graphLabel)
    # # plt.plot(np.asarray(rounds), np.asarray(values_loss), label=graphLabel)

    # # Set expected graph for data
    # plt.axhline(
    #     y=expected_maximum,
    #     color="r",
    #     linestyle="--",
    #     label=f"Paper's best result @{expected_maximum}",
    # )
    # # Set paper's results
    # plt.axhline(
    #     y=0.99,
    #     color="silver",
    #     label="Paper's baseline @0.9900",
    # )
    # #plt.ylim([0.97, 1])
    # plt.ylim([0, 1])
    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    # plt.xlabel("Rounds")
    # plt.ylabel("Accuracy")
    # plt.legend(loc="lower right")

    # # Set the apect ratio to 1.0
    # xleft, xright = axis.get_xlim()
    # ybottom, ytop = axis.get_ylim()
    # axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

    #plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png")) # Hannes, we do not use metric_type anymore
    plt.savefig(Path(save_plot_path) / Path(f"all_plots_metrics{suffix}.png"))
    plt.close()

def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    expected_maximum: float,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    expected_maximum : float
        The expected maximum accuracy from the original paper.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )

    rounds, values = zip(*metric_dict["accuracy"])
    fig = plt.figure()
    axis = fig.add_subplot(111)
    plt.plot(np.asarray(rounds), np.asarray(values), label="FedAvg")
    # Set expected graph for data
    plt.axhline(
        y=expected_maximum,
        color="r",
        linestyle="--",
        label=f"Paper's best result @{expected_maximum}",
    )
    # Set paper's results
    plt.axhline(
        y=0.99,
        color="silver",
        label="Paper's baseline @0.9900",
    )
    plt.ylim([0.97, 1])
    plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # Set the apect ratio to 1.0
    xleft, xright = axis.get_xlim()
    ybottom, ytop = axis.get_ylim()
    axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # # Hannes printlines
    # print(f"accuracies: {accuracies}")
    # print(f"examples: {examples}")
    # print(f"accuracy: {int(sum(accuracies)) / int(sum(examples))}")

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


def gen_evaluate_fn(
    testloader: DataLoader, device: torch.device
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] # Hannes 
    #[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, float]] # Hannes
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]: # Hannes 
    #) -> Optional[Tuple[float, float]]: # Hannes
        # pylint: disable=unused-argument
        """Use the entire MNIST test set for evaluation."""
        # determine device
        net = model.Net()
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

#         loss, accuracy = model.test(net, testloader, device=device)
        loss, accuracy = model.testNuscenes(net, testloader, device=device) # Daniel to flower

        # return statistics
        return loss, {"accuracy": accuracy} # Hannes 
        #return loss, accuracy # Hannes

    return evaluate

# Hannes_FedOpt
def get_initial_parameters() -> Parameters:
    """Returns initial parameters from a model.
    Args:
        num_classes (int, optional): Defines if using CIFAR10 or 100. Defaults to 10.
    Returns:
        Parameters: Parameters to be sent back to the server.
    """
    thisModel = model.Net()
    weights = [val.cpu().numpy() for _, val in thisModel.state_dict().items()]
    # weights = thisModel.parameters()
    parameters = ndarrays_to_parameters(weights)

    # initialParameters = TemporaryFile()
    # np.save(initialParameters, parameters)

    return parameters