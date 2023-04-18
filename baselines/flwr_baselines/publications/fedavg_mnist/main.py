"""Runs CNN federated learning for MNIST dataset."""

from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from flwr_baselines.publications.fedavg_mnist import client, utils

# Daniel
from flwr.common.typing import Parameters #Daniel for FedOpt
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path #Daniel for FedOpt

# Hannes 
from flwr.common.typing import Parameters # Hannes, for parameters fix Hannes_FedOpt
from hydra.utils import call # Hannes testar 15/3

# Daniel for saving weights
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy.fedavg import FedAvg
# Daniel for saving weights end

# DEVICE: torch.device = torch.device("cuda:0") #Daniel
# DEVICE: torch.device = torch.device("cpu") #Daniel
DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Daniel for GPU


@hydra.main(config_path="docs/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    client_fn, testloader = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        num_clients=cfg.num_clients,
        iid=cfg.iid,
        balance=cfg.balance,
        learning_rate=cfg.learning_rate,
    )

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE)

    # # # (BELOW) ##################################################### -------------- CL (Hannes) -------------- ##################################################################
    # from flwr_baselines.publications.fedavg_mnist import centralized_learning
    # training_loss_cl, training_accuracy_cl, validation_loss_cl, validation_accuracy_cl, test_loss_cl, test_accuracy_cl = centralized_learning.run_CL(cfg)
    # # # (ABOVE) ##################################################### -------------- CL (Hannes) -------------- ##################################################################

    
    # Hannes label
    this_test = cfg.current_test
    testdict = {
        "num_clients": cfg.num_clients,
        "num_rounds": cfg.num_rounds,
        "num_epochs": cfg.num_epochs,
        "iid": cfg.iid,
        "client_fraction": cfg.client_fraction,
        "fed_optimizer": cfg.fed_optimizer,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
    }
    thisLable = this_test + ": " + str(testdict[this_test])

    # File suffix
    file_suffix: str = (
        f"{'_iid' if cfg.iid else ''}"
        f"{'_balanced' if cfg.balance else ''}"
        f"_C={cfg.num_clients}"
        f"_Cf={cfg.client_fraction}" # Hannes 
        f"_Ef={cfg.frac_eval}" #Daniel
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_Opt={cfg.fed_optimizer}" # Hannes 
        f"_Lr={cfg.learning_rate}" # Hannes 
        # f"_eta={cfg.eta}" # Hannes 
        # f"_eta1={cfg.eta1}" # Hannes 
    )


    
    # The last of Hannes versions
    initial_parameters: Parameters = call(cfg.get_initial_parameters) # Hannes 

    # Hannes 
    fed_optimizer = cfg.fed_optimizer
    if fed_optimizer == "FedProx":
        print("Prox running")
        strategy = fl.server.strategy.FedProx(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval, # 1.0, # Hannes (changed from 0.0)
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            proximal_mu = cfg.mu,
        )

    elif fed_optimizer == "saveFedProx":
        print("Prox Save Model running")
     
        class saveFedProx(fl.server.strategy.FedProx):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                if aggregated_parameters is not None:
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                    # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"docs/tmpResults/weights_{file_suffix}.npz", *aggregated_ndarrays)
                   
                return aggregated_parameters, aggregated_metrics

        strategy = saveFedProx(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval, # 1.0, # Hannes (changed from 0.0)
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            proximal_mu = cfg.mu,
            )
        
    elif fed_optimizer == "FedAvgM":
        print("AvgM running")
        strategy = fl.server.strategy.FedAvgM(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
        )

    elif fed_optimizer == "saveFedAvgM":
        print("AvgM Save Model running")
     
        class saveFedAvgM(fl.server.strategy.FedAvgM):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                if aggregated_parameters is not None:
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                    # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"docs/tmpResults/weights_{file_suffix}.npz", *aggregated_ndarrays)
                   
                return aggregated_parameters, aggregated_metrics

        strategy = saveFedAvgM(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            )
        
    elif fed_optimizer == "FedAdam": # Hannes_FedOpt (Whole block) TODO: fix
        print("Adam running")
        strategy = fl.server.strategy.FedAdam(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            accept_failures = True, # Hannes, added to fit fedadagrad
            initial_parameters=initial_parameters, # Hannes  Hannes_FedOpt
            # initial_parameters=None, # Hannes  Hannes_FedOpt
            eta = cfg.eta, #1e-1, # Hannes
            eta_l = cfg.eta1, #1e-1, # Hannes 
            tau = cfg.tau, #1e-9, # Hannes, changed from e-9 to avoid nan
        )

    elif fed_optimizer == "saveFedAdam":
        print("Adam Save Model running")
        class saveFedAdam(fl.server.strategy.FedAdam):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                if aggregated_parameters is not None:
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                    # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"docs/tmpResults/weights_{file_suffix}.npz", *aggregated_ndarrays)
                   
                return aggregated_parameters, aggregated_metrics

        strategy = saveFedAdam(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            accept_failures = True, # Hannes, added to fit fedadagrad
            initial_parameters=initial_parameters, # Hannes  Hannes_FedOpt
            # initial_parameters=None, # Hannes  Hannes_FedOpt
            eta = cfg.eta, #1e-1, # Hannes
            eta_l = cfg.eta1, #1e-1, # Hannes 
            tau = cfg.tau, #1e-9, # Hannes, changed from e-9 to avoid nan
            )
        
    elif fed_optimizer == "FedAdagrad": # Hannes_FedOpt (Whole block) TODO: fix
        print("Adagrad running")
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            accept_failures = True, # Hannes
            initial_parameters=initial_parameters, # Hannes  Hannes_FedOpt
            eta = cfg.eta, #1e-1, # Hannes
            eta_l = cfg.eta1, #1e-1, # Hannes 
            tau = cfg.tau, #1e-9, # Hannes, changed from e-9 to avoid nan
        ) 

    elif fed_optimizer == "saveFedAdagrad":
        print("Adagrad Save Model running")
        class saveFedAdagrad(fl.server.strategy.FedAdagrad):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                if aggregated_parameters is not None:
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                    # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"docs/tmpResults/weights_{file_suffix}.npz", *aggregated_ndarrays)
                   
                return aggregated_parameters, aggregated_metrics

        strategy = saveFedAdagrad(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            accept_failures = True, # Hannes
            initial_parameters=initial_parameters, # Hannes  Hannes_FedOpt
            eta = cfg.eta, #1e-1, # Hannes
            eta_l = cfg.eta1, #1e-1, # Hannes 
            tau = cfg.tau, #1e-9, # Hannes, changed from e-9 to avoid nan
            )
        
    elif fed_optimizer == "FedYogi": # Hannes_FedOpt (Whole block) TODO: fix
        print("Yogi running")
        strategy = fl.server.strategy.FedYogi(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            accept_failures = True, # Hannes
            initial_parameters=initial_parameters, # Hannes  Hannes_FedOpt
            eta = cfg.eta, #1e-1, # Hannes
            eta_l = cfg.eta1, #1e-1, # Hannes 
            tau = cfg.tau, #1e-9, # Hannes, changed from e-9 to avoid nan
        ) 
    elif fed_optimizer == "saveFedYogi":
        print("Yogi Save Model running")
        class saveFedYogi(fl.server.strategy.FedYogi):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                if aggregated_parameters is not None:
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                    # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"docs/tmpResults/weights_{file_suffix}.npz", *aggregated_ndarrays)
                   
                return aggregated_parameters, aggregated_metrics

        strategy = saveFedYogi(
              fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval,
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            accept_failures = True, # Hannes
            initial_parameters=initial_parameters, # Hannes  Hannes_FedOpt
            eta = cfg.eta, #1e-1, # Hannes
            eta_l = cfg.eta1, #1e-1, # Hannes 
            tau = cfg.tau, #1e-9, # Hannes, changed from e-9 to avoid nan
            )
        
    elif fed_optimizer == "FedAvg": # Go with FedAvg
        print("Avg running")
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate = cfg.frac_eval, # 1.0, # Hannes (changed from 0.0)
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            # fit_metrics_aggregation_fn=utils.weighted_average, # Hannes, 2023-03-16 NOT WORKING 
            evaluate_metrics_aggregation_fn=utils.weighted_average,
        )
        
    elif fed_optimizer == "saveFedAvg":
        print("Avg Save Model running")
     
        class saveFedAvg(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

                if aggregated_parameters is not None:
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                    # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"docs/tmpResults/weights_{file_suffix}.npz", *aggregated_ndarrays)
                   
                return aggregated_parameters, aggregated_metrics

        strategy = saveFedAvg(
            fraction_fit=cfg.client_fraction,
            fraction_evaluate=cfg.frac_eval, # Hannes (changed from 0.0)
            min_fit_clients=int(cfg.num_clients * cfg.client_fraction),
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=utils.weighted_average,
            # fit_metrics_aggregation_fn=utils.weighted_average, #Daniel
            )
        
    # Hannes added line below (as done in example)
    strategy.initial_parameters = initial_parameters


    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        # OBS Hannes, tar bort GPU shit så länge
        client_resources={"num_gpus": 1.0}, #Daniel for GPU
        ray_init_args = { #Daniel for GPU
            "include_dashboard": True, # we need this one for tracking
            "num_cpus": 1,
            "num_gpus": 1,
            # "memory": ram_memory,
        }
    )

  

    np.save(
        Path(cfg.save_path) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

 # OG plot
    # utils.plot_metric_from_history(
    #     history,
    #     cfg.save_path,
    #     cfg.expected_maximum,
    #     file_suffix,
    # )


    utils.plot_metric_from_history_NEW(
        history,
        cfg.save_path,
        cfg.expected_maximum,
        file_suffix,
        graphLabel = thisLable, # Hannes 
    )


if __name__ == "__main__":
    main()
