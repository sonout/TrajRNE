import argparse
import os
import random
import sys
import uuid
from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch_geometric.transforms as T
from generator import RoadNetwork, Trajectory
from models import (
    TrajRNE,
    GAEModel,
    GATEncoder,
    GCNEncoder,
    StructuralRoadEncoder,
    SpatialFlowConvolution,
)

from evaluation import Evaluation
from tasks.task_loader import *

model_map = {
    "trajrne": (
        TrajRNE,
        {"models": ["sfc", "sre"], "aggregator": "concate"},
        "",
    ),
    "sre": (StructuralRoadEncoder, {"out_dim": 3}, "../models/model_states/sre"),
    "sfc": (SpatialFlowConvolution, {}, "../models/model_states/sfc/"),
    "gaegcn": (
        GAEModel,
        {"encoder": GCNEncoder},
        "../models/model_states/gaegcn/",
    ),
    "gaegat": (
        GAEModel,
        {"encoder": GATEncoder},
        "../models/model_states/gaegat/",
    ),
}


def generate_dataset(args, seed):
    """
    Generates the dataset with optional speed features.

    Args:
        args (dict): args from argparser

    Returns:
        pyg Data: Dataset as pyg Data Object with x and edge_index
    """
    city = args["city"]
    network = RoadNetwork()
    network.load(f"../osm_data/{city}")
    traj_test = pd.read_pickle(
        f"../datasets/trajectories/{city}/traj_train_test_split/test_{seed}.pkl"
    )
    traj_test["seg_seq"] = traj_test["seg_seq"].map(np.array)
    drop_label = [args["drop_label"]] if args["drop_label"] is not None else []

    if "speed" in args and args["speed"] is not None:
        feats = [f for f in args["speed"].split(",")]
        traj_features = pd.read_csv(
            f"../datasets/trajectories/{city}/speed_features_unnormalized.csv"
        )
        traj_features.set_index(["u", "v", "key"], inplace=True)
        traj_features["util"] = (
            traj_features["util"] - traj_features["util"].min()
        ) / (
            traj_features["util"].max() - traj_features["util"].min()
        )  # min max normalization
        traj_features["avg_speed"] = (
            traj_features["avg_speed"] - traj_features["avg_speed"].min()
        ) / (
            traj_features["avg_speed"].max() - traj_features["avg_speed"].min()
        )  # min max normalization
        traj_features.fillna(0, inplace=True)

        return (
            network,
            traj_test,
            network.generate_road_segment_pyg_dataset(
                traj_data=traj_features[["id"] + feats],
                include_coords=True,
                drop_labels=drop_label,
                dataset=city,
            ),
        )
    else:
        print("without speed")
        return (
            network,
            traj_test,
            network.generate_road_segment_pyg_dataset(
                include_coords=True, drop_labels=drop_label, dataset=city
            ),
        )


def get_model_path_for_task(base_path, tasks, model_name, seed, city):
    """This method appends the fitting model state to the path.
    The model state dict must match the dataset and task.
    For example Road Type clf task should exclude road type and therefore needs the state dict trained without road type.
    The model state dict names are normed:
        - model_base.pt for model with all features exluding speed features (except gtn, gtc) or no features (node2vec etc.)
        - model_noroad.pt for model trained with all features excluding the road label (for roadclf task)
        - model_nospeed.pt for tained with all features exluding the mean speed (for mean speed prediction task)

    Args:
        base_path (_type_): _description_
        tasks (_type_): desc
    """

    if "roadclf" in tasks:
        assert len(tasks) == 1
        return (
            os.path.join(base_path, f"model_noroad_{city}.pt")
            if model_name != "rfn"
            else os.path.join(base_path, f"model_noroad_{city}.params")
        )

    return (
        os.path.join(base_path, f"model_base_{city}.pt")
        if model_name != "rfn"
        else os.path.join(base_path, f"model_noroad_{city}.params")
    )


def evaluate_model(args, data, network, trajectory, seed):
    """
    trains the model given by the args argument on the corresponding data

    Args:
        args (dict): Args from arg parser
        data (pyg Data): Data to train on
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch.cuda.set_device(args["device"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose(
        [
            T.ToDevice(device),
        ]
    )
    data = transform(data)
    models = [m for m in args["models"].split(",")]
    tasks = [t for t in args["tasks"].split(",")]
    eva = Evaluation()

    if "roadclf" in tasks:
        eva.register_task("roadclf", init_roadclf(args, network, seed))

    if "meanspeed" in tasks:
        eva.register_task("meanspeed", init_meanspeed(args, network, seed))

    city = args["city"]
    adj = np.loadtxt(
        f"../models/training/sfc_precalc_adj/traj_adj_k_2_{city}.gz"
    )  # change to desired path
    for m in models:
        model, margs, model_path = model_map[m]
        if "trajrne" in m: 
            agg_models = []
            for agg_model_name in margs["models"]:
                agg_model, agg_margs, agg_model_path = model_map[agg_model_name]
                if agg_model_name in ["sfc"]:
                    agg_margs["adj"] = adj
                    agg_margs["network"] = network
                if agg_model_name in ["sre"]:
                    agg_margs["network"] = network
                agg_model = agg_model(data, device=device, **agg_margs)
                agg_model.load_model(
                    path=get_model_path_for_task(
                        agg_model_path,
                        tasks,
                        agg_model_name,
                        seed=seed,
                        city=args["city"],
                    )
                )
                agg_models.append(agg_model)

            margs["models"] = agg_models

        if m in ["gaegcn_no_features", "gaegat_no_features"]:
            data.x = None
            data = T.OneHotDegree(128)(data)

        if m in ["sfc"]:
            margs["network"] = network
            margs["traj_data"] = trajectory
            # margs["adj"] = adj

        if m in ["sre"] and args["drop_label"] == "highway_enc":
            margs["remove_highway_label"] = True


        model = model(data, device=device, **margs)
        model.load_model(
            path=get_model_path_for_task(model_path, tasks, m, seed, city=args["city"])
        )

        targs = {}

        eva.register_model(m, model, targs)

    res = eva.run()

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Evaluation")
    parser.add_argument(
        "-m", "--models", help="Models to evaluate", required=True, type=str
    )
    parser.add_argument(
        "-t", "--tasks", help="Tasks to evaluate on", required=True, type=str
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Epochs to train lstm for (time travel evaluation)",
         required=False,
        type=int,
        default=5,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size for lstm training",
        required=False,
        type=int,
        default=512,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="learning rate for lstm training",
        required=False,
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "-s",
        "--speed",
        help="Include speed features given as comma seperated column names",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Save path evaluation results",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-d", "--device", help="Cuda device number", type=int, required=True
    )
    parser.add_argument(
        "-se",
        "--seeds",
        help="Seed for the random operations like train/test split",
        default="69",
        type=str,
    )
    parser.add_argument(
        "-dl",
        "--drop_label",
        help="remove label from train dataset",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--city",
        help="trajectory dataset to evaluate on",
        type=str,
        default="porto",
    )  # sf, porto 

    args = vars(parser.parse_args())

    seeds = [int(s) for s in args["seeds"].split(",")]
    results = []
    for seed in seeds:
        network, test, data = generate_dataset(
            args, seed
        )  

        res = evaluate_model(args, data, network, test, int(seed))

        if len(results) == 0:
            for name, res in res:
                res["seed"] = seed
                results.append((name, res))
        else:
            for i, ((name, full_res), (_, seed_res)) in enumerate(zip(results, res)):
                seed_res["seed"] = seed
                results[i] = (name, pd.concat([full_res, seed_res], axis=0))

    path = os.path.join(
        args["path"],
        str(datetime.now().strftime("%m-%d-%Y-%H-%M")),
    )
    os.mkdir(path)
    for name, res in results:
        res.to_csv(path + "/" + name + ".csv")
