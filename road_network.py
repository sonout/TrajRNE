import warnings
from dataclasses import dataclass, field
from typing import List

warnings.simplefilter(action="ignore", category=UserWarning)

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import swifter
from shapely import wkt
from shapely.geometry import LineString, box
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch_geometric.transforms as T
    from torch_geometric.data import Data
except ImportError:
    ...


class RoadNetwork:
    """
    Class representing a Road Network.
    """

    G: nx.MultiDiGraph
    gdf_nodes: gpd.GeoDataFrame
    gdf_edges: gpd.GeoDataFrame

    def __init__(
        self,
        location: str = None,
        network_type: str = "roads",
        retain_all: bool = True,
        truncate_by_edge: bool = True,
    ):
        """
        Create network from edge and node file or from a osm query string.

        Args:
            location (str): _description_
            network_type (str, optional): _description_. Defaults to "roads".
            retain_all (bool, optional): _description_. Defaults to True.
            truncate_by_edge (bool, optional): _description_. Defaults to True.
        """
        if location != None:
            self.G = ox.graph_from_place(
                location,
                network_type=network_type,
                retain_all=retain_all,
                truncate_by_edge=truncate_by_edge,
            )
            self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.G)

    def save(self, path: str):
        """
        Save road network as node and edge shape file.
        Args:
            path (str): file saving path
        """
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.G)
        gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
        gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
        gdf_edges["fid"] = np.arange(
            0, gdf_edges.shape[0], dtype="int"
        )  # id for each edge

        gdf_nodes.to_file(path + "/nodes.shp", encoding="utf-8")
        gdf_edges.to_file(path + "/edges.shp", encoding="utf-8")

    def load(self, path):
        """
        Load graph from edges and nodes shape file
        """
        self.gdf_nodes = gpd.read_file(path + "/nodes.shp")
        self.gdf_edges = gpd.read_file(path + "/edges.shp")
        self.gdf_nodes.set_index("osmid", inplace=True)
        self.gdf_edges.set_index(["u", "v", "key"], inplace=True)

        # encode highway column
        self.gdf_edges["highway"] = self.gdf_edges["highway"].str.extract(r"(\w+)")
        le = LabelEncoder()
        self.gdf_edges["highway_enc"] = le.fit_transform(self.gdf_edges["highway"])

        self.G = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)

    def load_edges(self, path):
        self.gdf_edges = gpd.read_file(path + "/edges.shp")
        self.gdf_edges.set_index(["u", "v", "key"], inplace=True)

        self.gdf_edges.reset_index(inplace=True)
        self.G = nx.from_pandas_edgelist(
            self.gdf_edges, "u", "v", True, nx.MultiDiGraph, "key"
        )
        self.gdf_edges.set_index(["u", "v", "key"], inplace=True)


    @property
    def bounds(self):
        return self.gdf_nodes.geometry.total_bounds

    @property
    def bounds_edges(self):
        return self.gdf_edges.geometry.total_bounds

    @property
    def line_graph(self):
        return nx.line_graph(self.G, create_using=nx.DiGraph)

    def generate_road_segment_pyg_dataset(
        self,
        traj_data: gpd.GeoDataFrame = None,
        drop_labels: List = [],
        include_coords: bool = False,
        one_hot_enc: bool = True,
        return_df: bool = False,
        only_edge_index: bool = False,
        dataset: str = "porto",
    ):
        """
        Generates road segment feature dataset in the pyg Data format.
        if traj_data given it will also generate trajectory based features
        like avg. speed and avg. utilization on each road segment
        """
        # create edge_index for line
        LG = self.line_graph
        # create edge_index
        map_id = {j: i for i, j in enumerate(LG.nodes)}
        edge_list = nx.to_pandas_edgelist(LG)
        edge_list["sidx"] = edge_list["source"].map(map_id)
        edge_list["tidx"] = edge_list["target"].map(map_id)

        edge_index = np.array(edge_list[["sidx", "tidx"]].values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        if only_edge_index:
            return Data(edge_index=edge_index)

        # create feature matrix
        df = self.gdf_edges.copy()
        df["idx"] = df.index.map(map_id)
        df.sort_values(by="idx", axis=0, inplace=True)

        df.rename(columns={"fid": "id"}, inplace=True)
        if traj_data is not None:
            # incorperate trajectorie data in form of speed and volume
            traj_data.drop(["id"], axis=1, inplace=True)
            df = df.join(traj_data)

        if include_coords:
            df["x"] = df.geometry.centroid.x / 100  # normalize to -2/2
            df["y"] = df.geometry.centroid.y / 100  # normalize to -1/1

        highway = df["highway"].reset_index(drop=True)
        drops = [
            "osmid",
            "id",
            "geometry",
            "idx",
            "name",
            "highway",
            "ref",
            "access",
            "width",
        ]
        if dataset == "porto":
            drops.append("area")

        df.drop(
            drops,
            axis=1,
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)
        df = df.replace("False",0).replace(["True","[False, True]","['viaduct', 'yes']"],1)
        df["bridge"] = (
            df["bridge"]
            .fillna(0)
            .replace(
                [
                    "yes",
                    "viaduct",
                    "['yes', 'viaduct']",
                    "cantilever",
                    "['yes', 'movable']",
                    "movable",
                    "['no', 'yes']",
                ],
                1,
            )
        )
        df["tunnel"] = (
            df["tunnel"]
            .fillna(0)
            .replace(
                ["yes", "building_passage", "culvert", "['yes', 'building_passage']"], 1
            )
        )
        if dataset == "sf":
            df["reversed"] = (
                df["reversed"]
                .fillna(0)
                .replace(["True", "[False, True]"], 1)
                .replace(["False"], 0)
            )
        df["junction"] = (
            df["junction"]
            .fillna(0)
            .replace(["roundabout", "circular", "cloverleaf", "jughandle"], 1)
        )
        df["lanes"] = df["lanes"].str.extract(r"(\d+)")
        df["maxspeed"] = df["maxspeed"].str.extract(r"(\d+)")

        # normalize continiuos features
        df["length"] = (df["length"] - df["length"].min()) / (
            df["length"].max() - df["length"].min()
        )  # min max normalization

        imputer = KNNImputer(n_neighbors=1)
        imputed = imputer.fit_transform(df)
        df["lanes"] = imputed[:, 2].astype(int)
        df["maxspeed"] = imputed[:, 3].astype(int)
        if dataset == "sf":
            df["maxspeed"] = df["maxspeed"] * 1.61

        df.drop(drop_labels, axis=1, inplace=True)  # drop label?

        cats = ["lanes", "maxspeed"]
        if "highway_enc" not in drop_labels:
            cats.append("highway_enc")

        # revert changes and build it that without onehot it returns the right label
        if one_hot_enc:
            # Categorical features one hot encoding
            df = pd.get_dummies(
                df,
                columns=cats,
                drop_first=True,
            )
        else:
            df["highway"] = highway
            cats.append("highway")
            labels = {}
            for c in cats:
                code, label = pd.factorize(df[c])
                df[c] = code
                labels[c] = label

        if return_df:
            return df, labels

        features = torch.DoubleTensor(np.array(df.values, dtype=np.double))
        # print(features)
        # create pyg dataset
        data = Data(x=features, edge_index=edge_index)
        transform = T.Compose(
            [
                T.NormalizeFeatures(),
            ]
        )
        data = transform(data)

        return data
