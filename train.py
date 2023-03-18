import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from models import SpatialFlowConvolution, StructuralRoadEncoder, TrajRNE
from road_network import RoadNetwork
from trajectory import Trajectory





def main(city, device_nr):

    print("Load Data")
    network = RoadNetwork()
    network.load(f"data/network")
    trajectory = Trajectory(f"data/trajectory/road_segment_map_sample.csv", nrows=100000000).generate_TTE_datatset()

    traj_features = pd.read_csv(f"data/trajectory/speed_features_unnormalized.csv")
    traj_features.set_index(["u", "v", "key"], inplace=True)
    traj_features["util"] = (traj_features["util"] - traj_features["util"].min()) / (traj_features["util"].max() - traj_features["util"].min())  # min max normalization
    traj_features["avg_speed"] = (traj_features["avg_speed"] - traj_features["avg_speed"].min()) / (traj_features["avg_speed"].max() - traj_features["avg_speed"].min())  # min max normalization
    traj_features.fillna(0, inplace=True)

    sfc_data = network.generate_road_segment_pyg_dataset(include_coords=True, dataset=city)

    device = torch.device(f'cuda:{device_nr}' if torch.cuda.is_available() else 'cpu')

    ###### SFC Training ######
    adj = np.loadtxt(f"data/traj_adj_k_2_False.gz")
    device = torch.device(f'cuda:{device_nr}' if torch.cuda.is_available() else 'cpu')
    print("Start Traning")
    model_sfc = SpatialFlowConvolution(sfc_data, device, network, adj=adj)
    model_sfc.train(epochs=1000)
    model_sfc.save_model(path="models/model_states/sfc/")
    print("Finished SFC Training")

    ##### SRE Training #######
    file_path = "data/sre_traindata.json"
    # Open Train Data
    with open(file_path, "r") as fp:
            sre_data = np.array(json.load(fp))

    # In the negative selection we sometimes get positive pairs. 
    # Thus remove the negative ones
    #u, c = np.unique(sre_data[:,:2], return_counts=True, axis=0)
    #duplicates = u[c > 1]

    #res = Parallel(n_jobs=30)(delayed(check_for_duplicates)(sre_data[i], duplicates) for i in range(len(sre_data)))
    #res = [i for i in res if i is not None]
    #sre_data = np.vstack(res)

    model_sre = StructuralRoadEncoder(sre_data, device, network, out_dim = 4)
    model_sre.train(epochs=1, batch_size=256)
    model_sre.save_model(path="models/model_states/sre/")
    print("Finished SRE Training")


    ####### TrajRNE #######
    data = [] # just a placeholder
    device = torch.device('cpu')
    trajrne = TrajRNE(data, device, aggregator="concate", models=[model_sfc,model_sre])
    embs = trajrne.load_emb()
    print(embs.shape)
    np.save("data/road_embbeddings.npy", embs)


def check_for_duplicates(row, duplicates):
    # Only check those who are negative
    if(row[2] == 0):
        # Now check if this row is in duplicates
        if(any((duplicates[:]==row[:2]).all(1))):
            return None
        else:
            return row
    else:
        return row


if __name__ == '__main__':
    city = "porto"
    device_nr = "1"
    main(city, device_nr)
    