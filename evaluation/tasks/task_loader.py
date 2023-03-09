import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.neural_network import MLPRegressor

from . import (MeanSpeedRegTask, RoadTypeClfTask)


# index is correct
def init_roadclf(args, network, seed):
    decoder = linear_model.LogisticRegression(
        multi_class="multinomial", max_iter=1000, n_jobs=-1
    )
    y = np.array(
        [network.gdf_edges.loc[n]["highway_enc"] for n in network.line_graph.nodes]
    )
    roadclf = RoadTypeClfTask(decoder, y, seed=seed)
    roadclf.register_metric(
        name="f1_micro", metric_func=metrics.f1_score, args={"average": "micro"}
    )
    roadclf.register_metric(
        name="f1_macro", metric_func=metrics.f1_score, args={"average": "macro"}
    )
    roadclf.register_metric(
        name="f1_weighted",
        metric_func=metrics.f1_score,
        args={"average": "weighted"},
    )
    roadclf.register_metric(
        name="accuracy",
        metric_func=metrics.accuracy_score,
        args={"normalize": True},
    )
    roadclf.register_metric(
        name="AUC",
        metric_func=metrics.roc_auc_score,
        args={"multi_class": "ovo"},
        proba=True,
    )
    roadclf.register_metric(
        name="ConfMx",
        metric_func=metrics.confusion_matrix,
        args={},
    )

    return roadclf


# label index is right here;
def init_meanspeed(args, network, seed):
    city = args["city"]
    tf = pd.read_csv(f"../datasets/trajectories/{city}/speed_features_unnormalized.csv")
    tf.set_index(["u", "v", "key"], inplace=True)
    map_id = {j: i for i, j in enumerate(network.line_graph.nodes)}
    tf["idx"] = tf.index.map(map_id)
    tf.sort_values(by="idx", axis=0, inplace=True)
    # decoder = linear_model.LinearRegression(fit_intercept=True)
    decoder = MLPRegressor(hidden_layer_sizes=(1024,), random_state=seed, max_iter=30)
    y = tf["avg_speed"]
    y.fillna(0, inplace=True)
    y = y.round(2)
    mean_speed_reg = MeanSpeedRegTask(decoder, y, seed=seed)

    mean_speed_reg.register_metric(
        name="MSE", metric_func=metrics.mean_squared_error, args={}
    )
    mean_speed_reg.register_metric(
        name="MAE", metric_func=metrics.mean_absolute_error, args={}
    )
    mean_speed_reg.register_metric(
        name="RMSE", metric_func=metrics.mean_squared_error, args={"squared": False}
    )

    return mean_speed_reg
