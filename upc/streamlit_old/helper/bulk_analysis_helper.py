import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import umap

from plotly.subplots import make_subplots
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_val_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance


def select_dim(
    df,
    select_msg="Select dimension for the slider:",
    dims=["dp", "tp", "pp", "sp"],
    defualt_idx=3,
    key_prefix="",
    global_min=-1,
):
    col0, col1 = st.columns([2, 4])
    with col0:
        selected_dim = st.selectbox(
            "Select dimension for the slider:",
            dims,
            index=defualt_idx,
            key=f"{key_prefix}_selector",
        )
    with col1:
        dim_val = st.select_slider(
            f"Select {selected_dim} value:",
            options=sorted(df[selected_dim].unique()),
            value=df.loc[global_min, selected_dim],
            key=f"{key_prefix}_slider",
        )
    return selected_dim, dim_val


def get_dims():
    return ["dp", "tp", "pp", "sp"]


def get_remaining_dims(dims, excluded_dim):
    return [d for d in dims if d != excluded_dim]


def find_local_minima(df, x_dim, y_dim, z_dim, value_col="total"):
    local_minima = []
    discrete_dims = ["dp", "tp", "pp", "sp"]

    for idx, row in df.iterrows():
        x_neighbors = get_neighbors(row[x_dim], x_dim, df, discrete_dims)
        y_neighbors = get_neighbors(row[y_dim], y_dim, df, discrete_dims)

        neighbors_df = df[
            (df[x_dim].isin(x_neighbors)) & (df[y_dim].isin(y_neighbors))
        ].drop(index=idx)

        if neighbors_df.empty:
            continue

        if all(row[value_col] < neighbors_df[value_col]):
            local_minima.append(row)

    return pd.DataFrame(local_minima)


@st.cache_data
def find_local_minima_all_axis(df, x_dim, y_dim, z_dim, t_dim, value_col="total"):
    local_minima = []
    discrete_dims = ["dp", "tp", "pp", "sp"]

    for idx, row in df.iterrows():
        x_neighbors = get_neighbors(row[x_dim], x_dim, df, discrete_dims)
        y_neighbors = get_neighbors(row[y_dim], y_dim, df, discrete_dims)
        z_neighbors = get_neighbors(row[z_dim], z_dim, df, discrete_dims)
        t_neighbors = get_neighbors(row[t_dim], t_dim, df, discrete_dims)

        neighbors_df = df[
            (df[x_dim].isin(x_neighbors))
            & (df[y_dim].isin(y_neighbors))
            & (df[z_dim].isin(z_neighbors))
            & (df[t_dim].isin(t_neighbors))
        ].drop(index=idx)

        if neighbors_df.empty:
            continue

        if all(row[value_col] < neighbors_df[value_col]):
            local_minima.append(row)

    return pd.DataFrame(local_minima)


def get_neighbors(val, dim, df, discrete_dims):
    unique_vals = df[dim].unique()
    if dim in discrete_dims:
        candidates = [val]
        if val / 2 in unique_vals:
            candidates.append(val / 2)
        if val * 2 in unique_vals:
            candidates.append(val * 2)
        return candidates
    else:
        return []


def selected_dims_3d_fig(
    df, local_min_df, x_dim, y_dim, z_dim, color_dim, global_min=-1
):
    discrete_dims = ["dp", "tp", "pp", "sp"]

    colorscale = [[0, "green"], [0.5, "yellow"], [1, "red"]]

    fig = make_subplots(
        specs=[[{"type": "scene"}]],
    )

    if global_min != -1 and global_min in df.index:
        min_point = df.loc[global_min]
        fig.add_trace(
            go.Scatter3d(
                x=[min_point[x_dim]],
                y=[min_point[y_dim]],
                z=[min_point[z_dim]],
                mode="markers+text",
                marker=dict(size=7, color="blue", symbol="x"),
                text=["Min total"],
                textposition="top center",
                name="Global Min",
            )
        )

    if not local_min_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=local_min_df[x_dim],
                y=local_min_df[y_dim],
                z=local_min_df[z_dim],
                mode="markers",
                marker=dict(size=5, color="purple", symbol="diamond"),
                name="Local Minima",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=df[x_dim],
            y=df[y_dim],
            z=df[z_dim],
            mode="markers",
            marker=dict(size=4, color=df[color_dim], colorscale=colorscale),
            name="Data Points",
        )
    )

    neighbor_pairs = []
    for idx, row in df.iterrows():
        for dim in [x_dim, y_dim, z_dim]:
            neighbors = get_neighbors(row[dim], dim, df, discrete_dims)
            for n in neighbors:
                neighbor_df = df[df[dim] == n]
                for _, nbr_row in neighbor_df.iterrows():
                    pair = sorted([idx, nbr_row.name])
                    if pair not in neighbor_pairs:
                        neighbor_pairs.append(pair)

    for pair in neighbor_pairs:
        p1 = df.loc[pair[0]]
        p2 = df.loc[pair[1]]
        fig.add_trace(
            go.Scatter3d(
                x=[p1[x_dim], p2[x_dim]],
                y=[p1[y_dim], p2[y_dim]],
                z=[p1[z_dim], p2[z_dim]],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=800,
        width=1200,
        scene=dict(
            xaxis_title=x_dim.upper(),
            yaxis_title=y_dim.upper(),
            zaxis_title=z_dim.upper(),
        ),
    )

    return fig


@st.cache_data
def get_neighbours_map_fig(df, x_dim, y_dim):
    x_values = sorted(df[x_dim].unique())
    y_values = sorted(df[y_dim].unique())

    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = pd.DataFrame({x_dim: xx.ravel(), y_dim: yy.ravel()})

    existing_points = set(zip(df[x_dim], df[y_dim]))

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in grid_points.iterrows():
        x, y = row[x_dim], row[y_dim]
        if (x, y) in existing_points:
            ax.plot(x, y, "s", color="green", markersize=34)
            ax.text(
                x, y, f"({x},{y})", ha="center", va="center", fontsize=20, color="white"
            )
        else:
            ax.plot(x, y, "s", color="gray", markersize=34, alpha=0.3)

    ax.set_xticks(x_values)
    ax.set_yticks(y_values)
    ax.set_xlabel(x_dim, fontsize=20)
    ax.set_ylabel(y_dim, fontsize=20)
    ax.set_title(f"{x_dim} vs {y_dim} Neighbours Map", fontsize=20)

    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.grid(True, which="both", color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def select_umap_components(df_len):
    li = np.arange(0, 1, 0.01)
    min_dist = st.select_slider("Select min_dist value:", options=li, value=0.1)
    n_neighbors = st.select_slider(
        "Select n_neighbors value:", options=range(0, df_len), value=15
    )
    config_names = [
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "canberra",
        "braycurtis",
        "haversine",
        "mahalanobis",
        "wminkowski",
        "seuclidean",
        "cosine",
        "correlation",
    ]
    metric = st.selectbox("Select a Metric", config_names)

    return min_dist, n_neighbors, metric


def get_1d_umap_fig(df, min_dist, n_neighbors, metric):
    df_cols_val = df[df.columns[:-1]].values

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=1,
        metric=metric,
        random_state=42,
    )
    X_embedded = reducer.fit_transform(df_cols_val)
    df["1d"] = X_embedded.flatten()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("UMAP 1D")
    ax.set_title("Effect of Parallelism Parameters on Total via UMAP")
    sc = ax.scatter(df["1d"], df["total"], c=df["total"], cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Total")
    ax.set_ylabel("Total")
    ax.grid(True)

    return fig


def get_1d_pca_fig(df):
    df_cols_val = df[df.columns[:-1]].values

    pca = PCA(n_components=1)
    X_embedded = pca.fit_transform(df_cols_val)
    df["1d"] = X_embedded.flatten()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("PCA 1D")
    ax.set_title("Effect of Parallelism Parameters on Total via PCA")
    sc = ax.scatter(df["1d"], df["total"], c=df["total"], cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Total")
    ax.set_ylabel("Total")
    ax.grid(True)

    return fig


def get_2d_umap_fig(df, cols, min_dist, n_neighbors, metric):
    df_cols_val = df[cols].values

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=42,
    )
    X_embedded = reducer.fit_transform(df_cols_val)
    df["dim_1"] = X_embedded[:, 0]
    df["dim_2"] = X_embedded[:, 1]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["dim_1"],
                y=df["dim_2"],
                z=df["total"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=df["total"],
                    colorscale="Viridis",
                    colorbar=dict(title="Total"),
                    opacity=0.8,
                ),
            )
        ]
    )

    fig.add_trace(
        go.Mesh3d(
            x=df["dim_1"],
            y=df["dim_2"],
            z=df["total"],
            intensity=df["total"],
            colorscale="Viridis",
            opacity=0.3,
            showscale=True,
            colorbar=dict(title="Total"),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="Total",
        ),
        title="3D Plot: UMAP Components 1 & 2 vs Total",
        height=700,
        width=900,
    )
    return fig


def get_2d_pca_fig(df, cols):
    df_cols_val = df[cols].values

    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(df_cols_val)
    df["dim_1"] = X_embedded[:, 0]
    df["dim_2"] = X_embedded[:, 1]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["dim_1"],
                y=df["dim_2"],
                z=df["total"],
                mode="markers",
                marker=dict(
                    size=5,
                    color=df["total"],
                    colorscale="Viridis",
                    colorbar=dict(title="Total"),
                    opacity=0.8,
                ),
            )
        ]
    )

    fig.add_trace(
        go.Mesh3d(
            x=df["dim_1"],
            y=df["dim_2"],
            z=df["total"],
            intensity=df["total"],
            colorscale="Viridis",
            opacity=0.3,
            showscale=True,
            colorbar=dict(title="Total"),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="Total",
        ),
        title="3D Plot: PCA Components 1 & 2 vs Total",
        height=700,
        width=900,
    )
    return fig


def cluster_kmeans(df, n_clusters=15):
    X_2d = StandardScaler().fit_transform(df[["dim_1", "dim_2"]])
    kmeans_2d = KMeans(n_clusters=n_clusters)
    labels_2d = kmeans_2d.fit_predict(X_2d)

    X_1d = StandardScaler().fit_transform(df[["1d"]])
    kmeans_1d = KMeans(n_clusters=n_clusters)
    labels_1d = kmeans_1d.fit_predict(X_1d)

    return labels_1d, labels_2d


def cluster_dbscan(df, eps=0.35, min_samples=6, metric="euclidean"):
    X_2d = StandardScaler().fit_transform(df[["dim_1", "dim_2"]])
    db_2d = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X_2d)
    labels_2d = db_2d.labels_

    X_1d = StandardScaler().fit_transform(df[["1d"]])
    db_1d = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(X_1d)
    labels_1d = db_1d.labels_

    return labels_1d, labels_2d


def get_1d_clustering_fig(df, labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("viridis", n_colors=len(set(labels)))
    sns.scatterplot(
        x=df["1d"],
        y=df["total"],
        hue=df["cluster_1d"],
        palette=palette,
        legend="full",
        ax=ax,
    )
    ax.set_xlabel("1d")
    ax.set_ylabel("Total Value")
    ax.set_title("Clustering by Total Value")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    return fig


def get_2d_clustering_fig(df):
    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(111, projection="3d")

    unique_labels = df["cluster_2d"].unique()
    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    color_map = {
        label: colors[i % len(colors)] if label != -1 else "#888888"
        for i, label in enumerate(unique_labels)
    }
    traces = []
    for idx, label in enumerate(unique_labels):
        cluster_df = df[df["cluster_2d"] == label]
        traces.append(
            go.Scatter3d(
                x=cluster_df["dim_1"],
                y=cluster_df["dim_2"],
                z=cluster_df["total"],
                mode="markers",
                name=f"Cluster {label}" if label != -1 else "Noise",
                marker=dict(
                    size=6,
                    color=color_map[label],
                    opacity=0.8,
                    line=dict(width=0.5, color="black"),
                ),
                text=[
                    f"Index: {idx}<br>Total: {tot}"
                    for idx, tot in zip(cluster_df.index, cluster_df["total"])
                ],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Total"),
        title="Interactive 3D Clusters",
        margin=dict(l=0, r=0, b=0, t=30),
        height=700,
        width=900,
    )
    return fig


def get_best_1d_clusters(df):
    return df.groupby("cluster_1d")["total"].mean().sort_values()


def get_best_2d_clusters(df):
    return df.groupby("cluster_2d")["total"].mean().sort_values()


def ml_random_forest(df, X_train, X_test, y_train, y_test):
    param_dist = {
        "n_estimators": [100, 300, 500],
        "max_depth": [10, 30, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["log2", "sqrt"],
    }

    rf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=30, cv=5, scoring="r2", n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    preds_rf = search.predict(X_test)

    metric = pd.DataFrame(
        {
            "Model": ["Random Forest"],
            "R2 Score": [r2_score(y_test, preds_rf)],
            "RMSE": [np.sqrt(mean_squared_error(y_test, preds_rf))],
            "NRMSE": [
                np.sqrt(mean_squared_error(y_test, preds_rf))
                / (df["total"].max() - df["total"].min())
            ],
            "MAPE (%)": [np.mean(np.abs((y_test - preds_rf) / y_test)) * 100],
        }
    )

    return best_rf, preds_rf, metric


def ml_xgboost(df, X_train, X_test, y_train, y_test):
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    xgb = XGBRegressor(
        random_state=42,
        n_estimators=1000,
        max_depth=7,
        eta=0.1,
        subsample=0.7,
        colsample_bytree=0.8,
    )
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    cross_val_score(xgb, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)

    xgb.fit(X_train, y_train)

    preds_xgb = xgb.predict(X_test)

    metric = pd.DataFrame(
        {
            "Model": ["XGBoost"],
            "R2 Score": [r2_score(y_test, preds_xgb)],
            "RMSE": [np.sqrt(mean_squared_error(y_test, preds_xgb))],
            "NRMSE": [
                np.sqrt(mean_squared_error(y_test, preds_xgb))
                / (df["total"].max() - df["total"].min())
            ],
            "MAPE (%)": [np.mean(np.abs((y_test - preds_xgb) / y_test)) * 100],
        }
    )

    return xgb, preds_xgb, metric


def ml_mlp(df, X_train, X_test, y_train, y_test):
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", max_iter=1000, random_state=42
    )
    mlp.fit(X_train, y_train)

    preds_mlp = mlp.predict(X_test)

    metric = pd.DataFrame(
        {
            "Model": ["MLP"],
            "R2 Score": [r2_score(y_test, preds_mlp)],
            "RMSE": [np.sqrt(mean_squared_error(y_test, preds_mlp))],
            "NRMSE": [
                np.sqrt(mean_squared_error(y_test, preds_mlp))
                / (df["total"].max() - df["total"].min())
            ],
            "MAPE (%)": [np.mean(np.abs((y_test - preds_mlp) / y_test)) * 100],
        }
    )

    return mlp, preds_mlp, metric


def plot_feature_importance(model, features, title):
    """Plot and display model-based feature importances."""
    importance = pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )
    importance_df = pd.DataFrame(
        {"Feature": importance.index, "Importance": importance.values}
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    ax.set_title(title)

    return fig, importance_df


def plot_permutation_importance(model, X_test, y_test, features, title):
    """Plot and display permutation feature importances."""
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"Feature": features, "Importance": perm.importances_mean})
    perm_df = perm_df.sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x="Importance", y="Feature", data=perm_df, ax=ax)
    ax.set_title(title)

    return fig, perm_df


def compute_and_plot_pdp(model, X_test, features, title, degree_val):
    """Compute and plot partial dependence for a single model"""
    pdp_results = {}
    axes_results = {}

    for feat in features:
        disp = PartialDependenceDisplay.from_estimator(
            model, X_test, [feat], grid_resolution=100
        )
        pdp = disp.lines_[0][0].get_ydata()
        axis = disp.lines_[0][0].get_xdata()
        pdp_results[feat] = pdp
        axes_results[feat] = axis
        plt.close()

    plot_data = pd.DataFrame()
    for feat in features:
        vals = pdp_results[feat]
        norm_vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        temp_df = pd.DataFrame(
            {
                "Feature": feat,
                "Feature Value": axes_results[feat],
                "Normalized Effect on Total": norm_vals,
            }
        )
        plot_data = pd.concat([plot_data, temp_df])

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(
        data=plot_data,
        x="Feature Value",
        y="Normalized Effect on Total",
        hue="Feature",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Normalized Effect on Total")
    ax.set_xlabel("Feature Value")
    ax.legend(title="Feature")
    ax.grid(True)
    ax.set_xticks(degree_val)

    return fig
