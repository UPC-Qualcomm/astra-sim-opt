import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st

# Custom Modules
import helper.bulk_analysis_helper as helper


def render(df):
    st.subheader("Dimension reduction")

    col0, col1, col2 = st.columns(3)
    with col0:
        algo = ["PCA", "UMAP"]
        dim_reduction_method = st.selectbox("Select a dimension reduction method", algo)
    with col1:
        split_data = st.checkbox("Split data based on FSDP value", value=False)
    with col2:
        if split_data:
            is_fsdp = st.checkbox("FSDP Enabled?", value=True)
        else:
            is_fsdp = False

    if split_data:
        cols = ["dp", "tp", "sp", "pp"]
        df_map = df[df["fsdp"] == is_fsdp]
        df_map = df_map[["dp", "tp", "sp", "pp", "total"]]
    else:
        cols = ["dp", "tp", "sp", "pp", "fsdp"]
        df_map = df[cols]
        df_map["total"] = df["total"]

    if dim_reduction_method == "UMAP":
        min_dist, n_neighbors, metric = helper.select_umap_components(
            df_len=len(df_map)
        )
    else:
        min_dist, n_neighbors, metric = 0, 0, 0

    col0, col1 = st.columns(2)
    with col0:
        if dim_reduction_method == "UMAP":
            st.pyplot(helper.get_1d_umap_fig(df_map, min_dist, n_neighbors, metric))
        elif dim_reduction_method == "PCA":
            st.pyplot(helper.get_1d_pca_fig(df_map))
        else:
            st.warning("The selected dimension reduction method is not supported!!!")

    with col1:
        if dim_reduction_method == "UMAP":
            st.plotly_chart(
                helper.get_2d_umap_fig(df_map, cols, min_dist, n_neighbors, metric),
                use_container_width=True,
            )
        elif dim_reduction_method == "PCA":
            st.plotly_chart(
                helper.get_2d_pca_fig(df_map, cols), use_container_width=True
            )
        else:
            st.warning("The selected dimension reduction method is not supported!!!")

    st.markdown("---")

    st.subheader("Cluster the data points based on dimension reduction method.")

    cluster_algo = ["KMeans", "DBSCAN"]
    clustering_method = st.selectbox("Select a clustering method method", cluster_algo)

    if clustering_method == "KMeans":
        labels_1d, labels_2d = helper.cluster_kmeans(df_map, n_clusters=15)
    elif clustering_method == "DBSCAN":
        labels_1d, labels_2d = helper.cluster_dbscan(
            df_map, eps=0.35, min_samples=6, metric="euclidean"
        )
    else:
        labels_1d, labels_2d = [], []

    df_map["cluster_2d"] = labels_2d
    df_map["cluster_1d"] = labels_1d

    col0, col1 = st.columns(2)
    with col0:
        st.pyplot(helper.get_1d_clustering_fig(df_map, labels_1d))

    with col1:
        st.plotly_chart(helper.get_2d_clustering_fig(df_map), use_container_width=True)

    st.write("## Average Total per Cluster - sorted - best are (lowest_average)")
    best_1d_clusters = helper.get_best_1d_clusters(df_map)
    best_2d_clusters = helper.get_best_2d_clusters(df_map)

    col0, col1 = st.columns(2)
    with col0:
        st.write(best_1d_clusters)

    with col1:
        st.write(best_2d_clusters)

    st.markdown("---")

    st.title("Fit the data to an ML model")

    col0, col1 = st.columns(2)
    with col0:
        split_data = st.checkbox("Split dataset according FSDP value", value=False)
    with col1:
        if split_data:
            is_fsdp = st.checkbox("Enable FSDP ", value=True)
        else:
            is_fsdp = False

    if split_data:
        df_ml = df[df["fsdp"] == is_fsdp]
    else:
        df_ml = df

    st.write("Data Preview:")
    st.dataframe(df_ml)

    features = ["dp", "tp", "sp", "pp"]
    target = "total"

    X = df_ml[features]
    y = df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=42
    )

    best_rf, preds_rf, metric_rf = helper.ml_random_forest(
        df_ml, X_train, X_test, y_train, y_test
    )
    best_xgb, preds_xgb, metric_xgb = helper.ml_xgboost(
        df_ml, X_train, X_test, y_train, y_test
    )
    best_mlp, preds_mlp, metric_mlp = helper.ml_mlp(
        df_ml, X_train, X_test, y_train, y_test
    )

    merged_df = pd.concat([metric_rf, metric_xgb, metric_mlp], ignore_index=True)

    st.subheader("📈 ML Model Performance")
    st.dataframe(merged_df)

    st.subheader("🔍 Feature Importance (RF & XGB)")
    col0, col1 = st.columns(2)
    with col0:
        st.write("Random Forest:")
        fig, perm_rf_df = helper.plot_feature_importance(
            best_rf, features, "Feature Importances (Random Forest)"
        )
        st.dataframe(perm_rf_df)
        st.pyplot(fig)

    with col1:
        st.write("XGBoost:")
        fig, perm_xgb_df = helper.plot_feature_importance(
            best_xgb, features, "Feature Importances (XGBoost)"
        )
        st.dataframe(perm_xgb_df)
        st.pyplot(fig)

    st.subheader("🔀 Permutation Feature Importance")
    col0, col1 = st.columns(2)
    with col0:
        st.write("Random Forest:")
        fig, perm_rf_df = helper.plot_permutation_importance(
            best_rf, X_test, y_test, features, "Permutation Importance (Random Forest)"
        )
        st.dataframe(perm_rf_df)
        st.pyplot(fig)

    with col1:
        st.write("XGBoost:")
        fig, perm_xgb_df = helper.plot_permutation_importance(
            best_xgb, X_test, y_test, features, "Permutation Importance (XGBoost)"
        )
        st.dataframe(perm_xgb_df)
        st.pyplot(fig)

    possible_degree_val = df["dp"].unique()

    st.subheader("📊 Partial Dependence Plots")
    col0, col1 = st.columns(2)
    with col0:
        st.write("Random Forest:")
        rf_pdp_fig = helper.compute_and_plot_pdp(
            best_rf,
            X_test,
            features,
            "RF: Normalized Partial Dependence on Total",
            possible_degree_val,
        )
        st.pyplot(rf_pdp_fig)

    with col1:
        st.write("XGBoost:")
        xgb_pdp_fig = helper.compute_and_plot_pdp(
            best_xgb,
            X_test,
            features,
            "XGB: Normalized Partial Dependence on Total",
            possible_degree_val,
        )
        st.pyplot(xgb_pdp_fig)
