import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import spectral_clustering
import time

# 1. 设置全局绘图字体
plt.rc('font', family='serif', serif=['SimSun'])
plt.rc('axes', unicode_minus=False)


# 2. 通用数据加载函数
def load_data(dataset_type):
    if dataset_type == "credit_card":
        # 信用卡数据
        df = pd.read_csv('../data/CC GENERAL.csv')
        df = df.drop(['CUST_ID'], axis=1)
        df = df.dropna()
        X = pd.DataFrame(StandardScaler().fit_transform(df))
        X = np.asarray(X)
        pca = PCA(n_components=2, random_state=24)
        X = pca.fit_transform(X)
        return X, "信用卡数据"
    elif dataset_type == "mall_customers":
        dataset = pd.read_csv('../data/Mall_Customers.csv')
        X = np.array(dataset.iloc[:, [3, 4]])
        return X, "消费者数据"
    elif dataset_type == "stock":
        SP500 = np.genfromtxt('../data/SP500array.csv', delimiter=',').T
        X = (SP500 - np.mean(SP500, axis=1).reshape(-1, 1)) / np.std(SP500, axis=1).reshape(-1, 1)
        X = manifold.TSNE(n_components=2, random_state=0).fit_transform(X)
        return X, "股票数据"
    elif dataset_type == "2d_data":
        X = np.loadtxt("../data/data-8-2-1000.txt")
        return X, "二维数据"



# 4. 各聚类算法实现
## 4.1 层次聚类
def hierarchical_clustering(X, n_clusters=3):
    # 树状图模型
    dendro_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    dendro_model.fit(X)
    # 最终聚类
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return dendro_model, model, labels


## 4.2 K-Means聚类（含肘部法则）
def find_optimal_k(X, max_clusters=10):
    wcss = []
    for i in range(1, min(max_clusters + 1, len(X))):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss


def kmeans_clustering(X):
    wcss = find_optimal_k(X)
    optimal_k = 5 #为了简化 这里默认为5  实际可以根据肘部法则绘图观察得出
    model = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, random_state=0)
    labels = model.fit_predict(X)
    return model, labels, optimal_k


## 4.3 DBSCAN聚类
def dbscan_clustering(X):
    model = DBSCAN(eps=2, min_samples=4)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return model, labels, n_clusters, n_noise


## 4.4 谱聚类（基于DB指数选最优簇数）
def spectral_clustering_opt(X):
    X_pca = PCA(n_components=2, random_state=24).fit_transform(X)
    best_db = float('inf')
    best_k = 2
    best_labels = None
    for k in range(2, 11):
        affinity = pairwise_kernels(X_pca, metric='rbf')
        labels = spectral_clustering(affinity=affinity, n_clusters=k)
        db_score = davies_bouldin_score(X_pca, labels)
        if db_score < best_db:
            best_db = db_score
            best_k = k
            best_labels = labels
    return X_pca, best_labels, best_k


# 5. 统一指标计算函数
def calculate_metrics_unified(X, labels):
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels != -1]
    sil = ch = db = "N/A"
    if len(valid_labels) >= 2:
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
    return sil, ch, db


# 6. 可视化函数
## 6.1 单算法结果可视化（子图）
def plot_single_result(X, labels, title, ax, is_kmeans=False, centers=None):
    if is_kmeans and centers is not None:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = np.zeros_like(xx)
        for i in range(len(xx)):
            Z[i] = [np.argmin(np.linalg.norm(centers - np.array([xx[i, j], yy[i, j]]), axis=1)) for j in
                    range(len(yy[i]))]
        ax.contourf(xx, yy, Z, alpha=0.1, cmap='Set2')
        ax.scatter(centers[:, 0], centers[:, 1], s=100, c='tab:red', label='质心')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set2", ax=ax, s=50)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)


## 6.2 性能对比可视化（全局）
def plot_performance_comparison(all_results):
    # 转换结果为DataFrame
    df_results = pd.DataFrame(all_results)
    metrics = ['silhouette', 'ch_score', 'db_score', 'execution_time']
    metric_names = ['轮廓系数', 'CH指数', 'DB指数', '执行时间(秒)']

    # 创建2x2对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        # 过滤有效数据（排除N/A）
        df_valid = df_results[df_results[metric] != "N/A"].copy()
        if df_valid.empty:
            continue
        df_valid[metric] = df_valid[metric].astype(float)

        # 绘制分组柱状图
        sns.barplot(x='dataset', y=metric, hue='algorithm', data=df_valid, ax=axes[i], palette="Set3")
        axes[i].set_title(f'各算法{name}对比', fontsize=14)
        axes[i].set_xlabel('数据集', fontsize=12)
        axes[i].set_ylabel(name, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# 7. 主执行流程
if __name__ == "__main__":
    datasets = ["credit_card", "mall_customers", "stock", "2d_data"]
    algorithms = ["层次聚类", "K-Means", "DBSCAN", "谱聚类"]
    all_global_results = []

    # 创建单算法结果可视化图（4算法 x 4数据集 = 16子图）
    fig_single, axes_single = plt.subplots(4, 4, figsize=(20, 16))

    for algo_idx, algo_name in enumerate(algorithms):
        for data_idx, dataset_type in enumerate(datasets):
            try:
                print(f"\n正在处理 {dataset_type} 数据集 - {algo_name}...")
                # 加载预处理数据
                X_processed, dataset_name = load_data(dataset_type)

                # 执行聚类并计时
                start_time = time.time()
                if algo_name == "层次聚类":
                    dendro_model, model, labels = hierarchical_clustering(X_processed)
                    n_clusters = 3
                    n_noise = 0
                elif algo_name == "K-Means":
                    model, labels, n_clusters = kmeans_clustering(X_processed)
                    n_noise = 0
                elif algo_name == "DBSCAN":
                    model, labels, n_clusters, n_noise = dbscan_clustering(X_processed)
                elif algo_name == "谱聚类":
                    X_vis, labels, n_clusters = spectral_clustering_opt(X_processed)
                    n_noise = 0
                exec_time = time.time() - start_time

                # 计算指标
                sil, ch, db = calculate_metrics_unified(X_processed, labels)

                # 存储结果
                result = {
                    "dataset": dataset_name,
                    "algorithm": algo_name,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "silhouette": sil,
                    "ch_score": ch,
                    "db_score": db,
                    "execution_time": round(exec_time, 4)
                }
                all_global_results.append(result)

                # 绘制单算法结果子图
                ax = axes_single[algo_idx, data_idx]
                is_kmeans = (algo_name == "K-Means")
                centers = model.cluster_centers_ if is_kmeans else None
                plot_single_result(X_processed, labels, f"{dataset_name}\n{algo_name}\n簇数:{n_clusters} 噪声:{n_noise}",
                                   ax, is_kmeans=is_kmeans, centers=centers)

                # 打印中间结果
                print(f"  完成：簇数={n_clusters}, 噪声点={n_noise}, 执行时间={exec_time:.4f}秒")
                print(f"  轮廓系数={sil}, CH指数={ch}, DB指数={db}")

            except Exception as e:
                print(f"处理 {dataset_type} - {algo_name} 时出错：{e}")
                continue

    # 显示单算法结果图
    plt.tight_layout()
    plt.show()

    # 生成结果汇总表
    print("\n" + "=" * 130)
    print("所有算法性能评估汇总表")
    print("=" * 130)
    print(
        f"{'数据集':<15} {'算法':<12} {'簇数':<6} {'噪声点':<8} {'轮廓系数':<12} {'CH指数':<15} {'DB指数':<12} {'执行时间(秒)':<12}")
    print("-" * 130)
    for res in all_global_results:
        silhouette_str = f"{res['silhouette']:.4f}" if isinstance(res['silhouette'], (int, float)) else str(
            res['silhouette'])
        ch_score_str = f"{res['ch_score']:.4f}" if isinstance(res['ch_score'], (int, float)) else str(res['ch_score'])
        db_score_str = f"{res['db_score']:.4f}" if isinstance(res['db_score'], (int, float)) else str(res['db_score'])

        print(f"{res['dataset']:<15} {res['algorithm']:<12} {res['n_clusters']:<6} {res['n_noise']:<8} "
              f"{silhouette_str:<12} {ch_score_str:<15} {db_score_str:<12} {res['execution_time']:<12.4f}")

    # 绘制性能对比图
    plot_performance_comparison(all_global_results)