import numpy as np
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_means(data, k:int=3, plot:bool=True):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    labels, centers = kmeans.labels_, kmeans.cluster_centers_

    if plot:
        plt.figure(figsize=(8, 6))
        for i in range(k):
            plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i + 1}')

        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centers')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('K-Means Clustering Example')
        plt.legend()
        plt.show()
        plt.close()
    return labels


def cluster_connections(labels):
    k, l = len(np.unique(labels[0])), len(np.unique(labels[1]))
    source, target, value = [], [], []

    for i in range(k):
        for j in range(l):
            count = np.sum((labels[0] == i) & (labels[1] == j))
            if count > 0:
                source.append(i)
                target.append(k + j)
                value.append(count)
    return source, target, value


def sankey(source, target, value):
    k, l = len(np.unique(source)), len(np.unique(target))
    labels = [f'Cluster {i+1} (t0)' for i in range(k)] + [f'Cluster {i+1} (t1)' for i in range(l)]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=20, thickness=20, color="lightblue"),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Cluster Flow Between Two K-Means Clusterings", font_size=12)
    fig.show()

def multi_graph():
    raise NotImplementedError
