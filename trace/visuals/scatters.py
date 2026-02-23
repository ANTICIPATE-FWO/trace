import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt


colors = [
    [
        'red', 'peru', 'gold', 'orangered', 'tomato',
        'coral', 'salmon', 'crimson', 'firebrick', 'darkorange'
    ],
    [
        'forestgreen', 'cyan', 'darkviolet', 'teal', 'deepskyblue',
        'dodgerblue', 'royalblue', 'slateblue', 'mediumseagreen', 'turquoise'
    ]
]


def cluster_scatter(data, labels, similarity: bool = False, color_id: int = 0, graph_labels: tuple = None):
    if data.shape[1] > 2:
        from trace.visuals.tsne import tsne_transform
        data = tsne_transform(data, similarity=similarity)

    fig, ax = plt.subplots(figsize=(8, 6))
    c = colors[color_id]
    title, x_label, y_label = graph_labels

    for l in set(labels):
        x, y = data[labels == l, 0], data[labels == l, 1]
        ax.scatter(x, y, label=f"Cluster {l + 1}", color=c[l])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.set(xticks=[], yticks=[])
    return fig


def sankey(source, target, value):
    k, l = len(np.unique(source)), len(np.unique(target))
    labels = [f'Cluster {i+1} (b)' for i in range(k)] + [f'Cluster {i+1} (r)' for i in range(l)]
    node_colors = (colors[0][:k] +  colors[1][:l])

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=20, thickness=20, color=node_colors),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Cluster Flow Between Two K-Means Clusterings", font_size=12)
    return fig



