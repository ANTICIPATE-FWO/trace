import numpy as np
from policy_explainability.utils import generate_cluster_data
from policy_explainability.visuals import sankey, cluster_connections, k_means

data_3d = np.stack([generate_cluster_data(seed=i) for i in range(2)])
sankey(*cluster_connections([k_means(data, plot=False) for data in data_3d]))