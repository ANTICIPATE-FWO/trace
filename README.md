

# TRACE: Trajectory-based Clustering for Explainability
Developed at Vrije Universiteit Brussel (VUB) as part of the FWO-SBO project ANTICIPATE, in collaboration with
University of Antwerp (UA) and Universitair Ziekenhuis Brussel (UZBrussel).

**Dependencies:** Python 3.11<br>
**Contact:** Emily Palaska (emily.palaska@vub.be)<br>
**License:** GNU General Public License v3.0

<p align="center">
  <img src="plots/logos/vub_ai_lab.png" height="60px" />
  <img src="plots/logos/fwo_kleur.png" height="60px" />
  <img src="plots/logos/ua_logo.jpg" height="60px" />
  <img src="plots/logos/logo-uzbrussel.jpg" height="60px" />
</p>


## ⚡ Quickstart
Clone this repository with `git clone https://github.com/emily-palaska/trace`<br>
Install requirements with `pip install -r requirements.txt`<br>


## ⚙️ Example of usage

```python
from yaml import safe_load
import numpy as np

from trace.core import TrajectoryManager
from trace.behavior import EmpiricalDistribution, distance_matrix
from trace.clustering import k_medoids
from trace.visuals import temporal_alignment

# Trajectory loading
filepath = '/data/dst_ground_truth.json'
metadata = safe_load(open('trace/configs/dst-conc.yaml','r')) 
manager = TrajectoryManager(metadata).load(filepath)

# Behavior features
obs, acs, _ = manager.conditioning_features(flatten=True, pad=None)
models = [EmpiricalDistribution(metadata).fit(o, a) for o, a in zip(obs, acs)]
features = distance_matrix(models, metric='agreement')

# Clustering
k = 3
labels = k_medoids(features, k=k)
clusters = [manager.subset(np.array(labels) == l) for l in range(k)]

# Intuitive plots
for c, cluster in enumerate(clusters):
    acs = cluster.sequence('actions', flatten=True)
    fig = temporal_alignment(acs, metadata['actions'], title=f'Cluster {c}')
    fig.show()


```

## 🧠 About
TRACE analyzes pareto-optimal trajectories collected by SOTA MORl policies and ground truth algorithms with the goal of
clustering them based in their behavior. The produced groups of trajectories are presented through interpretable
explanations and graphs. In this way, TRACE manages to summarize large amounts of trajectory data into intuitive
strategy insights for decision makers. It consists of five modules:
- **behavior**: empirical conditioning, behavior modeling and distance features
- **clustering**: k-means, k-medoids, gaussian/dirichlet mixture models, spectral
- **core**: data flow management, mathematical methods
- **policies**: SOTA MORL methods, ground truth algorithms and modified environments. 
- **visuals**: plotting functions mainly to identify patterns from the formed clusters

Note: minimally adapted code files from the public repository [morl-baselines](https://github.com/LucasAlegre/morl-baselines)
can be found in the module `trace.policies`, with modifications concerning trajectory tracking. Credits are due to the
original creators of each MORL algorithm.

An abstract high-level flow chart of the mechanism:
<p align="center"> <img src="plots/sketches/pipeline.png"/> </p>

## 💭 Behavior Modeling

todo: pic of four plots from report


## Agreement and decisiveness

todo: explain and minimal mathematical formulas