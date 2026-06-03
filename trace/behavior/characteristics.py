from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

from trace.behavior.conditioning import EmpiricalDistribution

def tree_rules(obs, acs, feature_names: list = None):
    if feature_names is None: feature_names = ['x', 'y']
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        ccp_alpha=0.01
    )
    tree.fit(obs, acs)
    return export_text(tree, feature_names=feature_names)


def decisiveness(obs:np.ndarray, acs:np.ndarray, metadata:dict, entropy:bool=False, eps:float=1e-12):
    behavior_model = EmpiricalDistribution(metadata, alpha=0.5).fit(obs, acs)
    if len(behavior_model.counts) == 0: return 0.0

    if entropy:
        max_entropy = np.log(len(metadata['actions']))
        d = []
        for s in behavior_model.counts.keys():
            p = behavior_model.action_probs(s)
            h = -np.sum(p * np.log(p + eps))
            d.append(1.0 - h / max_entropy)
    else:
        variances = [np.std(behavior_model.action_probs(state)) for state in policy.get_visited()]
        d = np.array(variances, dtype=float)


    return d, behavior_model.get_visited()
