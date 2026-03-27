env_metadata = {
    "deep-sea-treasure-v0": {
        "ref_point": [-100.0, -100.0],
        "file_prefix": "dst",
        "action_mapping": {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        },
        "actions": {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right',
        },
        'feature_names': ['y', 'x'],
        "reward_dim": (2,),
        "observations_high": [11, 11],
        "observations_low": [0, 0],
    },
    "minecart-v0": {
        "ref_point": [-100.0, -100.0, -100.0],
        "file_prefix": "mc",
        "actions": {
            0: "Mine",
            1: "Left",
            2: "Right",
            3: "Accelerate",
            4: "Brake",
            5: "None"
        },
        "feature_names": ["x", "y", "speed", "sin", "cos", "ore1load", "ore2load"],
        "reward_dim": (3,),
        "observations_high": [1., 1., 1., 1., 1., 1., 1.],
        "observations_low": [-1., -1., -1., -1., -1., -1., -1.],
    }
}
colors = [
    [
        'red', 'gold', 'peru', 'orangered', 'tomato',
        'coral', 'salmon', 'crimson', 'firebrick', 'darkorange'
    ],
    [
        'forestgreen', 'cyan', 'darkviolet', 'teal', 'deepskyblue',
        'dodgerblue', 'royalblue', 'slateblue', 'mediumseagreen', 'turquoise'
    ]
]
