env_metadata = {
    "deep-sea-treasure-v0": {
        "ref_point": [-100.0, -100.0],
        "file_prefix": "dst",
        "actions": {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        },
        "reward_dim": (2,)
    },
    "minecart-v0": {
        "ref_point": [-100.0, -100.0, -100.0],
        "file_prefix": "mc"
    }
}
