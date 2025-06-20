import numpy as np

from utils.skeleletal_structures_helper import (
ORIGINAL_S2SL_SKEL,
ORIGINAL_S2SL_SKEL_INVERTED_HANDS,
)


VALID_NAMES = [
    "original_s2sl_skel",
    "original_s2sl_skel_inverted_hands",
]

def generate_t_pose(skel_name: str, bones_lengths: np.ndarray, root_pt: np.ndarray) -> np.ndarray:
    # =========== Note
    # `bones_lengths` => expected shape=(M, N_bones)
    #   | should be an array of arrays of bones lengths in the order of corresponding skel structure (tree)
    # `root_pt` => expected shape=(M, 3)
    #   | should be an array of root joints 3D coordinates
    # ================

    # Ensuring `skel_name` is valid
    if skel_name not in VALID_NAMES:
        raise ValueError(f"No T-pose generation method has been implemented for `skel_name` '{skel_name}'")

    # Ensuring inputs have consistent dimensions
    assert len(bones_lengths.shape) == len(root_pt.shape)
    if len(bones_lengths.shape) == 1:  # i.e. if shape=(N_bones,)
        bones_lengths = np.expand_dims(bones_lengths, axis=0)  # shape=(1, N_bones)
        root_pt = np.expand_dims(root_pt, axis=0)  # shape=(1, 3)

    # Dimensions
    M, N_bones = bones_lengths.shape

    # Skeletal pose structure and translations to apply
    skel_struct, translations = SKEL_STRUCTURES[skel_name], T_POSES_TRANSLATIONS[skel_name]

    # Safeguards
    assert N_bones == len(skel_struct), "`bones_lengths` arrays and `skel_struct` sizes do not match!"
    assert N_bones == len(translations), "`bones_lengths` arrays and `translations` sizes do not match!"

    # Building T-pose(s) iteratively from the root joint
    pts = np.zeros((M, N_bones + 1, 3))  # N_pts = N_bones + 1
    pts[:, 0] = root_pt.copy()
    for k_bone, (i_parent, i_child, _) in enumerate(skel_struct):
        pts[:, i_child] = pts[:, i_parent] + bones_lengths[:, [k_bone]] * translations[[k_bone]]  # shape=(M, 1, 3)

    return pts  # shape=(M, N_bones, 3)


# ------------------------------- /!\ WARNING /!\ -------------------------------
# `T_POSES_TRANSLATIONS` and `SKEL_STRUCTURES` must of course share the same keys!
# These keys should be the ones of `VALID_NAMES`.
# -------------------------------------------------------------------------------

T_POSES_TRANSLATIONS = {
    "original_s2sl_skel": np.array([
        # ----- neck
        [0, -1, 0],

        # ----- left shoulder + left arm (upper + fore)
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],

        # ----- right shoulder + right arm (upper + fore)
        [-1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],

        # ----- left hand
        # left hand - wrist
        [1, 0, 0],
        # left hand - palm
        [0, 0, 1],
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        [1, 0, 0],
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],
        # left hand - 1st finger
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        # left hand - 2nd finger
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        # left hand - 3rd finger
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        # left hand - 4th finger
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        # left hand - 5th finger
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],

        # ----- right hand
        # right hand - wrist
        [-1, 0, 0],
        # right hand - palm
        [0, 0, 1],
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        [-1, 0, 0],
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],
        # right hand - 1st finger
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        # right hand - 2nd finger
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        # right hand - 3rd finger
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        # right hand - 4th finger
        [-1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],
        # right hand - 5th finger
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],
    ]),
    "original_s2sl_skel_inverted_hands": np.array([
        # ----- neck
        [0, -1, 0],

        # ----- left shoulder + left arm (upper + fore)
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],

        # ----- right shoulder + right arm (upper + fore)
        [-1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],

        # ----- right hand
        # right hand - wrist
        [-1, 0, 0],
        # right hand - palm
        [0, 0, 1],
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        [-1, 0, 0],
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],
        # right hand - 1st finger
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        # right hand - 2nd finger
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [-np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        # right hand - 3rd finger
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, np.sin(np.pi / 8)],
        # right hand - 4th finger
        [-1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],
        # right hand - 5th finger
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],
        [-np.cos(np.pi / 8), 0, -np.sin(np.pi / 8)],

        # ----- left hand
        # left hand - wrist
        [1, 0, 0],
        # left hand - palm
        [0, 0, 1],
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        [1, 0, 0],
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],
        # left hand - 1st finger
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        # left hand - 2nd finger
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        [np.cos(np.pi/4), 0, np.sin(np.pi/4)],
        # left hand - 3rd finger
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, np.sin(np.pi/8)],
        # left hand - 4th finger
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        # left hand - 5th finger
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],
        [np.cos(np.pi/8), 0, -np.sin(np.pi/8)],
    ])
}


SKEL_STRUCTURES = {
    "original_s2sl_skel": ORIGINAL_S2SL_SKEL,
    "original_s2sl_skel_inverted_hands": ORIGINAL_S2SL_SKEL_INVERTED_HANDS,
}
