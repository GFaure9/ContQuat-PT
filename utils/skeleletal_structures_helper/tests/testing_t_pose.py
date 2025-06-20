import os
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from utils.skeleletal_structures_helper import (
generate_t_pose,
ORIGINAL_S2SL_SKEL,
)


def plot_3d_skel(
        pts: np.ndarray,
        skel_structure: Tuple[Tuple[int, int, int], ...],
        save_folder: str = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for joints
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", marker='o', s=2)

    # Draw bones
    for (iParent, iChild, _) in skel_structure:
        parent_pt = pts[iParent]
        child_pt = pts[iChild]

        # Plot line segment between parent and child
        ax.plot(
            [parent_pt[0], child_pt[0]],
            [parent_pt[1], child_pt[1]],
            [parent_pt[2], child_pt[2]],
            lw=0.5, color="black",
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 5)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Skeleton')

    if save_folder is not None:
        ax.view_init(elev=90, azim=-90)  # XY plane view
        ax.set_title('3D Skeleton - XY Plane')
        plt.savefig(f"{save_folder}/skel_XYview.png")

        ax.view_init(elev=0, azim=-90)  # XZ plane view
        ax.set_title('3D Skeleton - XZ Plane')
        plt.savefig(f"{save_folder}/skel_XZview.png")

        ax.view_init(elev=140, azim=-90)  # custom view
        ax.set_title('3D Skeleton - XZ Plane')
        plt.savefig(f"{save_folder}/skel_custom_view.png")

        plt.close()
    else:
        ax.view_init(elev=140, azim=-90)  # custom view
        plt.show()


def main():
    plots_dir = "./outputs_t_pose"
    os.makedirs(plots_dir, exist_ok=True)

    # ============== Original S2SL T-Pose
    bones_lengths = np.array([
        # neck
        0.2,
        # left shoulder + upper arm + forearm
        0.3, 0.4, 0.3,
        # right shoulder + upper arm + forearm + wrist
        0.3, 0.4, 0.3,
        # left wrist
        0.01,
        # left hand palm (1, 2, 3, 4, 5)
        0.1, 0.1, 0.1, 0.1, 0.1,
        # left finger 1
        0.03, 0.03, 0.01,
        # left finger 2
        0.07, 0.05, 0.02,
        # left finger 3
        0.08, 0.05, 0.02,
        # left finger 4
        0.06, 0.05, 0.02,
        # left finger 5
        0.05, 0.05, 0.02,
        # right wrist
        0.01,
        # right hand palm (1, 2, 3, 4, 5)
        0.1, 0.1, 0.1, 0.1, 0.1,
        # right finger 1
        0.03, 0.03, 0.01,
        # right finger 2
        0.07, 0.05, 0.02,
        # right finger 3
        0.08, 0.05, 0.02,
        # right finger 4
        0.06, 0.05, 0.02,
        # right finger 5
        0.05, 0.05, 0.02,
    ])

    t_pose = generate_t_pose("original_s2sl_skel", bones_lengths, np.array([0, 4, 0]))
    plot_3d_skel(t_pose[0], ORIGINAL_S2SL_SKEL, "./outputs_t_pose")

if __name__ == "__main__":
    main()