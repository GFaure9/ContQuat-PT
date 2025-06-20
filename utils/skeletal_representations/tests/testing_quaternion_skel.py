import os.path

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import time

from utils.skeletal_representations.quaternion_skel import quaternion_to_cartesian_pose, cartesian_to_quaternion_pose


# Utils functions
def visualize_poses(skel_new, skel_resting, struct, title="Skeleton Poses", other_skel=None, savepath: str = None):
    """Plot resting pose and current new pose together"""
    def plot_skeleton(ax, skeleton, color, label, alpha=1., ls="-"):
        for parent, child, _ in struct:
            p1 = skeleton[parent]  # Parent joint
            p2 = skeleton[child]  # Child joint
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2, alpha=alpha, ls=ls)

        ax.scatter(skeleton[:, 0], skeleton[:, 1], color=color, label=label, zorder=3, alpha=alpha)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_skeleton(ax, skel_resting, 'blue', 'Resting Pose', alpha=0.5)
    plot_skeleton(ax, skel_new, 'red', 'New Pose', alpha=1)
    if other_skel is not None:
        plot_skeleton(ax, other_skel, 'green', 'Reconstructed', alpha=.7, ls="--")

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_title(title)
    plt.grid(True)
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

def rotate_joint_position(joint_position, parent_position, theta):
    theta_rad = np.radians(theta)
    relative_position = np.array(joint_position) - np.array(parent_position)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    rotated_xy = np.dot(rotation_matrix, relative_position[:2])
    new_joint_position = np.array([rotated_xy[0], rotated_xy[1], relative_position[2]]) + parent_position
    return new_joint_position


# Tests
def main(with_plots: bool = True):
    #                Skeletal Structure (resting pose)
    #                *********************************
    #
    #                               (0) [ROOT]
    #                                |
    #                               {0}
    #                                |
    #       (3)---{2}---(2)---{1}---(1)---{3}---(4)---{4}---(5)

    skel_struct = (
        # head
        (0, 1, 0),
        # right arm
        (1, 2, 1),
        (2, 3, 2),
        # left arm
        (1, 4, 3),
        (4, 5, 4)
    )

    skel_resting_pose = np.array([
        # head
        [3., 4., 0.],
        # neck
        [3., 3., 0.],
        # right arm (2 joints)
        [1.5, 3., 0.],
        [1., 3., 0.],
        # left arm (2 joints)
        [4.5, 3., 0.],
        [5., 3., 0.]
    ])

    skel_new_pose_basic = np.array([
        # head
        [3., 4., 0.],
        # neck
        [3., 3., 0.],
        # right arm (2 joints)
        [1.5, 3., 0.],
        [1.5, 2.5, 0.],
        # left arm (2 joints)
        [4.5, 3., 0.],
        [4.5, 2.5, 0.]
    ])

    # Create different new poses by applying random rotations
    list_of_new_poses = [skel_new_pose_basic]

    np.random.seed(42)
    N = 20 if with_plots else 1000
    roots2d = np.random.random((N, 2)) / 2 + 3
    thetas = 360 * np.random.random((N, 4)) - 180

    for root, (theta1, theta2, theta3, theta4) in zip(roots2d, thetas):
        pose = skel_resting_pose.copy()
        pose += (np.array([*root, 0]) - pose[0])

        # right
        pose[2] = rotate_joint_position(pose[2], pose[1], theta1)
        pose[3] = rotate_joint_position(pose[3], pose[1], theta1)

        pose[3] = rotate_joint_position(pose[3], pose[2], theta2)

        # left
        pose[4] = rotate_joint_position(pose[4], pose[1], theta3)
        pose[5] = rotate_joint_position(pose[5], pose[1], theta3)

        pose[5] = rotate_joint_position(pose[5], pose[4], theta4)

        list_of_new_poses += [pose]

    plots_dir = "./outputs_quaternions_tests"
    os.makedirs(plots_dir, exist_ok=True)

    # *************** TESTING ONE BY ONE
    print("\nSEQUENTIAL...")
    print(50 * "=")

    plots_dir_sequential = f"{plots_dir}/sequential"
    os.makedirs(plots_dir_sequential, exist_ok=True)

    start = time.time()

    for k, skel_new_pose in enumerate(list_of_new_poses):
        # print(f"\n\nTEST n°{k}")

        rest_pose = skel_resting_pose.copy()
        rest_pose += (skel_new_pose[0] - skel_resting_pose[0])  # to ensure same root for T-pose and New pose

        # # =============================== No transformation ===============================
        # print("Original cartesian coordinates poses...")
        # visualize_poses(skel_new_pose, skel_resting_pose, skel_struct, f"Original - {k}")

        # ================ Apply transformation and inverse transformation =================
        # print("\nPoses after transformed to quaternion and then back to cartesian...")

        # --------------- Cartesian coordinates ==> Quaternions
        quaternions, bones_lengths = cartesian_to_quaternion_pose(
            skel_new_pose.copy(),
            rest_pose.copy(),
            skel_struct
        )
#         print("-------------------------")
#         print("Quaternions:\n", quaternions)
#         print("Bones lengths:\n", bones_lengths)

        # --------------- Quaternions ==> Cartesian coordinates
        skel_new_pose_reconstructed = quaternion_to_cartesian_pose(
            quaternions,
            np.array([rest_pose]),
            skel_struct,
        )
#         print("-------------------------")
#         print("Reconstructed skeleton:\n", skel_new_pose_reconstructed)

        if with_plots:
            visualize_poses(
                skel_new_pose,
                rest_pose,
                skel_struct,
                f"Reconstructed - {k}",
                skel_new_pose_reconstructed[0],
                f"{plots_dir_sequential}/test{k}.png"
            )

    duration_sequential = time.time() - start

    # *************** TESTING IN PARALLEL
    print("\nPARALLEL...")
    print(50 * "=")

    plots_dir_parallel = f"{plots_dir}/parallel"
    os.makedirs(plots_dir_parallel, exist_ok=True)

    array_of_poses = np.array(list_of_new_poses)
    array_of_rest_poses = np.array([
        skel_resting_pose + (new_pose[0] - skel_resting_pose[0]) for new_pose in list_of_new_poses
    ])

    start = time.time()

    # --------------- Cartesian coordinates ==> Quaternions
    array_of_quaternions, array_of_bones_lengths = cartesian_to_quaternion_pose(
        array_of_poses,
        array_of_rest_poses,
        skel_struct
    )

    # --------------- Quaternions ==> Cartesian coordinates
    array_of_skel_new_poses_reconstructed = quaternion_to_cartesian_pose(
        array_of_quaternions,
        array_of_rest_poses,
        skel_struct,
    )

    if with_plots:
        for i, skel_new_pose in enumerate(list_of_new_poses):
            visualize_poses(
                skel_new_pose,
                array_of_rest_poses[i],
                skel_struct,
                f"Reconstructed - {k}",
                array_of_skel_new_poses_reconstructed[i],
                f"{plots_dir_parallel}/test{i}.png"
            )

    duration_parallel = time.time() - start

    print(f"\n\nCOMPUTATION TIMES:\n- Sequential: {duration_sequential}s\n- Parallel: {duration_parallel}s")

if __name__ == "__main__":
    # main(with_plots=True)
    main(with_plots=False)
