import jax
import numpy as np
import torch

from fbx_convertor.smplx_joints import SMPLX_JOINTS
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_raw_multiply,
)


def swap_joints_rotation(joints_rotation, joint_part=["body", "hand"]):
    # joints_rotation: (joint_num, 3)
    left_joints_idx = []
    right_joints_idx = []
    if "body" in joint_part:
        left_joints_idx += SMPLX_JOINTS.left_body_idx
        right_joints_idx += SMPLX_JOINTS.right_body_idx
    if "hand" in joint_part:
        left_joints_idx += SMPLX_JOINTS.left_hand_idx
        right_joints_idx += SMPLX_JOINTS.right_hand_idx
    left_joints_rot = joints_rotation[left_joints_idx]
    right_joints_rot = joints_rotation[right_joints_idx]
    joints_rotation[left_joints_idx] = right_joints_rot
    joints_rotation[right_joints_idx] = left_joints_rot

    return joints_rotation


class HumanMotion:
    smplx_pose_dims = dict(
        cam_trans=3,
        smplx_root_pose=3,
        smplx_body_pose=len(SMPLX_JOINTS.orig_body_joint_part) * 3,
        smplx_lhand_pose=len(SMPLX_JOINTS.orig_lhand_joint_part) * 3,
        smplx_rhand_pose=len(SMPLX_JOINTS.orig_rhand_joint_part) * 3,
        smplx_jaw_pose=len(SMPLX_JOINTS.orig_jaw_joint_part) * 3,
        smplx_shape=10,
        smplx_expr=10,
    )
    rotation_names = [
        "smplx_root_pose",
        "smplx_body_pose",
        "smplx_lhand_pose",
        "smplx_rhand_pose",
        "smplx_jaw_pose",
    ]

    def __init__(
        self,
        smplx_poses: dict,
    ):
        smplx_poses = dict(sorted(smplx_poses.items(), key=lambda x: x[0]))
        self.axisangle2euler_func = lambda x, rotation_order: matrix_to_euler_angles(
            axis_angle_to_matrix(torch.tensor(x)), rotation_order
        ).numpy()
        # fill none
        assert smplx_poses is not None
        assert smplx_poses["cam_trans"] is not None
        _, frame_structure = jax.tree_util.tree_flatten(smplx_poses["cam_trans"])
        for name in smplx_poses.keys():
            smplx_poses[name] = self._fill_none(
                smplx_poses[name], self.smplx_pose_dims[name], frame_structure
            )

        self.process_smplx_poses(smplx_poses)
        print("Finish processing smplx poses")

        rot_matrix = lambda euler: matrix_to_quaternion(
            euler_angles_to_matrix(torch.tensor(euler), "XYZ")
        )[None, ...]
        self.rotation_euler = lambda euler_angel, x: quaternion_to_axis_angle(
            quaternion_raw_multiply(
                rot_matrix(euler_angel), axis_angle_to_quaternion(torch.from_numpy(x))
            )
        ).numpy()

    @staticmethod
    def _fill_none(item, dim, frame_structure):
        if item is None:
            frame = np.zeros(1, dim)
            item = jax.tree_util.tree_unflatten(frame_structure, frame)
        return item

    def rotation_coordinate_transformation(self, target_coordinate_type="blender"):
        # target_coordinate_type: blender, unity3d
        if target_coordinate_type == "blender":
            pass
        elif target_coordinate_type == "unity3d":
            hand_flip = np.ones_like(self.smplx_rotation[0][..., 0])
            hand_flip[SMPLX_JOINTS.hand_idx] = -1
            hand_flip_xyz = [hand_flip, 1, hand_flip]

            self.smplx_rotation = jax.tree_util.tree_map(
                lambda x: np.stack(
                    [
                        hand_flip_xyz[0] * x[..., 0],
                        hand_flip_xyz[1] * x[..., 1],
                        hand_flip_xyz[2] * x[..., 2],
                    ],
                    axis=-1,
                ),
                self.smplx_rotation,
            )
            self.smplx_rotation = jax.tree_util.tree_map(
                lambda x: np.concatenate(
                    [self.rotation_euler([-torch.pi, 0, 0], x[0:1]), x[1:]], axis=0
                ),
                self.smplx_rotation,
            )
            self.smplx_rotation = jax.tree_util.tree_map(
                lambda x: self.axisangle2euler_func(x, "ZXY")[..., [1, 2, 0]],
                self.smplx_rotation,
            )
        elif "/" in target_coordinate_type:
            symbol = target_coordinate_type.split("/")
            symbol = [-1 if "n" in s else 1 for s in symbol]

            hand_flip = np.ones_like(self.smplx_rotation[0][..., 0])
            hand_flip[SMPLX_JOINTS.hand_idx] = -1
            hand_flip_xyz = [hand_flip if s == -1 else 1 for s in symbol]

            self.smplx_rotation = jax.tree_util.tree_map(
                lambda x: np.stack(
                    [
                        hand_flip_xyz[0] * x[..., 0],
                        hand_flip_xyz[1] * x[..., 1],
                        hand_flip_xyz[2] * x[..., 2],
                    ],
                    axis=-1,
                ),
                self.smplx_rotation,
            )
            self.smplx_rotation = jax.tree_util.tree_map(
                lambda x: np.concatenate(
                    [self.rotation_euler([-torch.pi, 0, 0], x[0:1]), x[1:]], axis=0
                ),
                self.smplx_rotation,
            )
            self.smplx_rotation = jax.tree_util.tree_map(
                lambda x: self.axisangle2euler_func(x, "ZXY")[..., [1, 2, 0]],
                self.smplx_rotation,
            )
        else:
            raise NotImplementedError("Unknown coordinate type")

        self.smplx_rotation = jax.tree_util.tree_map(
            lambda x: x * 180 / np.pi,
            self.smplx_rotation,
        )
        print("Finish rotation coordinate transformation")

    def process_smplx_poses(self, smplx_poses):
        self.smplx_trans = jax.tree_util.tree_map(
            lambda x: x[0], smplx_poses["cam_trans"]
        )
        self.smplx_rotation = jax.tree_util.tree_map(
            lambda *x: np.concatenate(list(x), axis=1)[0].reshape(
                SMPLX_JOINTS.orig_joint_num, 3
            ),
            *([smplx_poses[name] for name in self.rotation_names]),
            is_leaf=lambda x: isinstance(x, np.ndarray),
        )
        self.smplx_betas = jax.tree_util.tree_map(
            lambda x: x[0] * 100, smplx_poses["smplx_shape"]
        )
        self.smplx_expr = jax.tree_util.tree_map(
            lambda x: x[0] * 100, smplx_poses["smplx_expr"]
        )
