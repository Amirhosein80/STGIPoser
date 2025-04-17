r"""
SMPL/MANO/SMPLH parametric model. Modified from https://github.com/CalciferZh/SMPL.
"""

import enum
import os
import pickle

import numpy as np

from kinematics import *


def append_value(x: torch.Tensor, value: float, dim=-1):
    r"""
    Append a value to a tensor in a specific dimension. (torch)

    e.g. append_value(torch.zeros(3, 3, 3), 1, dim=1) will result in a tensor of shape [3, 4, 3] where the extra
         part of the original tensor are all 1.

    :param x: Tensor in any shape.
    :param value: The value to be appended to the tensor.
    :param dim: The dimension to be expanded.
    :return: Tensor in the same shape except for the expanded dimension which is 1 larger.
    """
    app = (
        torch.ones_like(x.index_select(dim, torch.tensor([0], device=x.device))) * value
    )
    x = torch.cat((x, app), dim=dim)
    return x


append_zero = partial(append_value, value=0)
append_one = partial(append_value, value=1)


class ParametricModel:
    r"""
    SMPL/MANO/SMPLH parametric model.
    """

    def __init__(
        self,
        official_model_file: str,
        use_pose_blendshape=False,
        device=torch.device("cpu"),
    ):
        r"""
        Init an SMPL/MANO/SMPLH parametric model.

        :param official_model_file: Path to the official model to be loaded.
        :param use_pose_blendshape: Whether to use the pose blendshape.
        :param device: torch.device, cpu or cuda.
        """
        with open(official_model_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        self._J_regressor = (
            torch.from_numpy(data["J_regressor"].toarray()).float().to(device)
        )
        self._skinning_weights = torch.from_numpy(data["weights"]).float().to(device)
        self._posedirs = torch.from_numpy(data["posedirs"]).float().to(device)
        self._shapedirs = (
            torch.from_numpy(np.array(data["shapedirs"])).float().to(device)
        )
        self._v_template = torch.from_numpy(data["v_template"]).float().to(device)
        self._J = torch.from_numpy(data["J"]).float().to(device)
        self.face = data["f"]
        self.parent = data["kintree_table"][0].tolist()
        self.parent[0] = None
        self.use_pose_blendshape = use_pose_blendshape
        self.bone_connetcion = torch.from_numpy(data["kintree_table"].astype(np.int32))
        self.adj_matrix = torch.zeros(24, 24)
        for idx, (i, j) in enumerate(self.bone_connetcion.T[1:]):
            self.adj_matrix[i, j] = 1.0
            self.adj_matrix[j, i] = 1.0
        self.adj_matrix = self.adj_matrix.bool().to(device)
        self.device = device

    def save_obj_mesh(self, vertex_position, file_name="a.obj"):
        r"""
        Export an obj mesh using the input vertex position.

        :param vertex_position: Vertex position in shape [num_vertex, 3].
        :param file_name: Output obj file name.
        """
        with open(file_name, "w") as fp:
            for v in vertex_position:
                fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for f in self.face + 1:
                fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))

    @staticmethod
    def save_unity_motion(
        pose: torch.Tensor = None,
        tran: torch.Tensor = None,
        output_dir="saved_motions/",
    ):
        r"""
        Write motions into text files used by Unity3D `LoadMotion()`.

        :param pose: Pose tensor in shape [num_frames, *] that can reshape to [num_frame, num_joint, 3, 3]
                    (rotation matrices).
        :param tran: Translation tensor in shape [num_frames, 3] for root positions.
        :param output_dir: Output directory config.py.
        """
        os.makedirs(output_dir, exist_ok=True)

        if pose is not None:
            f = open(os.path.join(output_dir, "pose.txt"), "w")
            f.write(
                "\n".join(
                    [
                        ",".join(["%.4f" % _ for _ in p])
                        for p in pose.reshape(pose.shape[0], -1)
                    ]
                )
            )
            f.close()

        if tran is not None:
            f = open(os.path.join(output_dir, "tran.txt"), "w")
            f.write(
                "\n".join(
                    [
                        ",".join(["%.5f" % _ for _ in t])
                        for t in tran.view(tran.shape[0], 3)
                    ]
                )
            )
            f.close()

    def get_zero_pose_joint_and_vertex(self, shape: torch.Tensor = None):
        r"""
        Get the joint and vertex positions in zero pose. Root joint is aligned at zero.

        :param shape: Tensor for model shapes that can reshape to [batch_size, 10]. Use None for the mean(zero) shape.
        :return: Joint tensor in shape [batch_size, num_joint, 3] and vertex tensor in shape [batch_size, num_vertex, 3]
                 if shape is not None. Otherwise [num_joint, 3] and [num_vertex, 3] assuming the mean(zero) shape.
        """
        if shape is None:
            j, v = self._J - self._J[:1], self._v_template - self._J[:1]
        else:
            shape = shape.view(-1, 10)
            v = (
                torch.tensordot(shape, self._shapedirs, dims=([1], [2]))
                + self._v_template
            )
            j = torch.matmul(self._J_regressor, v)
            j, v = j - j[:, :1], v - j[:, :1]
        return j, v

    def bone_vector_to_joint_position(self, bone_vec: torch.Tensor):
        r"""
        Calculate joint positions in the base frame from bone vectors (position difference of child and parent joint)
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param bone_vec: Bone vector tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Joint position, in shape [batch_size, num_joint, 3].
        """
        return bone_vector_to_joint_position(bone_vec, self.parent)

    def joint_position_to_bone_vector(self, joint_pos: torch.Tensor):
        r"""
        Calculate bone vectors (position difference of child and parent joint) in the base frame from joint positions
        in the base frame. (torch, batch)

        Notes
        -----
        bone_vec[:, i] is the vector from parent[i] to i.

        Args
        -----
        :param joint_pos: Joint position tensor in shape [batch_size, *] that can reshape to [batch_size, num_joint, 3].
        :return: Bone vector, in shape [batch_size, num_joint, 3].
        """
        return joint_position_to_bone_vector(joint_pos, self.parent)

    def forward_kinematics_R(self, R_local: torch.Tensor):
        r"""
        :math:`R_global = FK(R_local)`

        Forward kinematics that computes the global rotation of each joint from local rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint global rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return forward_kinematics_R(R_local, self.parent)

    def inverse_kinematics_R(self, R_global: torch.Tensor):
        r"""
        :math:`R_local = IK(R_global)`

        Inverse kinematics that computes the local rotation of each joint from global rotations. (torch, batch)

        Notes
        -----
        A joint's *local* rotation is expressed in its parent's frame.

        A joint's *global* rotation is expressed in the base (root's parent) frame.

        Args
        -----
        :param R_global: Joint global rotation tensor in shape [batch_size, *] that can reshape to
                         [batch_size, num_joint, 3, 3] (rotation matrices).
        :return: Joint local rotation, in shape [batch_size, num_joint, 3, 3].
        """
        return inverse_kinematics_R(R_global, self.parent)

    def forward_kinematics_T(self, T_local: torch.Tensor):
        r"""
        :math:`T_global = FK(T_local)`

        Forward kinematics that computes the global homogeneous transformation of each joint from
        local homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_local: Joint local transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint global transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return forward_kinematics_T(T_local, self.parent)

    def inverse_kinematics_T(self, T_global: torch.Tensor):
        r"""
        :math:`T_local = IK(T_global)`

        Inverse kinematics that computes the local homogeneous transformation of each joint from
        global homogeneous transformations. (torch, batch)

        Notes
        -----
        A joint's *local* transformation is expressed in its parent's frame.

        A joint's *global* transformation is expressed in the base (root's parent) frame.

        Args
        -----
        :param T_global: Joint global transformation tensor in shape [batch_size, *] that can reshape to
                        [batch_size, num_joint, 4, 4] (homogeneous transformation matrices).
        :return: Joint local transformation matrix, in shape [batch_size, num_joint, 4, 4].
        """
        return inverse_kinematics_T(T_global, self.parent)

    def forward_kinematics(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor = None,
        tran: torch.Tensor = None,
        calc_mesh=False,
    ):
        r"""
        Forward kinematics that computes the global joint rotation, joint position, and additionally
        mesh vertex position from poses, shapes, and translations. (torch, batch)

        :param pose: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                     [batch_size, num_joint, 3, 3] (rotation matrices).
        :param shape: Tensor for model shapes that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran: Root position tensor in shape [batch_size, 3]. Use None for the zero positions.
        :param calc_mesh: Whether to calculate mesh vertex positions.
        :return: Joint global rotation in [batch_size, num_joint, 3, 3],
                 joint position in [batch_size, num_joint, 3],
                 and additionally mesh vertex position in [batch_size, num_vertex, 3] if calc_mesh is True.
        """

        def add_tran(x):
            return x if tran is None else x + tran.view(-1, 1, 3)

        pose = pose.view(pose.shape[0], -1, 3, 3)
        j, v = [
            _.expand(pose.shape[0], -1, -1)
            for _ in self.get_zero_pose_joint_and_vertex(shape)
        ]
        T_local = transformation_matrix(pose, self.joint_position_to_bone_vector(j))
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = decode_transformation_matrix(T_global)
        if calc_mesh is False:
            return pose_global, add_tran(joint_global), None, joint_global

        T_global[..., -1:] -= torch.matmul(
            T_global, append_zero(j, dim=-1).unsqueeze(-1)
        )
        T_vertex = torch.tensordot(
            T_global, self._skinning_weights, dims=([1], [1])
        ).permute(0, 3, 1, 2)
        if self.use_pose_blendshape:
            r = (pose[:, 1:] - torch.eye(3, device=pose.device)).flatten(1)
            v = v + torch.tensordot(r, self._posedirs, dims=([1], [2]))
        vertex_global = torch.matmul(
            T_vertex, append_one(v, dim=-1).unsqueeze(-1)
        ).squeeze(-1)[..., :3]
        return (
            pose_global,
            add_tran(joint_global),
            add_tran(vertex_global),
            joint_global,
        )

    def view_mesh(
        self, vertex_list: list, joints_list: list, distance_between_subjects=0.8
    ):
        r"""
        View model mesh (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param vertex_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_vertex, 3].
        :param vertex_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joints, 3].
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        v_list, f_list = [], []
        f = self.face.copy()
        for i in range(len(vertex_list)):
            v = vertex_list[i].clone().view(-1, self._v_template.shape[0], 3)
            v[:, :, 0] += distance_between_subjects * i
            joints_list[i][:, :, 0] += distance_between_subjects * i
            v_list.append(v)
            f_list.append(f.copy())
            f += v.shape[1]

        verts = torch.cat(v_list, dim=1).cpu().numpy()
        joints = torch.cat(joints_list, dim=1).cpu().numpy()
        faces = np.concatenate(f_list)
        return verts, faces, joints

    def view_motion(
        self, pose_list: list, tran_list: list = None, distance_between_subjects=0.8
    ):
        r"""
        View model motion (poses and translations) (single frame or a sequence).

        Notes
        -----
        If num_frame == 1, only show one picture.

        Args
        -----
        :param pose_list: List in length [num_subject] of tensors that can all reshape to [num_frame, num_joint, 3, 3].
        :param tran_list: List in length [num_subject] of tensors that can all reshape to [num_frame, 3].
        :param fps: Sequence FPS.
        :param distance_between_subjects: Distance in meters between subjects. 0.2 for hand and 0.8 for body is good.
        """
        verts = []
        joints = []
        for i in range(len(pose_list)):
            pose = pose_list[i].view(-1, len(self.parent), 3, 3)
            tran = (
                tran_list[i].view(-1, 3) - tran_list[i].view(-1, 3)[:1]
                if tran_list
                else None
            )
            _, _, vert, joint = self.forward_kinematics(pose, tran=tran, calc_mesh=True)
            verts.append(vert)
            joints.append(joint)
        return self.view_mesh(
            vertex_list=verts,
            joints_list=joints,
            distance_between_subjects=distance_between_subjects,
        )


class SMPLJoint(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """

    ROOT = 0
    PELVIS = 0
    SPINE = 0
    LHIP = 1
    RHIP = 2
    SPINE1 = 3
    LKNEE = 4
    RKNEE = 5
    SPINE2 = 6
    LANKLE = 7
    RANKLE = 8
    SPINE3 = 9
    LFOOT = 10
    RFOOT = 11
    NECK = 12
    LCLAVICLE = 13
    RCLAVICLE = 14
    HEAD = 15
    LSHOULDER = 16
    RSHOULDER = 17
    LELBOW = 18
    RELBOW = 19
    LWRIST = 20
    RWRIST = 21
    LHAND = 22
    RHAND = 23
