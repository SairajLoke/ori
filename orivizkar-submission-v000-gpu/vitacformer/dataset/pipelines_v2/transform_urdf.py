import os
import numpy as np
import copy
import time
import torch

try:
    import trimesh
    from urdfpy import URDF
except:
    print("want to use urdf, should pip install (trimesh, urdfpy), numpy version 1.21.1")

def find_base_link(robot):
    """
    Find the base link in the URDF. The base link is the one that is not a child of any joint.
    """
    all_links = {link.name: link for link in robot.links}
    child_links = {joint.child for joint in robot.joints}

    # The base link is the one that is not a child link
    base_link_name = next(link_name for link_name in all_links if link_name not in child_links)
    return all_links[base_link_name]

def load_mesh(stl_dir, mesh_file, downsample):
    mesh_path = os.path.join(stl_dir, os.path.basename(mesh_file))
    mesh = trimesh.load_mesh(mesh_path)

    if downsample is not None:
        current_vertices = len(mesh.vertices)
        target_vertices = current_vertices // downsample


        indices = np.random.choice(current_vertices, target_vertices, replace=False)


        index_map = {old_index: new_index for new_index, old_index in enumerate(indices)}


        new_faces = []
        for face in mesh.faces:
            if all(vertex in index_map for vertex in face):
                new_faces.append([index_map[vertex] for vertex in face])


        new_faces = np.array(new_faces)


        downsampled_vertices = mesh.vertices[indices]
        downsampled_mesh = trimesh.Trimesh(vertices=downsampled_vertices, faces=new_faces)
        mesh = downsampled_mesh

    return mesh

class LoadUrdfPoints(object):
    def __init__(
        self,
        urdf_path,
        stl_dir,
        ja_cfg,
        output_name="urdf_points",
        data_type="0.1.0",
        downsample=None,
        color=None,
        exclude_links=[],
        env=False,
        debug=False,
    ):

        self.urdf_path = urdf_path
        self.stl_dir = stl_dir
        self.ja_cfg = ja_cfg
        self.output_name = output_name
        self.downsample = downsample
        self.color = color
        self.data_type = data_type

        self.env = env
        self.debug = debug

        # load urdf&base_link
        self.robot = URDF.load(urdf_path)
        self.base_link = find_base_link(self.robot)
        self.exclude_links = exclude_links

        # load mesh
        self.meshes = dict()
        for visual in self.base_link.visuals:
            if visual.geometry.mesh is not None:
                mesh_file = visual.geometry.mesh.filename
                self.meshes[mesh_file] = load_mesh(
                    stl_dir=stl_dir, 
                    mesh_file=mesh_file, 
                    downsample=downsample
                )

    def load_and_merge_meshes(self, link, transform, joint_angles):
        """
        Recursively load and merge meshes of the robot, applying joint rotations based on given joint angles.
        """
        combined_meshes = []
        # Load meshes for the current link
        for idx, visual in enumerate(link.visuals):
            if visual.geometry.mesh is not None:
                if link.name in self.exclude_links:
                    continue
                mesh_file = visual.geometry.mesh.filename
                if mesh_file in self.meshes:
                    mesh = copy.deepcopy(self.meshes[mesh_file])
                else:
                    mesh = load_mesh(
                        stl_dir=self.stl_dir, 
                        mesh_file=mesh_file, 
                        downsample=self.downsample
                    )
                    self.meshes[mesh_file] = copy.deepcopy(mesh)

                # Apply the visual origin transform if present
                if visual.origin is not None:
                    mesh.apply_transform(visual.origin)
                # Apply the cumulative transform
                mesh.apply_transform(transform)
                combined_meshes.append(mesh)
                self.compute_times += 1

        # Recursively process child joints and links
        for joint in self.robot.joints:
            # Check if the parent of this joint is the current link
            parent_link_name = joint.parent
            if parent_link_name == link.name:
                # Get the child link
                child_link = self.robot.link_map[joint.child]

                # Get the joint's origin transform
                joint_transform = joint.origin if joint.origin is not None else np.eye(4)

                # Initialize the joint movement transform
                joint_motion = np.eye(4)

                # Apply joint rotation for revolute joints
                if joint.joint_type == 'revolute' or joint.joint_type == 'continuous':
                    # Get the joint axis
                    axis = joint.axis
                    # Get the joint angle from the provided joint_angles dictionary
                    angle = joint_angles.get(joint.name, 0.0)
                    # Compute the rotation matrix around the joint axis
                    rotation = trimesh.transformations.rotation_matrix(angle, axis)
                    # Update the joint motion transform
                    joint_motion = rotation
                elif joint.joint_type == 'prismatic':
                    # For prismatic joints, apply translation along the axis
                    axis = joint.axis
                    displacement = joint_angles.get(joint.name, 0.0)
                    translation = np.eye(4)
                    translation[:3, 3] = axis * displacement
                    joint_motion = translation
                else:
                    # Fixed joint or other types
                    pass

                # Combine the transforms: parent transform -> joint origin -> joint motion
                new_transform = transform @ joint_transform @ joint_motion

                # Recursively load meshes for the child link
                child_meshes = self.load_and_merge_meshes(child_link, new_transform, joint_angles)
                combined_meshes.extend(child_meshes)

        return combined_meshes

    def __single_frame_load(self, results):
        s_t = time.time()
        self.compute_times = 0
        joint_angle_v = results[self.ja_cfg["name"]]
        offset = self.ja_cfg.get("offset", -1)
        if self.env:
            if self.data_type == "sim1.0.0":
                joint_angle_v = joint_angle_v[offset][0]
            else:
                joint_angle_v = joint_angle_v[offset]   # list[np.array]
        else:
            if isinstance(joint_angle_v, torch.Tensor):
                joint_angle_v = joint_angle_v[offset].detach().cpu().numpy()
            else:
                joint_angle_v = joint_angle_v[offset].copy()

        joint_angles = {}
        using_index = 0
        # using_angles = []
        for joint in self.robot.joints:
            if joint.joint_type in ('revolute', 'prismatic'):
                value_index = self.ja_cfg["index"][using_index]
                joint_angles[joint.name] = joint_angle_v[value_index]   #-np.pi / 2  # -90 degrees in radians
                using_index += 1
                # joint_angles[joint.name] = -np.pi/9
                # using_angles.append(joint.name)
                
        # print(using_angles)

        initial_transform = np.eye(4)
        combined_meshes = self.load_and_merge_meshes(
            link=self.base_link, 
            transform=initial_transform, 
            joint_angles=joint_angles,
        )
        combined_meshes = trimesh.util.concatenate(combined_meshes)

        # transform mesh to numpy
        mesh_points = np.array(combined_meshes.vertices.reshape(-1, 3))
        if self.color is not None:
            color_points = np.array(self.color)[np.newaxis, :]
            color_points = np.repeat(color_points, mesh_points.shape[0], axis=0)
            mesh_points = np.concatenate([mesh_points, color_points], axis=-1)
        
        urdf_points = torch.from_numpy(mesh_points).to(torch.float32)
        if self.env:
            urdf_points = urdf_points.unsqueeze(0)
        # results["urdf_points"] = urdf_points

        e_t = time.time()
        if self.debug:
            import open3d

            print("[LoadUrdfPoints] cost time %.3f"%(e_t-s_t))
            print(f"compute times: {self.compute_times}")
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(mesh_points[..., :3])
            if self.color is not None:
                pcd.colors = open3d.utility.Vector3dVector(mesh_points[..., 3:]/255.0)
            show_list = [
                pcd,
                open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05),
            ]
            open3d.visualization.draw_geometries(show_list)

        return urdf_points


    def __call__(self, results):
        urdf_points = self.__single_frame_load(results)
        results[self.output_name] = urdf_points
        return results
        
if __name__ == '__main__':
    load_urdf = LoadUrdfPoints(
        urdf_path="/mnt/net-cloud4/Team/Robot/simulation_one/assets/robots/rm_luban/rm_luban.urdf",
        stl_dir="/mnt/net-cloud4/Team/Robot/simulation_one/assets/robots/rm_luban/meshes/",
        ja_cfg=None,
        downsample=None,
        env=False,
        debug=True,
    )
    results = dict()
    results = load_urdf.__call__(results)

    for i in range(10):
        results = load_urdf.__call__(results)
        # show_stl_with_open3d(results)
