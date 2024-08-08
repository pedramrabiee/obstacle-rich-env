import torch

# TODO: Generelize the LiDAR for both 2D and 3D Environment
class ObstacleLidar:
    def __init__(self, map, robot, lidar_cfg):
        self.map = map
        self.robot = robot
        self.lidar_cfg = lidar_cfg
        # TODO: Sensor location and angle needs to read from robot config.
        self.sensor_location = torch.tensor(lidar_cfg.sensor_mounting_xy)
        self.sensor_angle = torch.tensor(lidar_cfg.sensor_mounting_yaw)
        self._mesh_maker()

    def get_lidar_coor(self, x):
        # Repeat global mesh for each batch item
        vectorized_global_mesh = self.global_mesh.repeat(x.shape[0], 1)

        # Transform the mesh from robot to world coordinates
        transformed_vectorized_global_mesh = self._robot_to_world(vectorized_global_mesh, self.robot.get_robot_pos(x),
                                                                  self.robot.get_robot_rot(x))

        # Extract boundary points
        boundary = self._boundary_extractor(transformed_vectorized_global_mesh)

        # Add noise if required
        if self.lidar_cfg.has_noise:
            boundary = self.add_noise(boundary)

        if self.lidar_cfg.return_cartesian:
            return boundary

        return self._cartesian_to_polar(boundary - self.robot.get_robot_pos(x))

    def get_lidar(self, x):
        dist, _ = self.get_lidar_and_coor(x)
        return dist

    def get_lidar_and_coor(self, x):
        coord = self.get_lidar_coor(x)
        if self.lidar_cfg.return_cartesian:
            return torch.norm(coord - self.robot.get_robot_pos(x).unsqueeze(1), 2, dim=-1), coord
        return coord[..., 0], coord


    def _mesh_maker(self):
        range_space = torch.linspace(0, self.lidar_cfg.max_range, self.lidar_cfg.ray_sampling_rate, dtype=torch.float64)
        theta_space = torch.linspace(self.lidar_cfg.scan_angle[0], self.lidar_cfg.scan_angle[-1],
                                     self.lidar_cfg.ray_num, dtype=torch.float64)
        r, theta = torch.meshgrid(range_space, theta_space, indexing='xy')
        self.global_mesh = self._sensor_to_robot(self._polar_to_cartesian(r.flatten(), theta.flatten()))

    def _polar_to_cartesian(self, r, theta):
        # (r,theta) --> (x,y)
        return torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=-1)

    def _cartesian_to_polar(self, points):
        raise NotImplementedError


    def _robot_to_world(self, points, pos, rot):
        batch_size = pos.shape[0]
        # Pre-compute sine and cosine
        c, s = torch.cos(rot), torch.sin(rot)

        # Create rotation matrix
        R = torch.stack([c, -s, s, c], dim=1).view(batch_size, 2, 2)

        # Apply rotation and translation
        rotated = torch.bmm(points.view(batch_size, -1, self.lidar_cfg.space_dimension), R.transpose(1, 2))
        transformed = rotated + pos.unsqueeze(1).repeat(1, rotated.shape[1], 1)

        return transformed.view(-1, self.lidar_cfg.space_dimension)

    def _sensor_to_robot(self, points):
        # 2D transformation from sensor coordinate to robot coordinate
        c, s = torch.cos(self.sensor_angle), torch.sin(self.sensor_angle)

        x = c * points[..., 0] - s * points[..., 1] + self.sensor_location[0]
        y = s * points[..., 0] + c * points[..., 1] + self.sensor_location[1]

        return torch.stack((x, y), dim=-1)

    def _boundary_extractor(self, mesh):
        # Calculate barrier values
        z = self.map.map_barrier.min_barrier(mesh)
        # Reshape z and mesh
        batch_size = mesh.shape[0] // (self.lidar_cfg.ray_num * self.lidar_cfg.ray_sampling_rate)
        reshaped_z = z.view(batch_size, self.lidar_cfg.ray_num, self.lidar_cfg.ray_sampling_rate, 1)
        reshaped_mesh = mesh.view(batch_size, self.lidar_cfg.ray_num, self.lidar_cfg.ray_sampling_rate,
                                  self.lidar_cfg.space_dimension)

        # Find boundary points
        sign_change = (reshaped_z[:, :, :-1, :] >= 0) & (reshaped_z[:, :, 1:, :] < 0)
        change_indices = torch.nonzero(sign_change, as_tuple=True)

        # Initialize boundary points
        boundary_points = reshaped_mesh[:, :, -1, :]
        if change_indices[0].numel() > 0:
            # Find unique ray indices
            _, unique_indices = torch.unique(torch.stack(change_indices[:2]), dim=1, return_inverse=True)

            # Assign boundary points
            boundary_points[change_indices[0], change_indices[1]] = reshaped_mesh[
                change_indices[0], change_indices[1], change_indices[2]]
        return boundary_points

    def add_noise(self, points):
        # TODO: Complete the add_noise
        raise NotImplementedError
