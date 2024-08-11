from attrdict import AttrDict as AD
from hocbf_composition.barrier import Barrier, SoftCompositionBarrier, NonSmoothCompositionBarrier
from hocbf_composition.utils.utils import *
from obstacle_rich_env.envs.robot import Robot


class Map:
    def __init__(self, robot: Robot, random_generator, layout: dict = {}, cfg: dict = {}):
        self.cfg = AD(cfg)
        self.floor_size = np.array(self.cfg['floor_size'])
        self.obstacle_size_range = self.cfg['obstacle_size_range']

        self.random_generator = random_generator
        self.robot = robot
        self.dynamics = robot.dynamics

        # Update
        self.layout = AD(layout)
        self.update_geoms_layout()
        self.make_barrier_from_map()

    def make_barrier_from_map(self):
        self.pos_barriers = self.make_position_barrier_from_map()
        self.vel_barriers = self.make_velocity_barrier()
        self.barrier = SoftCompositionBarrier(
            cfg=self.cfg['map_config']).assign_barriers_and_rule(barriers=[*self.pos_barriers, *self.vel_barriers],
                                                                 rule='i',
                                                                 infer_dynamics=True)

        self.map_barrier = NonSmoothCompositionBarrier(
            cfg=self.cfg).assign_barriers_and_rule(barriers=[*self.pos_barriers],
                                                   rule='i',
                                                   infer_dynamics=True)
    def get_barriers(self):
        return self.pos_barriers, self.vel_barriers

    def make_position_barrier_from_map(self):
        geoms = self.layout.geoms
        barriers = []
        cfg = self.cfg['map_config']
        for geom_type, geom_info in geoms:
            if geom_type == 'cylinder':
                barrier_func = make_circle_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(cfg.obstacle_alpha)
            elif geom_type == 'box':
                barrier_func = make_affine_rectangular_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(cfg.obstacle_alpha)
            elif geom_type == 'norm_box':
                barrier_func = make_norm_rectangular_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(cfg.obstacle_alpha)
            elif geom_type == 'boundary':
                barrier_func = make_affine_rectangular_boundary_functional
                alphas = make_linear_alpha_function_form_list_of_coef(cfg.boundary_alpha)
            elif geom_type == 'norm_boundary':
                barrier_func = make_norm_rectangular_boundary_functional
                alphas = make_linear_alpha_function_form_list_of_coef(cfg.boundary_alpha)
            else:
                raise NotImplementedError
            barriers.append(
                Barrier().assign(barrier_func=barrier_func(**geom_info),
                                 rel_deg=cfg.pos_barrier_rel_deg,
                                 alphas=alphas).assign_dynamics(self.dynamics))
        return barriers

    def make_velocity_barrier(self):
        if 'velocity' not in self.layout or self.layout.velocity is None or len(self.layout.velocity) == 0:
            return []
        cfg = self.cfg['map_config']
        alphas = make_linear_alpha_function_form_list_of_coef(cfg.velocity_alpha)
        idx, bounds = self.layout.velocity
        vel_barriers = make_box_barrier_functionals(bounds=bounds, idx=idx)
        barriers = [Barrier().assign(
            barrier_func=vel_barrier,
            rel_deg=cfg.vel_barrier_rel_deg,
            alphas=alphas).assign_dynamics(self.dynamics) for vel_barrier in vel_barriers]

        return barriers

    def update_geoms_layout(self):
        updated_geom = []

        for geom_type, geom_data in self.layout.geoms:
            num = 1
            if 'num' in geom_data:
                if geom_data.num > 1:
                    # ignore center and rotation data
                    if 'center' in geom_data: del geom_data.center
                    if 'rotation' in geom_data: del geom_data.rotation
                    num = geom_data.num
                del geom_data.num

            for _ in range(num):
                if geom_type == 'box':
                    updated_geom.append(('box', self._make_box(geom_data)))
                elif geom_type == 'norm_box':
                    updated_geom.append(('norm_box', self._make_box(geom_data)))
                elif geom_type == 'cylinder':
                    updated_geom.append(('cylinder', self._make_cylinder(geom_data)))
                elif geom_type == 'boundary':
                    updated_geom.append(('boundary', self._make_box(geom_data)))
                elif geom_type == 'norm_boundary':
                    updated_geom.append(('norm_boundary', self._make_box(geom_data)))
                else:
                    raise 'geom_type is not supported'

        self.layout.geoms = updated_geom

    def _make_box(self, geom_data):
        geom_data.update(size=self._sample_size(2) if 'size' not in geom_data else geom_data.size)
        geom_data.update(center=self._sample_center(size) if 'center' not in geom_data else geom_data.center)
        geom_data.update(rotation=self._sample_rot() if 'rotation' not in geom_data else geom_data.rotation)
        return AD(geom_data)


    def _make_cylinder(self, geom_data):
        geom_data.update(radius=self._sample_size(1) if 'radius' not in geom_data else geom_data.radius)
        geom_data.update(center=self._sample_center(radius) if 'center' not in geom_data else geom_data.center)
        return AD(geom_data)

    def _generate_random(self, low, high, size):
        return list(self.random_generator.uniform(low=low, high=high, size=size))

    def _sample_center(self, obj_size):
        margin = max(obj_size) if (isinstance(obj_size, list) or isinstance(obj_size, tuple)) else obj_size
        return self._generate_random(low=-self.floor_size + margin, high=self.floor_size - margin, size=2)

    def _sample_size(self, size_numel):
        size = self._generate_random(low=self.obstacle_size_range[0], high=self.obstacle_size_range[1], size=size_numel)
        return size if isinstance(size, list) else size[0]

    def _sample_rot(self):
        return self._generate_random(low=-np.pi, high=np.pi, size=1)[0]


class EmptyMap(Map):
    def make_barrier_from_map(self):
        self.barrier = Barrier().assign(barrier_func=lambda x: 1.0 + 0.0 * vectorize_tensors(x).sum(-1),
                                        rel_deg=1).assign_dynamics(self.dynamics)

    def make_velocity_barrier(self):
        pass

    def update_geoms_layout(self):
        pass

    def make_position_barrier_from_map(self):
        pass
