import numpy as np
from attrdict import AttrDict as AD
from cbftorch.barriers.barrier import Barrier
from cbftorch.utils.make_map import GeometryProvider
from cbftorch.utils.make_map import Map as BaseMap
from cbftorch.utils.utils import vectorize_tensors


class RandomGeometryProvider(GeometryProvider):
    """Geometry provider that generates random obstacles."""

    def __init__(self, random_generator, layout, floor_size, obstacle_size_range):
        self.random_generator = random_generator
        self.layout = AD(layout)
        self.floor_size = np.array(floor_size)
        self.obstacle_size_range = obstacle_size_range
        self._generated_geoms = []
        self._generate_geometries()

    def get_geometries(self):
        return self._generated_geoms

    def get_velocity_constraints(self):
        return self.layout.get('velocity', None)

    def _generate_geometries(self):
        """Generate actual geometries from layout specification."""
        self._generated_geoms = []

        for geom_type, geom_spec in self.layout.get('geoms', []):
            # Handle multiple instances
            num = geom_spec.pop('num', 1)

            for _ in range(num):
                if geom_type in ['box', 'norm_box', 'boundary', 'norm_boundary']:
                    geom_info = self._make_box(geom_spec.copy())
                elif geom_type == 'cylinder':
                    geom_info = self._make_cylinder(geom_spec.copy())
                else:
                    raise ValueError(f"Unsupported geometry type: {geom_type}")

                self._generated_geoms.append((geom_type, geom_info))

    def _make_box(self, spec):
        """Generate box parameters with random values where not specified."""
        # Generate random size if not specified
        if 'size' not in spec:
            spec['size'] = self._sample_size(2)

        # Generate random center if not specified
        if 'center' not in spec:
            spec['center'] = self._sample_center(spec['size'])

        # Generate random rotation if not specified
        if 'rotation' not in spec:
            spec['rotation'] = self._sample_rotation()

        return spec

    def _make_cylinder(self, spec):
        """Generate cylinder parameters with random values where not specified."""
        # Generate random radius if not specified
        if 'radius' not in spec:
            spec['radius'] = self._sample_size(1)

        # Generate random center if not specified
        if 'center' not in spec:
            spec['center'] = self._sample_center(spec['radius'])

        return spec

    def _sample_size(self, ndim):
        """Sample random size within configured range."""
        size = self.random_generator.uniform(
            low=self.obstacle_size_range[0],
            high=self.obstacle_size_range[1],
            size=ndim
        )
        return list(size) if ndim > 1 else float(size[0])

    def _sample_center(self, obj_size):
        """Sample random center position with margin from boundaries."""
        if isinstance(obj_size, (list, tuple)):
            margin = max(obj_size)
        else:
            margin = obj_size

        return list(self.random_generator.uniform(
            low=-self.floor_size + margin,
            high=self.floor_size - margin,
            size=2
        ))

    def _sample_rotation(self):
        """Sample random rotation angle."""
        return float(self.random_generator.uniform(low=-np.pi, high=np.pi))


class Map(BaseMap):
    """Extended Map class with random generation capabilities."""

    def __init__(self, robot, random_generator, layout={}, cfg={}):
        """
        Initialize map with random obstacle generation.

        Args:
            robot: Robot instance with dynamics
            random_generator: Random number generator
            layout: Layout specification dict
            cfg: Configuration dict
        """
        # Create random geometry provider
        geometry_provider = RandomGeometryProvider(
            random_generator=random_generator,
            layout=layout,
            floor_size=cfg['floor_size'],
            obstacle_size_range=cfg['obstacle_size_range']
        )

        # Initialize base map
        super().__init__(
            dynamics=robot.dynamics,
            cfg=cfg.get('map_config', cfg),
            geometry_provider=geometry_provider
        )

        # Store additional attributes
        self.robot = robot
        self.random_generator = random_generator
        self.floor_size = np.array(cfg['floor_size'])
        self.obstacle_size_range = cfg['obstacle_size_range']

    @property
    def layout(self):
        """For backward compatibility with existing code."""
        return self.geometry_provider.layout

    def regenerate(self):
        """Regenerate random obstacles with same configuration."""
        self.geometry_provider._generate_geometries()
        self._create_barriers()


class EmptyMap(BaseMap):
    """Empty map with no obstacles."""

    class EmptyProvider(GeometryProvider):
        def get_geometries(self):
            return []

        def get_velocity_constraints(self):
            return None

    def __init__(self, robot, **kwargs):
        """Initialize empty map with trivial barrier."""
        super().__init__(
            dynamics=robot.dynamics,
            cfg=kwargs.get('cfg', {}),
            geometry_provider=self.EmptyProvider()
        )

        # Override barrier to be trivial (always safe)
        self.barrier = Barrier().assign(
            barrier_func=lambda x: 1.0 + 0.0 * vectorize_tensors(x).sum(-1),
            rel_deg=1
        ).assign_dynamics(robot.dynamics)

        # Also make map_barrier trivial
        self.map_barrier = self.barrier
