from attrdict import AttrDict as AD

map_layout_goal0 = dict()

map_layout_goal1 = dict(
    geoms=(
        ('cylinder', AD(center=[-5.0, 0.0], radius=2.0)),
        ('cylinder', AD(center=[5.0, 0.0], radius=2.0)),
    ),
)

map_layout_goal2 = dict(
    geoms=(
        ('cylinder', AD(center=[-5.0, 0.0], radius=2.0)),
        ('cylinder', AD(center=[5.0, 0.0], radius=2.0)),
        ('cylinder', AD(center=[0.0, 5.0], radius=2.0)),
        ('cylinder', AD(center=[0.0, -5.0], radius=2.0)),
    ),
    velocity=(2, [-9.0, 9.0]),
)

map_layout_goal3 = dict(
    geoms=(
        ('cylinder', AD(center=[-2.5, 0.0], radius=2.5)),
        ('cylinder', AD(center=[2.5, 0.0], radius=2.5)),
        ('box', AD(center=[6.5, 6.5], size=[1.0, 1.0], rotation=0.0)),
        ('box', AD(center=[-6.5, 6.5], size=[1.0, 1.0], rotation=0.0)),
        ('box', AD(center=[-6.5, -6.5], size=[1.0, 1.0], rotation=0.0)),
        ('box', AD(center=[6.5, -6.5], size=[1.0, 1.0], rotation=0.0)),

    ),
    velocity=(2, [-9.0, 9.0]),
)

map_layout_goal4 = dict(
    geoms=(
        ('box', AD(center=[2.0, 1.5], size=[2.0, 2.0], rotation=0.0)),
        ('box', AD(center=[-2.5, 2.5], size=[1.25, 1.25], rotation=0.0)),
        ('box', AD(center=[-5.0, -5.0], size=[1.875, 1.875], rotation=0.0)),
        ('box', AD(center=[5.0, -6.0], size=[3.0, 3.0], rotation=0.0)),
        ('box', AD(center=[-7.0, 5.0], size=[2.0, 2.0], rotation=0.0)),
        ('box', AD(center=[6.0, 7.0], size=[2.0, 2.0], rotation=0.0)),
        ('boundary', AD(center=[0.0, 0.0], size=[10.0, 10.0], rotation=0.0)),
    ),
    velocity=(2, [-1.0, 9.0]),
)

