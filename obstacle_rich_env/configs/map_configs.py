from attrdict import AttrDict as AD

map_config_unicycle = AD(softmax_rho=20,
                         softmin_rho=20,
                         pos_barrier_rel_deg=2,
                         vel_barrier_rel_deg=1,
                         obstacle_alpha=[2.5],
                         boundary_alpha=[1.0],
                         velocity_alpha=[],
                         )
