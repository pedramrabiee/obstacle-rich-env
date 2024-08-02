from attrdict import AttrDict as AD
import torch

lidar_configs = AD(max_range=5,
                   ray_num=100,
                   scan_angle=[-torch.pi, torch.pi],
                   update_rate=0.2,
                   ray_sampling_rate=1000,
                   space_dimension=2,
                   sensor_mounting_location=[0, 0],
                   sensor_mounting_angle=[0],
                   has_noise=False,
                   return_cartesian=True
                   )
