import math
import torch


def fix_normal_map(view, normal, normal_in_view_space=True):
    """_summary_

    Args:
        view (_type_): _description_
        normal (_type_): 

    Returns:
        _type_: _description_
    """
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins_inv = torch.tensor(
        [[1/fx, 0.,-W/(2 * fx)],
        [0., 1/fy, -H/(2 * fy),],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0).reshape(3, -1).float().cuda()
    rays_d = (intrins_inv @ points).reshape(3, H, W)

    if normal_in_view_space:
        normal_view = normal
    else:
        normal_view = normal.clone()
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(1, 2, 0)
        normal_view = normal_view @ view.world_view_transform[:3,:3]
        if normal.shape[0] == 3:
            normal_view = normal_view.permute(2, 0, 1)

    if normal_view.shape[0] != 3:
        rays_d = rays_d.permute(1, 2, 0)
        dim_to_sum = -1
    else:
        dim_to_sum = 0

    return torch.sign((-rays_d * normal_view).sum(dim=dim_to_sum, keepdim=True)) * normal_view
