import torch
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras


def get_camera(distance, elevation, azimuth, image_size, device=torch.device("cuda:0")):
    R, T = look_at_view_transform(distance, elevation, azimuth)
    camera = PerspectiveCameras(R=R, T=T, device=device, image_size=torch.tensor([image_size, image_size]).unsqueeze(0))
    return camera

