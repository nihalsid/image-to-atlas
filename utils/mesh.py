import torch
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import TexturesUV
from torchvision import transforms
from PIL import Image


def load_mesh_with_dummy_textures(path, device=torch.device("cuda:0")):
    verts, faces, aux = load_obj(path, device=device)
    mean = verts.mean(dim=0).unsqueeze(0)
    verts = verts - mean
    mesh = load_objs_as_meshes([path], device=device)
    mesh = mesh.offset_verts(-mean.expand(verts.shape[0], -1))
    dummy_texture = Image.open("resources/dummy_texture.png").convert('RGB')
    mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(dummy_texture)[None, ...].permute(0, 2, 3, 1).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=aux.verts_uvs[None, ...]
    )
    return mesh, verts, faces, aux
