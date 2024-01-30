import torch

from utils.camera import get_camera
from utils.mesh import load_mesh_with_dummy_textures
from utils.misc import concat_pil_images, tensor1F_to_PIL3U
from utils.renderer import get_softphong_ambientlight_renderer, backproject_to_atlas
from utils.shader import similarity_shading


def test_image_to_atlas():
    # projection test

    distance, elevation, azimuth = 4, 15, 30
    image_size = 256
    atlas_size = 256
    mapping_size = 256
    mesh_path = "data/samples/nascar/nascar_blender.obj"
    mesh, _verts, faces, aux = load_mesh_with_dummy_textures(mesh_path)
    camera = get_camera(distance, elevation, azimuth, image_size)
    renderer = get_softphong_ambientlight_renderer(image_size)
    _, fragments = renderer(mesh, cameras=camera)
    # create a dummy render image as an example
    rendered_image = similarity_shading(mesh, fragments, camera)[0, :, :, :, 0]  # HxWx1
    # backproject the rendered image to the atlas
    atlas_map = backproject_to_atlas(mesh, faces, aux.verts_uvs, camera, mapping_size, atlas_size, rendered_image)  # H'xW'x1
    valid_map = backproject_to_atlas(mesh, faces, aux.verts_uvs, camera, mapping_size, atlas_size, torch.ones_like(rendered_image)) > 0  # H'xW'x1
    output = concat_pil_images([
        tensor1F_to_PIL3U(rendered_image).resize((256, 256)),
        tensor1F_to_PIL3U(atlas_map).resize((256, 256)),
        tensor1F_to_PIL3U(valid_map).resize((256, 256))
    ])
    output.save(f"runs/test_f{image_size:03d}_a{atlas_size:03d}_m{mapping_size:03d}.jpg")


if __name__ == "__main__":
    test_image_to_atlas()
