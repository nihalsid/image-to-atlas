import torch

from utils.camera import get_camera
from utils.mesh import load_mesh_with_dummy_textures
from utils.misc import concat_pil_images, tensor1F_to_PIL3U, tensor3F_to_PIL3U
from utils.renderer import get_softphong_ambientlight_renderer, backproject_to_atlas, render_atlas_on_mesh
from utils.shader import similarity_shading, binary_shading


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


def test_render_texture_map():
    # render test
    distance, elevation, azimuth = 4, 15, 30
    image_size = 512
    atlas_size = 256
    mapping_size = 1024
    mesh_path = "data/samples/nascar/nascar_blender.obj"
    mesh, _verts, faces, aux = load_mesh_with_dummy_textures(mesh_path)
    camera_0 = get_camera(distance, elevation, azimuth, image_size)
    renderer = get_softphong_ambientlight_renderer(image_size)
    _, fragments = renderer(mesh, cameras=camera_0)
    # create a dummy render image as an example
    rendered_image_0 = binary_shading(mesh, fragments)[0, :, :, 0, :]  # HxWx1
    output = tensor3F_to_PIL3U(rendered_image_0).resize((256, 256))
    output.save("runs/test_render_0.jpg")

    distance, elevation, azimuth = 4, 15, 120
    camera_1 = get_camera(distance, elevation, azimuth, image_size)
    _, fragments = renderer(mesh, cameras=camera_1)
    rendered_image_1 = binary_shading(mesh, fragments)[0, :, :, 0, :]  # HxWx1
    output = tensor3F_to_PIL3U(rendered_image_1).resize((256, 256))
    output.save("runs/test_render_1.jpg")
    
    valid_map_0 = backproject_to_atlas(mesh, faces, aux.verts_uvs, camera_0, mapping_size, atlas_size, torch.ones_like(rendered_image_0)) > 0  # H'xW'x1
    valid_map_1 = backproject_to_atlas(mesh, faces, aux.verts_uvs, camera_1, mapping_size, atlas_size, torch.ones_like(rendered_image_0)) > 0  # H'xW'x1
    valid_map_01 = torch.logical_and(valid_map_0, valid_map_1).float() 

    valid_01_itersect = concat_pil_images([
        tensor1F_to_PIL3U(valid_map_0).resize((256, 256)),
        tensor1F_to_PIL3U(valid_map_1).resize((256, 256)),
        tensor1F_to_PIL3U(valid_map_01).resize((256, 256))
    ])
    valid_01_itersect.save(f"runs/test_valid_union.jpg")

    all_renders = []
    for azimuth in range(0, 360, 60):
        camera = get_camera(distance, elevation, azimuth, image_size)
        rendered = render_atlas_on_mesh(valid_map_01, mesh, faces, aux.verts_uvs, camera, image_size)
        all_renders.append(tensor1F_to_PIL3U(rendered))
    
    rendered_texture = concat_pil_images(all_renders)
    rendered_texture.save(f"runs/test_texture.jpg")



if __name__ == "__main__":
    # test image -> atlas
    test_image_to_atlas()
    # test intersection between two views
    test_render_texture_map()
