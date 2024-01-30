import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import RasterizationSettings, AmbientLights, MeshRendererWithFragments, MeshRasterizer, BlendParams, SoftPhongShader, TexturesUV
import torch_scatter

from utils.misc import get_all_4_locations
from utils.shader import FlatTexelShader


def get_softphong_ambientlight_renderer(image_size, faces_per_pixel=1, device=torch.device("cuda:0")):
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel)
    lights = AmbientLights(device=device)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            lights=lights,
            device=device,
            blend_params=BlendParams()
        )
    )
    return renderer


def get_feature_atlas_renderer(image_size, faces_per_pixel=1, device=torch.device("cuda:0")):
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=FlatTexelShader(
            device=device,
            blend_params=BlendParams()
        )
    )
    return renderer


def backproject_to_atlas(mesh, faces, verts_uvs, camera, mapping_size, atlas_size, rendered_image, use_4_way_paste=False, device=torch.device("cuda:0")):
    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    mapping_renderer = get_softphong_ambientlight_renderer(mapping_size)
    mapping_fragments = mapping_renderer.rasterizer(mesh, cameras=camera)

    pixel_uvs = interpolate_face_attributes(
        mapping_fragments.pix_to_face, mapping_fragments.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(-1, 2)

    if use_4_way_paste:
        texture_locations_y, texture_locations_x = get_all_4_locations(
            (1 - pixel_uvs[:, 1]).reshape(-1) * (atlas_size - 1),
            pixel_uvs[:, 0].reshape(-1) * (atlas_size - 1)
        )
    else:
        texture_locations_y = torch.round((1 - pixel_uvs[:, 1]).reshape(-1) * (atlas_size - 1)).long()
        texture_locations_x = torch.round(pixel_uvs[:, 0].reshape(-1) * (atlas_size - 1)).long()

    if mapping_size != rendered_image.shape[0]:
        resampled_size = (mapping_size, mapping_size)
        # H_m x W_m x F
        texture_values = torch.nn.functional.interpolate(rendered_image.permute((2, 0, 1)).unsqueeze(0), size=resampled_size, mode='bilinear').squeeze(0).permute((1, 2, 0))
    else:
        texture_values = rendered_image

    # atlas H_a . W_a x F
    atlas_tensor = torch.zeros(atlas_size * atlas_size, rendered_image.shape[-1]).to(device)
    # replace with scatter mean
    torch_scatter.scatter_mean(texture_values.reshape(-1, texture_values.shape[-1]), (texture_locations_y * atlas_size + texture_locations_x).unsqueeze(-1), dim=0, out=atlas_tensor)
    atlas_tensor = atlas_tensor.reshape((atlas_size, atlas_size, rendered_image.shape[-1]))
    return atlas_tensor


def render_atlas_on_mesh(feature_atlas, mesh, faces, verts_uvs, camera, render_size, device=torch.device("cuda:0")):
    rmesh = mesh.clone()
    rmesh.textures = TexturesUV(
        maps=feature_atlas.unsqueeze(0),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...]
    )
    renderer = get_feature_atlas_renderer(render_size, device=device)
    images, _ = renderer(rmesh, cameras=camera)
    return images[0]
