import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.shader import ShaderBase


def phong_normal_shading(meshes, fragments):
    faces = meshes.faces_packed()
    vertex_normals = meshes.verts_normals_packed()
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )

    return pixel_normals


def similarity_shading(meshes, fragments, cameras):
    faces = meshes.faces_packed()
    vertex_normals = meshes.verts_normals_packed()
    faces_normals = vertex_normals[faces]
    vertices = meshes.verts_packed()
    face_positions = vertices[faces]
    view_directions = torch.nn.functional.normalize((cameras.get_camera_center().reshape(1, 1, 3) - face_positions), p=2, dim=2)
    cosine_similarity = torch.nn.CosineSimilarity(dim=2)(faces_normals, view_directions)
    pixel_similarity = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, cosine_similarity.unsqueeze(-1)
    )
    pixel_similarity = pixel_similarity * 0.5 + 0.5
    # similarity for invalid pixels? 0 or 0.5?
    # pixel_similarity[fragments.pix_to_face == -1] = 0
    return pixel_similarity


def binary_shading(meshes, fragments):
    faces = meshes.faces_packed()
    vertex_normals = meshes.verts_normals_packed()
    faces_normals = vertex_normals[faces]
    object_mask = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, torch.ones_like(faces_normals)
    )
    return object_mask


def get_relative_depth_map(fragments, pad_value=10):
    absolute_depth = fragments.zbuf[..., 0]
    no_depth = -1
    depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
    target_min, target_max = 50, 255

    depth_value = absolute_depth[absolute_depth != no_depth]
    depth_value = depth_max - depth_value  # reverse values

    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (target_max - target_min) + target_min

    relative_depth = absolute_depth.clone()
    relative_depth[absolute_depth != no_depth] = depth_value
    relative_depth[absolute_depth == no_depth] = pad_value  # not completely black

    return relative_depth


class FlatTexelShader(ShaderBase):

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__(device, cameras, lights, materials, blend_params)

    def forward(self, fragments, meshes, **_kwargs):
        texels = meshes.sample_textures(fragments)
        texels[(fragments.pix_to_face == -1), :] = 0
        return texels.squeeze(-2)
