bl_info = {
    "name": "Import PSD to 3D Comic",
    "description": "",
    "author": "funige",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "support": "TESTING",
    "category": "Import-Exoprt",
    "location": "File > Import > PSD to 3D Comic",
    "warning": "",
    "wiki_url": "",
}

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

from bpy_extras.object_utils import (
    AddObjectHelper,
    world_to_camera_view,
)

from psd_tools import PSDImage
from bpy_extras.image_utils import load_image

import os
from math import pi
from mathutils import Vector

class IMPORT_OT_psd_to_3dcomic(bpy.types.Operator, ImportHelper):
    bl_idname= "import_image.psd_to_3dcomic"
    bl_label = "Import PSD to 3D Comic"
    filename_ext = ".psd"
    filter_glob = StringProperty(default="*.psd", options={'HIDDEN'})

    def execute(self, context):
        path = self.properties.filepath
        psd = PSDImage.open(path)
        for layer in psd:
            print(layer)

        # カメラ設定
        height = 1024 # 画像は高さ1024pxで作成
        bpy.context.scene.render.resolution_x = height * (psd.width / psd.height)
        bpy.context.scene.render.resolution_y = height
        bpy.context.scene.camera.location = (0, -12, 0)
        bpy.context.scene.camera.rotation_euler = (pi / 2, 0, 0)

        # 立体視の設定
        bpy.context.scene.camera.data.stereo.convergence_distance = 12
        bpy.context.scene.render.use_multiview = True
        bpy.context.scene.render.image_settings.views_format = 'STEREO_3D'
        bpy.context.scene.render.image_settings.stereo_3d_format.display_mode = 'ANAGLYPH'
        
        # psdと同じディレクトリにテクスチャを置くフォルダを用意する
        basename = bpy.path.basename(path)
        directory = os.path.join(os.path.dirname(path), os.path.splitext(basename)[0])
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # カメラに対して等間隔にレイヤーを配置
        self.names = {}
        y = 0
        for layer in reversed(psd):
            if layer.is_visible():
                image = self.create_layer_image(layer, directory)
                material = self.create_material(image)
                plane = self.create_plane(context, y, psd.size, layer.bbox)

                plane.data.name = plane.name = material.name
                plane.data.materials.append(material)                
                y = y + 4
            
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def create_layer_image(self, layer, directory):
        texture_name = self.get_unique_name(layer.name) + ".png"
        path = os.path.join(directory, texture_name)
        if not os.path.exists(path):
            layer.composite().save(path)
        return load_image(texture_name, directory, force_reload=True)

    def get_unique_name(self, name):
        if name in self.names:
            self.names[name] += 1
            return name + '.' + format(self.names[name], '0>3')
        else:
            self.names[name] = 0
            return name

    def create_texnode(self, node_tree, image):
        tex_image = node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = image
        tex_image.show_texture = True
        tex_image.extension = 'CLIP'
        return tex_image

    def create_material(self, image):
        name_compat = bpy.path.display_name_from_filepath(image.filepath)
        material = None
        for mat in bpy.data.materials:
            if mat.name == name_compat:
                material = mat
        if not material:
            material = bpy.data.materials.new(name=name_compat)

        material.use_nodes = True
        material.blend_method = 'BLEND'
        node_tree = material.node_tree
        out_node = clean_node_tree(node_tree)
        tex_image = self.create_texnode(node_tree, image)
        core_shader = get_shadeless_node(node_tree)
        node_tree.links.new(core_shader.inputs[0], tex_image.outputs[0])

        bsdf_transparent = node_tree.nodes.new('ShaderNodeBsdfTransparent')
        mix_shader = node_tree.nodes.new('ShaderNodeMixShader')
        node_tree.links.new(mix_shader.inputs[0], tex_image.outputs[1])
        node_tree.links.new(mix_shader.inputs[1], bsdf_transparent.outputs[0])
        node_tree.links.new(mix_shader.inputs[2], core_shader.outputs[0])
        core_shader = mix_shader
        node_tree.links.new(out_node.inputs[0], core_shader.outputs[0])

        auto_align_nodes(node_tree)
        return material

    def create_plane(self, context, offset, size, bbox):
        iw = size[0]
        ih = size[1]

        w, h = compute_camera_size(context, Vector([0, offset, 0]), 'FIT', iw / ih)

        sx = (bbox[2] - bbox[0]) / iw
        sy = (bbox[3] - bbox[1]) / ih

        x = (bbox[2] + bbox[0]) / (iw * 2) - 0.5
        y = (bbox[3] + bbox[1]) / (ih * 2) - 0.5
        
        bpy.ops.mesh.primitive_plane_add('INVOKE_REGION_WIN')
        plane = context.active_object
        if plane.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        plane.dimensions = w * sx, h * sy, 0.0
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        plane.location.x = w * x
        plane.location.y = offset
        plane.location.z = -h * y
        plane.rotation_euler.x = pi / 2
        return plane

# -----------------------------------------------------------------------------
# Position & Size Helpers
# io_import_images_as_planes.py から丸写し

def compute_camera_size(context, center, fill_mode, aspect):
    """Determine how large an object needs to be to fit or fill the camera's field of view."""
    scene = context.scene
    camera = scene.camera
    view_frame = camera.data.view_frame(scene=scene)
    frame_size = \
        Vector([max(v[i] for v in view_frame) for i in range(3)]) - \
        Vector([min(v[i] for v in view_frame) for i in range(3)])
    camera_aspect = frame_size.x / frame_size.y

    # Convert the frame size to the correct sizing at a given distance
    if camera.type == 'ORTHO':
        frame_size = frame_size.xy
    else:
        # Perspective transform
        distance = world_to_camera_view(scene, camera, center).z
        frame_size = distance * frame_size.xy / (-view_frame[0].z)

    # Determine what axis to match to the camera
    match_axis = 0  # match the Y axis size
    match_aspect = aspect
    if (fill_mode == 'FILL' and aspect > camera_aspect) or \
            (fill_mode == 'FIT' and aspect < camera_aspect):
        match_axis = 1  # match the X axis size
        match_aspect = 1.0 / aspect

    # scale the other axis to the correct aspect
    frame_size[1 - match_axis] = frame_size[match_axis] / match_aspect

    return frame_size

# -----------------------------------------------------------------------------
# Cycles/Eevee utils
# io_import_images_as_planes.py から丸写し
    
def get_input_nodes(node, links):
    """Get nodes that are a inputs to the given node"""
    # Get all links going to node.
    input_links = {lnk for lnk in links if lnk.to_node == node}
    # Sort those links, get their input nodes (and avoid doubles!).
    sorted_nodes = []
    done_nodes = set()
    for socket in node.inputs:
        done_links = set()
        for link in input_links:
            nd = link.from_node
            if nd in done_nodes:
                # Node already treated!
                done_links.add(link)
            elif link.to_socket == socket:
                sorted_nodes.append(nd)
                done_links.add(link)
                done_nodes.add(nd)
        input_links -= done_links
    return sorted_nodes


def auto_align_nodes(node_tree):
    """Given a shader node tree, arrange nodes neatly relative to the output node."""
    x_gap = 200
    y_gap = 180
    nodes = node_tree.nodes
    links = node_tree.links
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL' or node.type == 'GROUP_OUTPUT':
            output_node = node
            break

    else:  # Just in case there is no output
        return

    def align(to_node):
        from_nodes = get_input_nodes(to_node, links)
        for i, node in enumerate(from_nodes):
            node.location.x = min(node.location.x, to_node.location.x - x_gap)
            node.location.y = to_node.location.y
            node.location.y -= i * y_gap
            node.location.y += (len(from_nodes) - 1) * y_gap / (len(from_nodes))
            align(node)

    align(output_node)

    
def clean_node_tree(node_tree):
    """Clear all nodes in a shader node tree except the output.

    Returns the output node
    """
    nodes = node_tree.nodes
    for node in list(nodes):  # copy to avoid altering the loop's data source
        if not node.type == 'OUTPUT_MATERIAL':
            nodes.remove(node)

    return node_tree.nodes[0]


def get_shadeless_node(dest_node_tree):
    """Return a "shadless" cycles/eevee node, creating a node group if nonexistent"""
    try:
        node_tree = bpy.data.node_groups['IAP_SHADELESS']

    except KeyError:
        # need to build node shadeless node group
        node_tree = bpy.data.node_groups.new('IAP_SHADELESS', 'ShaderNodeTree')
        output_node = node_tree.nodes.new('NodeGroupOutput')
        input_node = node_tree.nodes.new('NodeGroupInput')

        node_tree.outputs.new('NodeSocketShader', 'Shader')
        node_tree.inputs.new('NodeSocketColor', 'Color')

        # This could be faster as a transparent shader, but then no ambient occlusion
        diffuse_shader = node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        node_tree.links.new(diffuse_shader.inputs[0], input_node.outputs[0])

        emission_shader = node_tree.nodes.new('ShaderNodeEmission')
        node_tree.links.new(emission_shader.inputs[0], input_node.outputs[0])

        light_path = node_tree.nodes.new('ShaderNodeLightPath')
        is_glossy_ray = light_path.outputs['Is Glossy Ray']
        is_shadow_ray = light_path.outputs['Is Shadow Ray']
        ray_depth = light_path.outputs['Ray Depth']
        transmission_depth = light_path.outputs['Transmission Depth']

        unrefracted_depth = node_tree.nodes.new('ShaderNodeMath')
        unrefracted_depth.operation = 'SUBTRACT'
        unrefracted_depth.label = 'Bounce Count'
        node_tree.links.new(unrefracted_depth.inputs[0], ray_depth)
        node_tree.links.new(unrefracted_depth.inputs[1], transmission_depth)

        refracted = node_tree.nodes.new('ShaderNodeMath')
        refracted.operation = 'SUBTRACT'
        refracted.label = 'Camera or Refracted'
        refracted.inputs[0].default_value = 1.0
        node_tree.links.new(refracted.inputs[1], unrefracted_depth.outputs[0])
        
        reflection_limit = node_tree.nodes.new('ShaderNodeMath')
        reflection_limit.operation = 'SUBTRACT'
        reflection_limit.label = 'Limit Reflections'
        reflection_limit.inputs[0].default_value = 2.0
        node_tree.links.new(reflection_limit.inputs[1], ray_depth)

        camera_reflected = node_tree.nodes.new('ShaderNodeMath')
        camera_reflected.operation = 'MULTIPLY'
        camera_reflected.label = 'Camera Ray to Glossy'
        node_tree.links.new(camera_reflected.inputs[0], reflection_limit.outputs[0])
        node_tree.links.new(camera_reflected.inputs[1], is_glossy_ray)

        shadow_or_reflect = node_tree.nodes.new('ShaderNodeMath')
        shadow_or_reflect.operation = 'MAXIMUM'
        shadow_or_reflect.label = 'Shadow or Reflection?'
        node_tree.links.new(shadow_or_reflect.inputs[0], camera_reflected.outputs[0])
        node_tree.links.new(shadow_or_reflect.inputs[1], is_shadow_ray)

        shadow_or_reflect_or_refract = node_tree.nodes.new('ShaderNodeMath')
        shadow_or_reflect_or_refract.operation = 'MAXIMUM'
        shadow_or_reflect_or_refract.label = 'Shadow, Reflect or Refract?'
        node_tree.links.new(shadow_or_reflect_or_refract.inputs[0], shadow_or_reflect.outputs[0])
        node_tree.links.new(shadow_or_reflect_or_refract.inputs[1], refracted.outputs[0])

        mix_shader = node_tree.nodes.new('ShaderNodeMixShader')
        node_tree.links.new(mix_shader.inputs[0], shadow_or_reflect_or_refract.outputs[0])
        node_tree.links.new(mix_shader.inputs[1], diffuse_shader.outputs[0])
        node_tree.links.new(mix_shader.inputs[2], emission_shader.outputs[0])

        node_tree.links.new(output_node.inputs[0], mix_shader.outputs[0])

        auto_align_nodes(node_tree)

    group_node = dest_node_tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = node_tree

    return group_node

# -----------------------------------------------------------------------------
# Register

def menu_fn(self, context):
    self.layout.operator(IMPORT_OT_psd_to_3dcomic.bl_idname, text="PSD to 3D Comic")
    
classes = [
    IMPORT_OT_psd_to_3dcomic,
]
    
def register():
    bpy.types.TOPBAR_MT_file_import.append(menu_fn)
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_fn)
    for c in classes:
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()

