from pygltflib import GLTF2
import numpy as np
import base64

# Replace 'your_model.glb' with the path to your .glb file
glb_file = '/home/mingjun/PycharmProjects/smoothing/Robust_Algorithm/Input_Data/glb_and_txt_file/tmpxu9o_zug_scene.glb'

# Load the .glb file
gltf = GLTF2().load(glb_file)

def get_data_from_accessor(gltf, accessor_idx):
    accessor = gltf.accessors[accessor_idx]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]

    # For .glb files, buffer data is in gltf.binary_blob()
    buffer_bytes = gltf.binary_blob()

    # Calculate the start and end positions
    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    component_type_size = {
        5120: 1,  # int8
        5121: 1,  # uint8
        5122: 2,  # int16
        5123: 2,  # uint16
        5125: 4,  # uint32
        5126: 4,  # float32
    }[accessor.componentType]
    num_components = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16
    }[accessor.type]
    length = accessor.count * component_type_size * num_components
    end = start + length

    buffer_data = buffer_bytes[start:end]

    # Map component type to numpy dtype
    dtype_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    dtype = dtype_map[accessor.componentType]

    # Read the data
    array = np.frombuffer(buffer_data, dtype=dtype)
    array = array.reshape(accessor.count, num_components)
    return array

# Collect all positions and colors
all_positions = []
all_colors = []

# Iterate over all meshes and primitives
for mesh in gltf.meshes:
    for primitive in mesh.primitives:
        # Extract positions
        positions_accessor_idx = primitive.attributes.POSITION
        positions = get_data_from_accessor(gltf, positions_accessor_idx)
        all_positions.append(positions)

        # Extract colors if available
        if hasattr(primitive.attributes, 'COLOR_0') and primitive.attributes.COLOR_0 is not None:
            colors_accessor_idx = primitive.attributes.COLOR_0
            colors = get_data_from_accessor(gltf, colors_accessor_idx)
            # Convert colors to 0-255 range if necessary
            if colors.dtype == np.float32 and colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            elif colors.dtype != np.uint8:
                colors = colors.astype(np.uint8)
        else:
            # Assign default colors if none are present
            colors = np.zeros((positions.shape[0], 3), dtype=np.uint8)
        all_colors.append(colors[:, :3])

# Concatenate all positions and colors
positions = np.vstack(all_positions)
colors = np.vstack(all_colors)

# Combine positions and colors
data = np.hstack((positions, colors))

# Save data to a .txt file
np.savetxt('/home/mingjun/PycharmProjects/smoothing/Robust_Algorithm/Input_Data/glb_and_txt_file/segmented_ash.txt', data, fmt='%f %f %f %d %d %d')

print(f'Conversion complete. Total points: {len(positions)}. Data saved to output.txt')
