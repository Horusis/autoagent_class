import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def align_red_rectangle_to_xz(base_points, all_points=None):
    base_points = np.asarray(base_points, dtype=np.float64)

    if all_points is not None:
        all_points = np.asarray(all_points, dtype=np.float64)
        assert all_points.ndim == 2 and all_points.shape[1] == 3

    center = base_points.mean(axis=0)
    X = base_points - center

    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    axis0 = Vt[0]
    axis1 = Vt[1]

    proj0 = X @ axis0
    proj1 = X @ axis1
    len0 = proj0.max() - proj0.min()
    len1 = proj1.max() - proj1.min()

    if len0 >= len1:
        long_axis = axis0
        short_axis = axis1
    else:
        long_axis = axis1
        short_axis = axis0

    long_axis = long_axis / np.linalg.norm(long_axis)

    short_axis = short_axis - np.dot(short_axis, long_axis) * long_axis
    short_axis = short_axis / np.linalg.norm(short_axis)

    normal_axis = np.cross(long_axis, short_axis)
    normal_axis = normal_axis / np.linalg.norm(normal_axis)

    world_to_aligned = np.stack(
        [short_axis, normal_axis, long_axis], axis=1
    )

    aligned_to_world = world_to_aligned.T

    base_local = (base_points - center) @ world_to_aligned

    y_med = np.median(base_local[:, 1])
    base_local[:, 1] -= y_med

    min_x, max_x = base_local[:, 0].min(), base_local[:, 0].max()
    min_z, max_z = base_local[:, 2].min(), base_local[:, 2].max()

    width = max_x - min_x
    height = max_z - min_z

    post_translation = np.array([
        -min_x,
        0.0,
        -min_z
    ], dtype=np.float64)

    base_aligned = base_local + post_translation

    rectangle_corners_aligned = np.array([
        [0.0,   0.0,    0.0],
        [width, 0.0,    0.0],
        [width, 0.0, height],
        [0.0,   0.0, height],
    ], dtype=np.float64)

    rectangle_corners_world = (
        (rectangle_corners_aligned - post_translation + np.array([0.0, y_med, 0.0]))
        @ aligned_to_world
    ) + center

    result = {
        "base_aligned": base_aligned,
        "all_aligned": None,
        "world_to_aligned": world_to_aligned,
        "aligned_to_world": aligned_to_world,
        "center_world": center,
        "post_translation": post_translation,
        "rectangle_corners_aligned": rectangle_corners_aligned,
        "rectangle_corners_world": rectangle_corners_world,
        "width": width,
        "height": height,
    }

    if all_points is not None:
        all_local = (all_points - center) @ world_to_aligned
        all_local[:, 1] -= y_med
        all_aligned = all_local + post_translation
        result["all_aligned"] = all_aligned

    return result


def uniform_downsample(pcd, colors, n_neighbors=10):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(pcd)
    distances, indices = nbrs.kneighbors(pcd)

    mask = np.arange(len(pcd)) < indices[:, 1]
    downsampled_pcd = pcd[mask]
    downsampled_colors = colors[mask]

    return downsampled_pcd, downsampled_colors


def remove_outliers(pcd, colors, nb_neighbors=10, std_ratio=2.0):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=nb_neighbors).fit(pcd)
    distances, _ = nbrs.kneighbors(pcd)

    mean_distances = distances.mean(axis=1)
    threshold = mean_distances.mean() + std_ratio * mean_distances.std()

    mask = mean_distances < threshold
    filtered_pcd = pcd[mask]
    filtered_colors = colors[mask]

    return filtered_pcd, filtered_colors


def extract_by_color(pcd, colors, color, threshold=30):
    if color == 'red':
        color_idx = 0
    elif color == 'green':
        color_idx = 1
    elif color == 'blue':
        color_idx = 2
    else:
        raise ValueError("Unsupported color: choose from 'red', 'green', 'blue'")

    inverse_color_idx = (color_idx + 1) % 3, (color_idx + 2) % 3

    # extract points with the specified color
    thres_color = np.maximum(colors.astype(np.int32) - threshold, 0)

    mask = (thres_color[:, inverse_color_idx] == 0).all(axis=1)

    if mask.sum() == 0:
        print(f"No points found for color {color} with threshold {threshold}")
        return np.empty((0, 3))

    extracted_points = pcd[mask]
    extracted_colors = colors[mask]

    return extracted_points, extracted_colors


def get_height_width_from_aligned_pcd(aligned_pcd):
    height = aligned_pcd[:, 2].max() - aligned_pcd[:, 2].min()
    width = aligned_pcd[:, 0].max() - aligned_pcd[:, 0].min()
    depth = aligned_pcd[:, 1].max() - aligned_pcd[:, 1].min()
    return height, width, depth


def get_real_coordinates_from_aligned(target_height, target_width, target_depth, base_height, real_base_height):
    scale_factor = real_base_height / base_height
    real_target_height = target_height * scale_factor
    real_target_width = target_width * scale_factor
    real_target_depth = target_depth * scale_factor
    return real_target_height, real_target_width, real_target_depth


def plot_pcd(pcd, colors, post_pcd=None, post_colors=None):
    fig = plt.figure(figsize=(10, 10))
    if post_pcd is not None and post_colors is not None:
        ax1_min = pcd.min() * 1.1; ax1__max = pcd.max() * 1.1
        ax2_min = post_pcd.min() * 1.1; ax2_max = post_pcd.max() * 1.1
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=colors/255, s=1)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(post_pcd[:, 0], post_pcd[:, 1], post_pcd[:, 2], c=post_colors/255, s=1)
        ax1.set_xlim(ax1_min, ax1__max); ax1.set_ylim(ax1_min, ax1__max); ax1.set_zlim(ax1_min, ax1__max)
        ax2.set_xlim(ax2_min, ax2_max); ax2.set_ylim(ax2_min, ax2_max); ax2.set_zlim(ax2_min, ax2_max)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=colors/255, s=1)
        ax_min = pcd.min() * 1.1; ax_max = pcd.max() * 1.1
        ax.set_xlim(ax_min, ax_max); ax.set_ylim(ax_min, ax_max); ax.set_zlim(ax_min, ax_max)
    plt.show(block=True)
    plt.close()


if __name__ == "__main__":
    glb_path = "./SLAM3R/recon_multi.glb"

    do_downsample = True
    n_for_downsample = 10
    do_remove_outliers = True
    n_for_remove_outliers = 50
    std = 2.0

    plot_fig = True


    mesh = trimesh.load_scene(glb_path).dump()
    pcd = np.array(mesh[0].vertices)
    colors = np.array(mesh[0].colors[:, :3])

    red_pcd, red_colors = extract_by_color(pcd, colors, color='red', threshold=30)
    # green_pcd = extract_by_color(pcd, colors, color='green', threshold=30)
    # blue_pcd = extract_by_color(pcd, colors, color='blue', threshold=30)

    red_result = align_red_rectangle_to_xz(red_pcd, all_points=pcd)

    red_aligned = red_result["base_aligned"]
    all_aligned = red_result["all_aligned"]

    if do_downsample:
        downsample_pcd, downsample_colors = uniform_downsample(all_aligned, colors, n_neighbors=n_for_downsample)
        if plot_fig:
            plot_pcd(all_aligned, colors, post_pcd=downsample_pcd, post_colors=downsample_colors)
        all_aligned = downsample_pcd
        colors = downsample_colors

    green_aligned_pcd, green_aligned_colors = extract_by_color(all_aligned, colors, color='green', threshold=30)
    blue_aligned_pcd, blue_aligned_colors = extract_by_color(all_aligned, colors, color='blue', threshold=30)

    if do_remove_outliers:
        post_green_aligned_pcd, post_green_aligned_colors = remove_outliers(green_aligned_pcd, green_aligned_colors, nb_neighbors=n_for_remove_outliers, std_ratio=std)
        post_blue_aligned_pcd, post_blue_aligne_colors = remove_outliers(blue_aligned_pcd, blue_aligned_colors, nb_neighbors=n_for_remove_outliers, std_ratio=std)
        if plot_fig:
            plot_pcd(green_aligned_pcd, green_aligned_colors, post_pcd=post_green_aligned_pcd, post_colors=post_green_aligned_colors)
            plot_pcd(blue_aligned_pcd, blue_aligned_colors, post_pcd=post_blue_aligned_pcd, post_colors=post_blue_aligne_colors)
        green_aligned_pcd = post_green_aligned_pcd
        green_aligned_colors = post_green_aligned_colors
        blue_aligned_pcd = post_blue_aligned_pcd
        blue_aligned_colors = post_blue_aligne_colors

    green_pcd_height, green_pcd_width, green_pcd_depth = get_height_width_from_aligned_pcd(green_aligned_pcd)
    blue_pcd_height, blue_pcd_width, blue_pcd_depth = get_height_width_from_aligned_pcd(blue_aligned_pcd)

    real_green_height, real_green_width, real_green_depth = get_real_coordinates_from_aligned(green_pcd_height, green_pcd_width, green_pcd_depth, red_result["height"], 2.1)
    real_blue_height, real_blue_width, real_blue_depth = get_real_coordinates_from_aligned(blue_pcd_height, blue_pcd_width, blue_pcd_depth, red_result["height"], 2.1)

    print(f'Green Object - Height: {real_green_height:.2f}m, Width: {real_green_width:.2f}m, Depth: {real_green_depth:.2f}m')
    print(f'Blue Object - Height: {real_blue_height:.2f}m, Width: {real_blue_width:.2f}m, Depth: {real_blue_depth:.2f}m')
