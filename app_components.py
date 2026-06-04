import pandas as pd
import gradio
import copy
import numpy as np
import os
import subprocess
import re
import open3d as o3d
import torch.cuda
import trimesh
from dataclasses import dataclass
from PIL import Image, ImageOps
from tqdm import tqdm
import torch
import shutil

palette_rgb = np.array([
    [163, 81, 251], [64, 222, 138], [255, 64, 64],
    [255, 161, 160], [255, 118, 51], [255, 182, 51], [209, 212, 53], [27, 150, 64]
    , [76, 251, 18], [148, 207, 26], [0, 214, 193], [46, 156, 170], [0, 196, 255], [54, 71, 151], [102, 117, 255],
    [0, 25, 239], [134, 58, 255], [83, 0, 135], [205, 58, 255], [255, 151, 202], [255, 57, 201]], dtype=np.uint8)


@dataclass
class OctreeResult:
    volume_unit: float or None
    volume_original: float or None
    save_path: str or None


@dataclass
class SegmentedPCDResult:
    aligned_pcd: np.ndarray or None
    aligned_colors: np.ndarray or None
    height: float or None
    width: float or None
    depth: float or None
    volume: float or None
    save_path: str or None


CUSTOM_COLOR = {
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "gray": [127, 127, 127],
}

CUSTOM_CATEGORIES = [
    {"color": CUSTOM_COLOR['gray'], "id": 0, "isthing": 1, "name": "door"},
    {"color": CUSTOM_COLOR['gray'], "id": 1, "isthing": 1, "name": "chair"},
    {"color": CUSTOM_COLOR['gray'], "id": 2, "isthing": 1, "name": "table"},
    {"color": CUSTOM_COLOR['gray'], "id": 3, "isthing": 1, "name": "cabinet"},
    {"color": CUSTOM_COLOR['gray'], "id": 4, "isthing": 0, "name": "floor"},
    {"color": CUSTOM_COLOR['gray'], "id": 5, "isthing": 0, "name": "wall"},
    {"color": CUSTOM_COLOR['gray'], "id": 6, "isthing": 1, "name": "refrigerator"},
    {"color": CUSTOM_COLOR['gray'], "id": 7, "isthing": 1, "name": "sofa"},
    {"color": CUSTOM_COLOR['gray'], "id": 8, "isthing": 1, "name": "bed"},
    {"color": CUSTOM_COLOR['gray'], "id": 9, "isthing": 1, "name": "toilet"},
    {"color": CUSTOM_COLOR['gray'], "id": 10, "isthing": 1, "name": "tv"},
    {"color": CUSTOM_COLOR['gray'], "id": 11, "isthing": 1, "name": "laptop"},
    {"color": CUSTOM_COLOR['gray'], "id": 12, "isthing": 1, "name": "mouse"},
    {"color": CUSTOM_COLOR['gray'], "id": 13, "isthing": 1, "name": "remote"},
    {"color": CUSTOM_COLOR['gray'], "id": 14, "isthing": 1, "name": "keyboard"},
    {"color": CUSTOM_COLOR['gray'], "id": 15, "isthing": 1, "name": "cell phone"},
    {"color": CUSTOM_COLOR['gray'], "id": 16, "isthing": 1, "name": "book"},
    {"color": CUSTOM_COLOR['gray'], "id": 17, "isthing": 1, "name": "clock"},
    {"color": CUSTOM_COLOR['gray'], "id": 18, "isthing": 1, "name": "vase"},
    {"color": CUSTOM_COLOR['gray'], "id": 19, "isthing": 1, "name": "scissors"},
    {"color": CUSTOM_COLOR['gray'], "id": 20, "isthing": 1, "name": "vacuum cleaner"},
    {"color": CUSTOM_COLOR['gray'], "id": 21, "isthing": 1, "name": "fire hydrant"},
    {"color": CUSTOM_COLOR['gray'], "id": 22, "isthing": 1, "name": "power distribution box"},
    {"color": CUSTOM_COLOR['gray'], "id": 23, "isthing": 1, "name": "sink"},
    {"color": CUSTOM_COLOR['gray'], "id": 24, "isthing": 1, "name": "paper box"},
    {"color": CUSTOM_COLOR['gray'], "id": 25, "isthing": 1, "name": "mirror"},
    {"color": CUSTOM_COLOR['gray'], "id": 26, "isthing": 1, "name": "information board"},
    {"color": CUSTOM_COLOR['gray'], "id": 27, "isthing": 1, "name": "microwave"},
    {"color": CUSTOM_COLOR['gray'], "id": 28, "isthing": 1, "name": "air conditioner"},
    {"color": CUSTOM_COLOR['gray'], "id": 29, "isthing": 1, "name": "window"},
]


COCO_CATEGORIES = [
    {'color': CUSTOM_COLOR['gray'], 'id': 30, 'isthing': 1, 'name': 'person'},
    {'color': CUSTOM_COLOR['gray'], 'id': 31, 'isthing': 1, 'name': 'bicycle'},
    {'color': CUSTOM_COLOR['gray'], 'id': 32, 'isthing': 1, 'name': 'car'},
    {'color': CUSTOM_COLOR['gray'], 'id': 33, 'isthing': 1, 'name': 'motorcycle'},
    {'color': CUSTOM_COLOR['gray'], 'id': 34, 'isthing': 1, 'name': 'airplane'},
    {'color': CUSTOM_COLOR['gray'], 'id': 35, 'isthing': 1, 'name': 'bus'},
    {'color': CUSTOM_COLOR['gray'], 'id': 36, 'isthing': 1, 'name': 'train'},
    {'color': CUSTOM_COLOR['gray'], 'id': 37, 'isthing': 1, 'name': 'truck'},
    {'color': CUSTOM_COLOR['gray'], 'id': 38, 'isthing': 1, 'name': 'boat'},
    {'color': CUSTOM_COLOR['gray'], 'id': 39, 'isthing': 1, 'name': 'traffic light'},
    {'color': CUSTOM_COLOR['gray'], 'id': 41, 'isthing': 1, 'name': 'stop sign'},
    {'color': CUSTOM_COLOR['gray'], 'id': 42, 'isthing': 1, 'name': 'parking meter'},
    {'color': CUSTOM_COLOR['gray'], 'id': 43, 'isthing': 1, 'name': 'bench'},
    {'color': CUSTOM_COLOR['gray'], 'id': 44, 'isthing': 1, 'name': 'bird'},
    {'color': CUSTOM_COLOR['gray'], 'id': 45, 'isthing': 1, 'name': 'cat'},
    {'color': CUSTOM_COLOR['gray'], 'id': 46, 'isthing': 1, 'name': 'dog'},
    {'color': CUSTOM_COLOR['gray'], 'id': 47, 'isthing': 1, 'name': 'horse'},
    {'color': CUSTOM_COLOR['gray'], 'id': 48, 'isthing': 1, 'name': 'sheep'},
    {'color': CUSTOM_COLOR['gray'], 'id': 49, 'isthing': 1, 'name': 'cow'},
    {'color': CUSTOM_COLOR['gray'], 'id': 50, 'isthing': 1, 'name': 'elephant'},
    {'color': CUSTOM_COLOR['gray'], 'id': 51, 'isthing': 1, 'name': 'bear'},
    {'color': CUSTOM_COLOR['gray'], 'id': 52, 'isthing': 1, 'name': 'zebra'},
    {'color': CUSTOM_COLOR['gray'], 'id': 53, 'isthing': 1, 'name': 'giraffe'},
    {'color': CUSTOM_COLOR['gray'], 'id': 54, 'isthing': 1, 'name': 'backpack'},
    {'color': CUSTOM_COLOR['gray'], 'id': 55, 'isthing': 1, 'name': 'umbrella'},
    {'color': CUSTOM_COLOR['gray'], 'id': 56, 'isthing': 1, 'name': 'handbag'},
    {'color': CUSTOM_COLOR['gray'], 'id': 57, 'isthing': 1, 'name': 'tie'},
    {'color': CUSTOM_COLOR['gray'], 'id': 58, 'isthing': 1, 'name': 'suitcase'},
    {'color': CUSTOM_COLOR['gray'], 'id': 59, 'isthing': 1, 'name': 'frisbee'},
    {'color': CUSTOM_COLOR['gray'], 'id': 60, 'isthing': 1, 'name': 'skis'},
    {'color': CUSTOM_COLOR['gray'], 'id': 61, 'isthing': 1, 'name': 'snowboard'},
    {'color': CUSTOM_COLOR['gray'], 'id': 62, 'isthing': 1, 'name': 'sports ball'},
    {'color': CUSTOM_COLOR['gray'], 'id': 63, 'isthing': 1, 'name': 'kite'},
    {'color': CUSTOM_COLOR['gray'], 'id': 64, 'isthing': 1, 'name': 'baseball bat'},
    {'color': CUSTOM_COLOR['gray'], 'id': 65, 'isthing': 1, 'name': 'baseball glove'},
    {'color': CUSTOM_COLOR['gray'], 'id': 66, 'isthing': 1, 'name': 'skateboard'},
    {'color': CUSTOM_COLOR['gray'], 'id': 67, 'isthing': 1, 'name': 'surfboard'},
    {'color': CUSTOM_COLOR['gray'], 'id': 68, 'isthing': 1, 'name': 'tennis racket'},
    {'color': CUSTOM_COLOR['gray'], 'id': 69, 'isthing': 1, 'name': 'bottle'},
    {'color': CUSTOM_COLOR['gray'], 'id': 70, 'isthing': 1, 'name': 'wine glass'},
    {'color': CUSTOM_COLOR['gray'], 'id': 71, 'isthing': 1, 'name': 'cup'},
    {'color': CUSTOM_COLOR['gray'], 'id': 72, 'isthing': 1, 'name': 'fork'},
    {'color': CUSTOM_COLOR['gray'], 'id': 73, 'isthing': 1, 'name': 'knife'},
    {'color': CUSTOM_COLOR['gray'], 'id': 74, 'isthing': 1, 'name': 'spoon'},
    {'color': CUSTOM_COLOR['gray'], 'id': 75, 'isthing': 1, 'name': 'bowl'},
    {'color': CUSTOM_COLOR['gray'], 'id': 76, 'isthing': 1, 'name': 'banana'},
    {'color': CUSTOM_COLOR['gray'], 'id': 77, 'isthing': 1, 'name': 'apple'},
    {'color': CUSTOM_COLOR['gray'], 'id': 78, 'isthing': 1, 'name': 'sandwich'},
    {'color': CUSTOM_COLOR['gray'], 'id': 79, 'isthing': 1, 'name': 'orange'},
    {'color': CUSTOM_COLOR['gray'], 'id': 80, 'isthing': 1, 'name': 'broccoli'},
    {'color': CUSTOM_COLOR['gray'], 'id': 81, 'isthing': 1, 'name': 'carrot'},
    {'color': CUSTOM_COLOR['gray'], 'id': 82, 'isthing': 1, 'name': 'hot dog'},
    {'color': CUSTOM_COLOR['gray'], 'id': 83, 'isthing': 1, 'name': 'pizza'},
    {'color': CUSTOM_COLOR['gray'], 'id': 84, 'isthing': 1, 'name': 'donut'},
    {'color': CUSTOM_COLOR['gray'], 'id': 85, 'isthing': 1, 'name': 'cake'},
    {'color': CUSTOM_COLOR['gray'], 'id': 87, 'isthing': 1, 'name': 'couch'},
    {'color': CUSTOM_COLOR['gray'], 'id': 88, 'isthing': 1, 'name': 'potted plant'},
    {'color': CUSTOM_COLOR['gray'], 'id': 90, 'isthing': 1, 'name': 'dining table'},
    {'color': CUSTOM_COLOR['gray'], 'id': 99, 'isthing': 1, 'name': 'oven'},
    {'color': CUSTOM_COLOR['gray'], 'id': 100, 'isthing': 1, 'name': 'toaster'},
    {'color': CUSTOM_COLOR['gray'], 'id': 107, 'isthing': 1, 'name': 'teddy bear'},
    {'color': CUSTOM_COLOR['gray'], 'id': 108, 'isthing': 1, 'name': 'hair drier'},
    {'color': CUSTOM_COLOR['gray'], 'id': 109, 'isthing': 1, 'name': 'toothbrush'},
    {'color': CUSTOM_COLOR['gray'], 'id': 110, 'isthing': 0, 'name': 'banner'},
    {'color': CUSTOM_COLOR['gray'], 'id': 111, 'isthing': 0, 'name': 'blanket'},
    {'color': CUSTOM_COLOR['gray'], 'id': 112, 'isthing': 0, 'name': 'bridge'},
    {'color': CUSTOM_COLOR['gray'], 'id': 113, 'isthing': 0, 'name': 'cardboard'},
    {'color': CUSTOM_COLOR['gray'], 'id': 114, 'isthing': 0, 'name': 'counter'},
    {'color': CUSTOM_COLOR['gray'], 'id': 115, 'isthing': 0, 'name': 'curtain'},
    {'color': CUSTOM_COLOR['gray'], 'id': 116, 'isthing': 0, 'name': 'flower'},
    {'color': CUSTOM_COLOR['gray'], 'id': 117, 'isthing': 0, 'name': 'fruit'},
    {'color': CUSTOM_COLOR['gray'], 'id': 118, 'isthing': 0, 'name': 'gravel'},
    {'color': CUSTOM_COLOR['gray'], 'id': 119, 'isthing': 0, 'name': 'house'},
    {'color': CUSTOM_COLOR['gray'], 'id': 120, 'isthing': 0, 'name': 'light'},
    {'color': CUSTOM_COLOR['gray'], 'id': 121, 'isthing': 0, 'name': 'mirror-stuff'},
    {'color': CUSTOM_COLOR['gray'], 'id': 122, 'isthing': 0, 'name': 'net'},
    {'color': CUSTOM_COLOR['gray'], 'id': 123, 'isthing': 0, 'name': 'pillow'},
    {'color': CUSTOM_COLOR['gray'], 'id': 124, 'isthing': 0, 'name': 'platform'},
    {'color': CUSTOM_COLOR['gray'], 'id': 125, 'isthing': 0, 'name': 'playingfield'},
    {'color': CUSTOM_COLOR['gray'], 'id': 126, 'isthing': 0, 'name': 'railroad'},
    {'color': CUSTOM_COLOR['gray'], 'id': 127, 'isthing': 0, 'name': 'river'},
    {'color': CUSTOM_COLOR['gray'], 'id': 128, 'isthing': 0, 'name': 'road'},
    {'color': CUSTOM_COLOR['gray'], 'id': 129, 'isthing': 0, 'name': 'roof'},
    {'color': CUSTOM_COLOR['gray'], 'id': 130, 'isthing': 0, 'name': 'sand'},
    {'color': CUSTOM_COLOR['gray'], 'id': 131, 'isthing': 0, 'name': 'sea'},
    {'color': CUSTOM_COLOR['gray'], 'id': 132, 'isthing': 0, 'name': 'shelf'},
    {'color': CUSTOM_COLOR['gray'], 'id': 133, 'isthing': 0, 'name': 'snow'},
    {'color': CUSTOM_COLOR['gray'], 'id': 134, 'isthing': 0, 'name': 'stairs'},
    {'color': CUSTOM_COLOR['gray'], 'id': 135, 'isthing': 0, 'name': 'tent'},
    {'color': CUSTOM_COLOR['gray'], 'id': 136, 'isthing': 0, 'name': 'towel'},
    {'color': CUSTOM_COLOR['gray'], 'id': 137, 'isthing': 0, 'name': 'water-other'},
    {'color': CUSTOM_COLOR['gray'], 'id': 138, 'isthing': 0, 'name': 'window-blind'},
    {'color': CUSTOM_COLOR['gray'], 'id': 139, 'isthing': 0, 'name': 'window-other'},
    {'color': CUSTOM_COLOR['gray'], 'id': 140, 'isthing': 0, 'name': 'tree-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 141, 'isthing': 0, 'name': 'fence-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 142, 'isthing': 0, 'name': 'ceiling-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 143, 'isthing': 0, 'name': 'sky-other-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 144, 'isthing': 0, 'name': 'pavement-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 145, 'isthing': 0, 'name': 'mountain-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 146, 'isthing': 0, 'name': 'grass-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 147, 'isthing': 0, 'name': 'dirt-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 148, 'isthing': 0, 'name': 'food-other-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 149, 'isthing': 0, 'name': 'building-other-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 150, 'isthing': 0, 'name': 'rock-merged'},
    {'color': CUSTOM_COLOR['gray'], 'id': 151, 'isthing': 0, 'name': 'rug-merged'}
]


CUSTOM_SEGMENTS_INFO = CUSTOM_CATEGORIES + COCO_CATEGORIES

for i, item in enumerate(CUSTOM_SEGMENTS_INFO):
    item["id"] = i
# --------------------------------------
# Data & UI management functions
# --------------------------------------

# list of file path ( not sorted )
# * sorted file path list
def sort_file_list(file_list):
    def key_func(path):
        # seperate character / number -> natural sort
        name = os.path.basename(path)
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]

    # * sorted file path list
    return sorted(file_list, key=key_func)

# segementation category dictionary list
# * object name data -> pandas data frame
def info_to_df(segment_info):
    # * object name data -> pandas data frame
    return pd.DataFrame({"name": [x["name"] for x in segment_info]})

# Data from Gradio table component
# * valid object name list ( not void, not duplicate )
def extract_names_from_table(table_value):
    if table_value is None:
        names = []
    elif isinstance(table_value, pd.DataFrame):
        names = table_value.iloc[:, 0].tolist()
    else:
        names = [row[0] for row in table_value if row]

    # check name by recurrent table data
    cleaned = []
    for x in names:
        if x is None:
            continue
        x = str(x).strip()
        if not x:
            continue
        if x not in cleaned:
            cleaned.append(x)

    # * valid object name list ( not void, not duplicate )
    return cleaned

# present segmentation category name , new object name list from GRADIO
# * updated instance segmentation categroy list
def rebuild_segment_info(names, prev_segment_info):

    # update present name list refer to new object list
    prev_map = {item["name"]: copy.deepcopy(item) for item in prev_segment_info}
    new_info = []
    for name in names:
        if name in prev_map:
            item = copy.deepcopy(prev_map[name])
        else:
            item = {
                "color": CUSTOM_COLOR["gray"],
                "id": 0,
                "isthing": 1,
                "name": name,
            }
        new_info.append(item)

    # * updated instance segmentation categroy list
    return new_info

# updated table data, selected target 1,2 , present category info
# * updated table dataframe, drop down 1,2 , updated category info , text prompts will send to model
def sync_ui(table_value, selected_target1, selected_target2, segment_info_state):
    # when table data change -> reconstruct categroy info -> drop down menu update
    names = extract_names_from_table(table_value)
    updated_info = rebuild_segment_info(names, segment_info_state)
    df = info_to_df(updated_info)

    target_choices = [x["name"] for x in updated_info]
    if selected_target1 not in target_choices:
        selected_target1 = target_choices[0] if target_choices else None

    valid_target2 = [x for x in target_choices if x != selected_target1]
    if selected_target2 not in target_choices:
        selected_target2 = valid_target2[0] if valid_target2 else selected_target1

    changed_prompt = 'sem: ' + ', '.join(x["name"] for x in updated_info)

    # * updated table dataframe, drop down 1,2 , updated category info , text prompts will send to model
    return (
        df,
        gradio.update(choices=target_choices, value=selected_target1),
        gradio.update(choices=target_choices, value=selected_target2),
        updated_info,
        gradio.update(value=changed_prompt)
    )

# present table data, new text by user, present categroy info
# * table data frame add data, updated drop down componets, newest category info, initialized text box
def add_segment_row(table_value, tbox, segment_info_state):
    names = extract_names_from_table(table_value)

    # when user input new object name in text box -> test validation -> if not dupicated -> add new object info in category
    if tbox is not None and not re.match(r'^[a-zA-Z0-9\s]+$', tbox.strip()):
        return (
            gradio.update(),
            gradio.update(),
            gradio.update(),
            segment_info_state,
            gradio.update(),
            gradio.update(value="")
        )
    elif tbox is not None and str(tbox).strip() != "" and tbox.strip() not in names:
        names.append(tbox.strip())
        updated_info = rebuild_segment_info(names, segment_info_state)
        df = info_to_df(updated_info)

        target_choices = [x["name"] for x in updated_info]

        changed_prompt = 'sem: ' + ', '.join(x["name"] for x in updated_info)


        return (
            df,
            gradio.update(choices=target_choices),
            gradio.update(choices=target_choices),
            updated_info,
            gradio.update(value=changed_prompt),
            gradio.update(value="")
        )
    else:
        return (
            gradio.update(),
            gradio.update(),
            gradio.update(),
            segment_info_state,
            gradio.update(),
            gradio.update(value="")
        )

# base object , target 1, 2
# * if dupication occur -> not available button + error message / else ( normal ) -> activate button
def validate_segment_target(selected_base, selected_target1, selected_target2):
    t0 = str(selected_base).strip()
    t1 = str(selected_target1).strip()
    t2 = str(selected_target2).strip()

    # check base and target 1,2 is duplicated
    same = t0 == t1 or t0 == t2 or t1 == t2

    if same:
        warning_html = """
        <div style="color:#ef4444; font-weight:600; margin-top:4px;">
            Warning: Base Target and Main Target 1, 2 must be different.
        </div>
        """
    else:
        warning_html = "<div style='min-height: 20px;'></div>"

    # * if dupication occur -> not available button + error message / else ( normal ) -> activate button
    return (gradio.update(interactive=not same), gradio.update(value=warning_html))

# reize image and rotation control
def preprocess_image(img_path, save_path=None, target_size=(1086, 1448)):

    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(target_size, Image.Resampling.LANCZOS)

            if save_path:
                img.save(save_path, quality=95)
            return img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# --------------------------------------
# 3D pcd process & sorting
# --------------------------------------

# pcd, color , reference color
# * hsv quantize color mapping <-new add
def map_pcd_colors_by_hsv(points, colors_rgb, reference_colors_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)

    # sor, dbscan refinement
    pcd_sor, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=4.0)
    labels_global = np.array(pcd_sor.cluster_dbscan(eps=0.05, min_points=15, print_progress=False))
    valid_mask = labels_global >= 0
    pcd_clean = pcd_sor.select_by_index(np.where(valid_mask)[0].tolist())

    # hsv mapping
    points_rgb = np.asarray(pcd_clean.colors)
    ref_colors_norm = reference_colors_rgb / 255.0

    import matplotlib.colors as mcolors
    points_hsv = mcolors.rgb_to_hsv(points_rgb)
    ref_hsv = mcolors.rgb_to_hsv(ref_colors_norm)

    SATURATION_THRESHOLD = 0.15
    gray_bg_mask = points_hsv[:, 1] < SATURATION_THRESHOLD

    h_points = points_hsv[:, 0:1]
    h_refs = ref_hsv[:, 0:1].T
    h_diff = np.abs(h_points - h_refs)
    h_dist = np.minimum(h_diff, 1.0 - h_diff)
    s_points = points_hsv[:, 1:2]
    s_refs = ref_hsv[:, 1:2].T
    s_dist = np.abs(s_points - s_refs)

    total_dist = (h_dist * 1.0) + (s_dist * 1.0)
    min_distances = np.min(total_dist, axis=1)
    color_ids = np.argmin(total_dist, axis=1)

    COLOR_THRESHOLD = 1.3
    color_ids[min_distances >= COLOR_THRESHOLD] = -1
    color_ids[gray_bg_mask] = -1

    # mapping color
    mapped_colors = np.zeros_like(points_rgb)
    for idx, ref_color in enumerate(ref_colors_norm):
        mapped_colors[color_ids == idx] = ref_color
    mapped_colors[color_ids == -1] = [0.9, 0.9, 0.9] # gray

    # 255 scale return
    return np.asarray(pcd_clean.points), mapped_colors * 255.0

# whole pcd posion, color, criterion color data distribution from 2d image segmenation, instance name will extract, statics std
# * extracted instance's 3D position, color
def extract_by_color(pcd, pcd_color, extract_color, instance_name, std_interval_multiplier=2):
    # extract_color -> pure rgb list by segmentation color  table
    target_color = np.array(extract_color)

    # pcd_color already mapping from table
    # masking exactly same as color table
    color_diff = np.linalg.norm(pcd_color - target_color, axis=1)
    mask = color_diff < 5.0

    if mask.sum() == 0:
        print(f"No points found for {instance_name} (Target RGB: {target_color})")
        return np.empty((0, 3)), np.empty((0, 3))

    extracted_points = pcd[mask]
    extracted_colors = pcd_color[mask]

    return extracted_points, extracted_colors

# axis aligned pcd
# * height, width ,depth in 3d
def get_height_width_from_aligned_pcd(aligned_pcd):
    # calculate bbox, using point's x,y,z
    width = aligned_pcd[:, 0].max() - aligned_pcd[:, 0].min()
    depth = aligned_pcd[:, 2].max() - aligned_pcd[:, 2].min()
    height = aligned_pcd[:, 1].max() - aligned_pcd[:, 1].min()
    return height, width, depth

# target instance's measurement in 3d ( height,width, depth), base instance's height in 3d, base instance's height in real
# * target instance's measurement in real world
def get_real_coordinates_from_aligned(target_height, target_width, target_depth, base_height, real_base_height):
    scale_factor = real_base_height / base_height
    real_target_height = target_height * scale_factor
    real_target_width = target_width * scale_factor
    real_target_depth = target_depth * scale_factor

    # * target instance's measurement in real world
    return real_target_height, real_target_width, real_target_depth

# ! base instance pcd, whole pcd
# * aligned base instance, aligned scene pcd, transform matrix, scale ...
def align_pcd_by_rectangle(base_points, all_points=None):
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

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    world_front = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.dot(long_axis, world_up) < 0:
        long_axis = -long_axis

    if np.dot(normal_axis, world_front) < 0:
        normal_axis = -normal_axis
        short_axis = -short_axis

    world_to_aligned = np.stack([short_axis, long_axis, normal_axis], axis=1)

    aligned_to_world = world_to_aligned.T

    base_local = (base_points - center) @ world_to_aligned

    z_med = np.median(base_local[:, 2])
    base_local[:, 2] -= z_med

    min_x, max_x = base_local[:, 0].min(), base_local[:, 0].max()
    min_y, max_y = base_local[:, 1].min(), base_local[:, 1].max()

    width = max_x - min_x
    height = max_y - min_y

    post_translation = np.array([
        -min_x,
        -min_y,
        0.0,
    ], dtype=np.float64)

    base_aligned = base_local + post_translation

    rectangle_corners_aligned = np.array([
        [0.0,   0.0,    0.0],
        [width, 0.0,    0.0],
        [width, height, 0.0],
        [0.0,   height, 0.0],
    ], dtype=np.float64)

    rectangle_corners_world = (
        (rectangle_corners_aligned - post_translation + np.array([0.0, 0.0, z_med]))
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
        all_local[:, 2] -= z_med
        all_aligned = all_local + post_translation
        result["all_aligned"] = all_aligned

    # * aligned base instance, aligned scene pcd, transform matrix, scale ...
    return result


def extract_frames(video_path: str, fps: float, output_dir: str) -> str:
    frame_count = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", video_path]).decode().strip())
    max_index = (frame_count - 1) // fps
    width = len(str(max_index)) + 1

    save_dir = os.path.join(output_dir, "extracted_frames")
    if os.path.exists(save_dir):
        os.remove(save_dir)

    output_path = os.path.join(save_dir, f"frame_%0{width}d.jpg")
    os.makedirs(save_dir, exist_ok=True)
    command = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", output_path]
    subprocess.run(command, check=True)
    return save_dir, width

# original scene pcd
# * normalizaed pcd, original space's min/max boundary, scale ratio
def normalize_pcd_to_unit(pcd: np.ndarray):
    min_bound = pcd.min(axis=0)
    max_bound = pcd.max(axis=0)
    max_extent = (max_bound - min_bound).max()
    normalized_pcd = (pcd - min_bound) / max_extent
    return normalized_pcd, min_bound, max_bound, max_extent

# * dbscan
def pcd_clustering(pcd, colors, eps=0.02, min_points=15):
    # normalized_pcd, min_bound, max_bound, max_extent = normalize_pcd_to_unit(pcd)
    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(normalized_pcd)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.)
    # labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    # mask = labels != -1
    # clustered_pcd = normalized_pcd[mask]
    # clustered_colors = np.asarray(pcd_o3d.colors)[mask] * 255
    # restored_pcd = clustered_pcd * max_extent + min_bound

    if len(pcd) == 0:
        return pcd, colors

    normalized_pcd, min_bound, max_bound, max_extent = normalize_pcd_to_unit(pcd)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(normalized_pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.)

    # DBSCAN 클러스터링 수행
    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    max_label = labels.max()
    # 유효한 클러스터가 하나도 없는 경우 (-1 즉, 모두 노이즈인 경우)
    if max_label < 0:
        return np.empty((0, 3)), np.empty((0, 3))

    # [핵심 수정 부분] 노이즈(-1)를 제외한 유효 라벨 중에서 가장 포인트 개수가 많은 군집 찾기
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    largest_cluster_label = np.bincount(valid_labels).argmax()

    # 가장 큰 군집에 해당하는 포인트만 마스킹
    mask = labels == largest_cluster_label

    clustered_pcd = normalized_pcd[mask]
    clustered_colors = np.asarray(pcd_o3d.colors)[mask] * 255
    restored_pcd = clustered_pcd * max_extent + min_bound

    clean_pcd, clean_colors = pcd_statistical_outlier_removal(
        restored_pcd,
        clustered_colors,
        nb_neighbors=30,
        std_ratio=1.0
    )
    # return restored_pcd, clustered_colors
    return clean_pcd,clean_colors

# * sor
def pcd_statistical_outlier_removal(pcd, colors, nb_neighbors=20, std_ratio=2.0):
    if len(pcd) < nb_neighbors:
        return pcd, colors

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors / 255.)

    # remove point long distance away from over std by mean distance
    cl, ind = pcd_o3d.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    clean_pcd = np.asarray(cl.points)
    clean_colors = np.asarray(cl.colors) * 255.
    return clean_pcd, clean_colors

# scene pcd, color , dbscan parameter
# * indepent instance and color pair list
def extract_all_instances(pcd, colors, eps=0.032, min_points=10):
    normalized_pcd, min_bound, max_bound, max_extent = normalize_pcd_to_unit(pcd)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(normalized_pcd)
    labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    instances = []
    max_label = labels.max()
    if max_label < 0: return instances

    for i in range(max_label + 1):
        mask = labels == i
        if mask.sum() < min_points: continue
        inst_pcd = normalized_pcd[mask] * max_extent + min_bound
        inst_colors = colors[mask]
        instances.append((inst_pcd, inst_colors))

    # * indepent instance and color pair list
    return instances

# --------------------------------------
# 3D reconstruction pipeline(slam3r base)
# --------------------------------------

# whole scene pcd , base real height, base instance, target list...
# * aligned base, target pcd, measurement , save path ,... result structure
def process_pcd_to_segmented_pcd(pcd, colors, real_height, base_instance, target_instance_list, instance_info, std_interval_multiplier,
                                 do_clustering_before_alignment, do_clustering, clustering_eps, clustering_min_points, save_dir: str):
    from slam3r.utils.recon_utils import join

    # extract base object using color from scene
    base_pcd, base_colors = extract_by_color(pcd=pcd, pcd_color=colors, extract_color=instance_info[base_instance],
                                             instance_name=base_instance, std_interval_multiplier=std_interval_multiplier)

    target_pcd_result_list = {}
    # refinement pcd using sor
    if do_clustering_before_alignment:
        outlier_neighbor = 25
        outlier_std = 2.0
        base_pcd, base_colors = pcd_statistical_outlier_removal(base_pcd, base_colors, nb_neighbors=outlier_neighbor, std_ratio=outlier_std)

    if len(base_pcd) < 3:
        base_pcd_result = SegmentedPCDResult(aligned_pcd=None, aligned_colors=None, height=None, width=None, depth=None,
                                             volume=None, save_path=None)
        target_pcd_result_list = {
            target: SegmentedPCDResult(aligned_pcd=None, aligned_colors=None, height=None, width=None, depth=None,
                                       volume=None, save_path=None)
            for target in target_instance_list
        }
        return base_pcd_result, target_pcd_result_list, pcd

    # aligned pcd by base target
    aligned_result = align_pcd_by_rectangle(base_pcd, all_points=pcd)

    base_aligned_pcd = aligned_result["base_aligned"]

    if do_clustering:
        base_aligned_pcd, base_colors = pcd_clustering(base_aligned_pcd, base_colors, eps=clustering_eps, min_points=clustering_min_points)

    # calculate real target measurement from base height ratio
    if len(base_aligned_pcd) != 0:
        base_pcd_height, base_pcd_width, base_pcd_depth = get_height_width_from_aligned_pcd(base_aligned_pcd)
        real_base_height, real_base_width, real_base_depth = get_real_coordinates_from_aligned(base_pcd_height, base_pcd_width, base_pcd_depth,
                                                                                            base_height=aligned_result["height"], real_base_height=real_height)
        real_base_bbox_volume = real_base_height * real_base_width * real_base_depth
        base_scene = trimesh.Scene()
        base_aligned_pcd_save_path = join(save_dir, "base_aligned_pcd.glb")
        base_scene.add_geometry(trimesh.PointCloud(vertices=base_aligned_pcd, colors=base_colors / 255.))
        base_scene.export(base_aligned_pcd_save_path)
        base_pcd_result = SegmentedPCDResult(aligned_pcd=base_aligned_pcd, aligned_colors=base_colors,
                                            height=real_base_height, width=real_base_width, depth=real_base_depth,
                                            volume=real_base_bbox_volume, save_path=base_aligned_pcd_save_path)
    else:
        base_pcd_result = SegmentedPCDResult(aligned_pcd=None, aligned_colors=None, height=None, width=None, depth=None,
                                             volume=None, save_path=None)
        target_pcd_result_list = {
            target: SegmentedPCDResult(aligned_pcd=None, aligned_colors=None, height=None, width=None, depth=None,
                                       volume=None, save_path=None)
            for target in target_instance_list
        }
        return base_pcd_result, target_pcd_result_list, pcd

    for target_instance in target_instance_list:
        target_pcd, target_colors = extract_by_color(pcd=aligned_result["all_aligned"], pcd_color=colors, extract_color=instance_info[target_instance],
                                                     instance_name=target_instance, std_interval_multiplier=std_interval_multiplier)
        if do_clustering:
            target_pcd, target_colors = pcd_clustering(target_pcd, target_colors, eps=clustering_eps, min_points=clustering_min_points)

        if len(target_pcd) >= 3:
            target_pcd_height, target_pcd_width, target_pcd_depth = get_height_width_from_aligned_pcd(target_pcd)
            real_target_height, real_target_width, real_target_depth = get_real_coordinates_from_aligned(target_pcd_height, target_pcd_width, target_pcd_depth,
                                                                                                      base_height=aligned_result["height"], real_base_height=real_height)
            real_target_bbox_volume = real_target_height * real_target_width * real_target_depth
            target_scene = trimesh.Scene()
            target_aligned_pcd_save_path = join(save_dir, f"{target_instance}_aligned_pcd.glb")
            target_scene.add_geometry(trimesh.PointCloud(vertices=target_pcd, colors=target_colors / 255.))
            target_scene.export(target_aligned_pcd_save_path)
            target_pcd_result_list[target_instance] = SegmentedPCDResult(aligned_pcd=target_pcd, aligned_colors=target_colors,
                                                                         height=real_target_height, width=real_target_width, depth=real_target_depth,
                                                                         volume=real_target_bbox_volume, save_path=target_aligned_pcd_save_path)
        else:
            target_pcd_result_list[target_instance] = SegmentedPCDResult(aligned_pcd=None, aligned_colors=None, height=None, width=None,
                                                                         depth=None, volume=None, save_path=None)

    # * aligned base, target pcd, measurement , sava path ,... result structure
    return base_pcd_result, target_pcd_result_list, aligned_result["all_aligned"]

# model, device, img list, key frame and option ...
# * registered 3d pcd
def recon_scene(i2p_model, l2w_model, device, per_gpu, img_dir_or_list, keyframe_stride, win_r, initial_winsize, conf_thres_i2p, num_scene_frame, update_buffer_intv, buffer_strategy, buffer_size, save_dir,
                base_instance, segment_info, instance_info, num_points_save, conf_thres_l2w, real_height, std_interval_multiplier, do_clustering_before_alignment, do_clustering, clustering_eps, clustering_min_points):

    from slam3r.pipeline.recon_offline_pipeline import get_img_tokens, initialize_scene, adapt_keyframe_stride, i2p_inference_batch, l2w_inference, normalize_views, scene_frame_retrieve
    from slam3r.datasets.wild_seq import Seq_Data
    from slam3r.utils.recon_utils import transform_img, to_device

    np.random.seed(42)
    if per_gpu:
        torch.cuda.set_device(1)

    try:
        # extract file name list
        img_paths = []
        if isinstance(img_dir_or_list, (str, os.PathLike)) and os.path.isdir(img_dir_or_list):
            img_paths = [os.path.join(img_dir_or_list, f) for f in os.listdir(img_dir_or_list) if
                         os.path.isfile(os.path.join(img_dir_or_list, f))]
        elif isinstance(img_dir_or_list, dict):
            img_paths = [x for x in img_dir_or_list["file_name"]]
        elif isinstance(img_dir_or_list, list):
            img_paths = [f.name if hasattr(f, 'name') else f for f in img_dir_or_list]
        else:
            img_paths = img_dir_or_list

        # name sorting
        img_paths = sort_file_list(img_paths)

        # slam name parsing error block
        safe_slam_dir = os.path.join(save_dir, "safe_slam_inputs")
        os.makedirs(safe_slam_dir, exist_ok=True)

        safe_img_paths = []
        for idx, f_path in enumerate(img_paths):
            ext = os.path.splitext(f_path)[1]
            safe_name = f"frame_{idx:06d}{ext}"  # ex) frame_000000.png
            safe_path = os.path.join(safe_slam_dir, safe_name)

            preprocess_image(f_path, save_path=safe_path, target_size=(1086, 1448))
            safe_img_paths.append(safe_path)

        # safe file name to slma
        dataset = Seq_Data(safe_img_paths, to_tensor=True)
        data_views = dataset[0][:]
        num_views = len(data_views)

        # Pre-save the RGB images along with their corresponding masks
        # in preparation for visualization at last.
        rgb_imgs = []
        for i in range(len(data_views)):
            if data_views[i]['img'].shape[0] == 1:
                data_views[i]['img'] = data_views[i]['img'][0]
            rgb_imgs.append(transform_img(dict(img=data_views[i]['img'][None]))[..., ::-1])

        # preprocess data for extracting their img tokens with encoder
        for view in data_views:
            view['img'] = torch.tensor(view['img'][None])
            view['true_shape'] = torch.tensor(view['true_shape'][None])
            for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
                if key in view:
                    del view[key]
            to_device(view, device=device)
        # pre-extract img tokens by encoder, which can be reused
        # in the following inference by both i2p and l2w models
        res_shapes, res_feats, res_poses = get_img_tokens(data_views, i2p_model)  # 300+fps
        print('finish pre-extracting img tokens')

        # re-organize input views for the following inference.
        # Keep necessary attributes only.
        input_views = []
        for i in range(num_views):
            input_views.append(
                dict(label=data_views[i]['label'], img_tokens=res_feats[i], true_shape=data_views[i]['true_shape'], img_pos=res_poses[i]))

        # decide the stride of sampling keyframes, as well as other related parameters
        if keyframe_stride == -1:
            kf_stride = adapt_keyframe_stride(input_views, i2p_model, win_r=3, adapt_min=1, adapt_max=20, adapt_stride=1)
        else:
            kf_stride = keyframe_stride

        # initialize the scene with the first several frames
        initial_winsize = min(initial_winsize, num_views // kf_stride)
        assert initial_winsize >= 2, "not enough views for initializing the scene reconstruction"
        initial_pcds, initial_confs, init_ref_id = initialize_scene(input_views[:initial_winsize * kf_stride:kf_stride], i2p_model, winsize=initial_winsize, return_ref_id=True)  # 5*(1,224,224,3)

        # start reconstrution of the whole scene
        init_num = len(initial_pcds)
        per_frame_res = dict(i2p_pcds=[], i2p_confs=[], l2w_pcds=[], l2w_confs=[])
        for key in per_frame_res:
            per_frame_res[key] = [None for _ in range(num_views)]

        registered_confs_mean = [_ for _ in range(num_views)]

        # set up the world coordinates with the initial window
        for i in range(init_num):
            per_frame_res['l2w_confs'][i * kf_stride] = initial_confs[i][0].to(device)  # 224,224
            registered_confs_mean[i * kf_stride] = per_frame_res['l2w_confs'][i * kf_stride].mean().cpu()

        # initialize the buffering set with the initial window
        assert buffer_size <= 0 or buffer_size >= init_num
        buffering_set_ids = [i * kf_stride for i in range(init_num)]

        # set up the world coordinates with frames in the initial window
        for i in range(init_num):
            input_views[i * kf_stride]['pts3d_world'] = initial_pcds[i]

        initial_valid_masks = [conf > conf_thres_i2p for conf in initial_confs]  # 1,224,224
        normed_pts = normalize_views([view['pts3d_world'] for view in input_views[:init_num * kf_stride:kf_stride]],
                                     initial_valid_masks)
        for i in range(init_num):
            input_views[i * kf_stride]['pts3d_world'] = normed_pts[i]
            # filter out points with low confidence
            input_views[i * kf_stride]['pts3d_world'][~initial_valid_masks[i]] = 0
            per_frame_res['l2w_pcds'][i * kf_stride] = normed_pts[i]  # 224,224,3

        # recover the pointmap of each view in their local coordinates with the I2P model
        # TODO: batchify
        local_confs_mean = []
        adj_distance = kf_stride
        for view_id in tqdm(range(num_views), desc="I2P resonstruction"):
            # skip the views in the initial window
            if view_id in buffering_set_ids:
                # trick to mark the keyframe in the initial window
                if view_id // kf_stride == init_ref_id:
                    per_frame_res['i2p_pcds'][view_id] = per_frame_res['l2w_pcds'][view_id].cpu()
                else:
                    per_frame_res['i2p_pcds'][view_id] = torch.zeros_like(per_frame_res['l2w_pcds'][view_id],
                                                                          device="cpu")
                per_frame_res['i2p_confs'][view_id] = per_frame_res['l2w_confs'][view_id].cpu()
                continue
            # construct the local window
            sel_ids = [view_id]
            for i in range(1, win_r + 1):
                if view_id - i * adj_distance >= 0:
                    sel_ids.append(view_id - i * adj_distance)
                if view_id + i * adj_distance < num_views:
                    sel_ids.append(view_id + i * adj_distance)
            local_views = [input_views[id] for id in sel_ids]
            ref_id = 0
            # recover points in the local window, and save the keyframe points and confs
            output = i2p_inference_batch([local_views], i2p_model, ref_id=ref_id, tocpu=False, unsqueeze=False)['preds']
            # save results of the i2p model
            per_frame_res['i2p_pcds'][view_id] = output[ref_id]['pts3d'].cpu()  # 1,224,224,3
            per_frame_res['i2p_confs'][view_id] = output[ref_id]['conf'][0].cpu()  # 224,224

            # construct the input for L2W model
            input_views[view_id]['pts3d_cam'] = output[ref_id]['pts3d']  # 1,224,224,3
            valid_mask = output[ref_id]['conf'] > conf_thres_i2p  # 1,224,224
            input_views[view_id]['pts3d_cam'] = normalize_views([input_views[view_id]['pts3d_cam']], [valid_mask])[0]
            input_views[view_id]['pts3d_cam'][~valid_mask] = 0

        local_confs_mean = [conf.mean() for conf in per_frame_res['i2p_confs']]  # 224,224
        print(f'finish recovering pcds of {len(local_confs_mean)} frames in their local coordinates, with a mean confidence of {torch.stack(local_confs_mean).mean():.2f}')

        # Special treatment: register the frames within the range of initial window with L2W model
        # TODO: batchify
        if kf_stride > 1:
            max_conf_mean = -1
            for view_id in tqdm(range((init_num - 1) * kf_stride), desc="pre-registering"):
                if view_id % kf_stride == 0:
                    continue
                # construct the input for L2W model
                l2w_input_views = [input_views[view_id]] + [input_views[id] for id in buffering_set_ids]
                # (for defination of ref_ids, see the doc of l2w_model)
                output = l2w_inference(l2w_input_views, l2w_model, ref_ids=list(range(1, len(l2w_input_views))),
                                       device=device, normalize=False)

                # process the output of L2W model
                input_views[view_id]['pts3d_world'] = output[0]['pts3d_in_other_view']  # 1,224,224,3
                conf_map = output[0]['conf']  # 1,224,224
                per_frame_res['l2w_confs'][view_id] = conf_map[0]  # 224,224
                registered_confs_mean[view_id] = conf_map.mean().cpu()
                per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']

                if registered_confs_mean[view_id] > max_conf_mean:
                    max_conf_mean = registered_confs_mean[view_id]
            print(
                f'finish aligning {(init_num - 1) * kf_stride} head frames, with a max mean confidence of {max_conf_mean:.2f}')

            # A problem is that the registered_confs_mean of the initial window is generated by I2P model,
            # while the registered_confs_mean of the frames within the initial window is generated by L2W model,
            # so there exists a gap. Here we try to align it.
            max_initial_conf_mean = -1
            for i in range(init_num):
                if registered_confs_mean[i * kf_stride] > max_initial_conf_mean:
                    max_initial_conf_mean = registered_confs_mean[i * kf_stride]
            factor = max_conf_mean / max_initial_conf_mean
            # print(f'align register confidence with a factor {factor}')
            for i in range(init_num):
                per_frame_res['l2w_confs'][i * kf_stride] *= factor
                registered_confs_mean[i * kf_stride] = per_frame_res['l2w_confs'][i * kf_stride].mean().cpu()

        # register the rest frames with L2W model
        next_register_id = (init_num - 1) * kf_stride + 1  # the next frame to be registered
        milestone = (
                                init_num - 1) * kf_stride + 1  # All frames before milestone have undergone the selection process for entry into the buffering set.
        num_register = max(1, min((kf_stride + 1) // 2, 10))  # how many frames to register in each round
        # num_register = 1
        update_buffer_intv = kf_stride * update_buffer_intv  # update the buffering set every update_buffer_intv frames
        max_buffer_size = buffer_size
        strategy = buffer_strategy
        candi_frame_id = len(buffering_set_ids)  # used for the reservoir sampling strategy

        pbar = tqdm(total=num_views, desc="registering")
        pbar.update(next_register_id - 1)

        del i
        while next_register_id < num_views:
            ni = next_register_id
            max_id = min(ni + num_register, num_views) - 1  # the last frame to be registered in this round

            # select sccene frames in the buffering set to work as a global reference
            cand_ref_ids = buffering_set_ids
            ref_views, sel_pool_ids = scene_frame_retrieve([input_views[i] for i in cand_ref_ids],
                                                           input_views[ni:ni + num_register:2], i2p_model,
                                                           sel_num=num_scene_frame,
                                                           # cand_recon_confs=[per_frame_res['l2w_confs'][i] for i in cand_ref_ids],
                                                           depth=2)

            # register the source frames in the local coordinates to the world coordinates with L2W model
            l2w_input_views = ref_views + input_views[ni:max_id + 1]
            input_view_num = len(ref_views) + max_id - ni + 1
            assert input_view_num == len(l2w_input_views)

            output = l2w_inference(l2w_input_views, l2w_model, ref_ids=list(range(len(ref_views))), device=device,
                                   normalize=False)

            # process the output of L2W model
            src_ids_local = [id + len(ref_views) for id in
                             range(max_id - ni + 1)]  # the ids of src views in the local window
            src_ids_global = [id for id in range(ni, max_id + 1)]  # the ids of src views in the whole dataset
            succ_num = 0
            for id in range(len(src_ids_global)):
                output_id = src_ids_local[id]  # the id of the output in the output list
                view_id = src_ids_global[id]  # the id of the view in all views
                conf_map = output[output_id]['conf']  # 1,224,224
                input_views[view_id]['pts3d_world'] = output[output_id]['pts3d_in_other_view']  # 1,224,224,3
                per_frame_res['l2w_confs'][view_id] = conf_map[0]
                registered_confs_mean[view_id] = conf_map[0].mean().cpu()
                per_frame_res['l2w_pcds'][view_id] = input_views[view_id]['pts3d_world']
                succ_num += 1
            # TODO:refine scene frames together
            # for j in range(1, input_view_num):
            # views[i-j]['pts3d_world'] = output[input_view_num-1-j]['pts3d'].permute(0,3,1,2)

            next_register_id += succ_num
            pbar.update(succ_num)

            # update the buffering set
            if next_register_id - milestone >= update_buffer_intv:
                while (next_register_id - milestone >= kf_stride):
                    candi_frame_id += 1
                    full_flag = max_buffer_size > 0 and len(buffering_set_ids) >= max_buffer_size
                    insert_flag = (not full_flag) or ((strategy == 'fifo') or (
                                strategy == 'reservoir' and np.random.rand() < max_buffer_size / candi_frame_id))
                    if not insert_flag:
                        milestone += kf_stride
                        continue
                    # Use offest to ensure the selected view is not too close to the last selected view
                    # If the last selected view is 0,
                    # the next selected view should be at least kf_stride*3//4 frames away
                    start_ids_offset = max(0, buffering_set_ids[-1] + kf_stride * 3 // 4 - milestone)

                    # get the mean confidence of the candidate views
                    mean_cand_recon_confs = torch.stack(
                        [registered_confs_mean[i] for i in range(milestone + start_ids_offset, milestone + kf_stride)])
                    mean_cand_local_confs = torch.stack(
                        [local_confs_mean[i] for i in range(milestone + start_ids_offset, milestone + kf_stride)])
                    # normalize the confidence to [0,1], to avoid overconfidence
                    mean_cand_recon_confs = (mean_cand_recon_confs - 1) / mean_cand_recon_confs  # transform to sigmoid
                    mean_cand_local_confs = (mean_cand_local_confs - 1) / mean_cand_local_confs
                    # the final confidence is the product of the two kinds of confidences
                    mean_cand_confs = mean_cand_recon_confs * mean_cand_local_confs

                    most_conf_id = mean_cand_confs.argmax().item()
                    most_conf_id += start_ids_offset
                    id_to_buffer = milestone + most_conf_id
                    buffering_set_ids.append(id_to_buffer)
                    # print(f"add ref view {id_to_buffer}")
                    # since we have inserted a new frame, overflow must happen when full_flag is True
                    if full_flag:
                        if strategy == 'reservoir':
                            buffering_set_ids.pop(np.random.randint(max_buffer_size))
                        elif strategy == 'fifo':
                            buffering_set_ids.pop(0)
                    # print(next_register_id, buffering_set_ids)
                    milestone += kf_stride
            # transfer the data to cpu if it is not in the buffering set, to save gpu memory
            for i in range(next_register_id):
                to_device(input_views[i], device=device if i in buffering_set_ids else 'cpu')

        pbar.close()

        fail_view = {}
        for i, conf in enumerate(registered_confs_mean):
            if conf < 10:
                fail_view[i] = conf.item()
        print(f'mean confidence for whole scene reconstruction: {torch.tensor(registered_confs_mean).mean().item():.2f}')
        print(f"{len(fail_view)} views with low confidence: ", {key: round(fail_view[key], 2) for key in fail_view.keys()})

        per_frame_res['rgb_imgs'] = rgb_imgs

        scene_output = get_model_from_scene(per_frame_res=per_frame_res, save_dir=save_dir, num_points_save=num_points_save, conf_thres_res=conf_thres_l2w,
                                            segment_info=segment_info, instance_info=instance_info,
                                            base_instance=base_instance, std_interval_multiplier=std_interval_multiplier, real_height=real_height,
                                            do_clustering_before_alignment=do_clustering_before_alignment,
                                            do_clustering=do_clustering, clustering_eps=clustering_eps,
                                            clustering_min_points=clustering_min_points)

        return *scene_output, per_frame_res

    except Exception as e:
        print("Error during scene reconstruction:", str(e))
        scene_output = get_model_from_scene(per_frame_res=None, save_dir=save_dir, num_points_save=num_points_save, conf_thres_res=conf_thres_l2w,
                                            segment_info=segment_info, instance_info=instance_info,
                                            base_instance=base_instance, std_interval_multiplier=std_interval_multiplier, real_height=real_height,
                                            do_clustering_before_alignment=do_clustering_before_alignment,
                                            do_clustering=do_clustering, clustering_eps=clustering_eps,
                                            clustering_min_points=clustering_min_points)

        return *scene_output, None

# 3d pcd, base instance, segment info, instance_info
def get_model_from_scene(per_frame_res, save_dir, base_instance, segment_info, instance_info , num_points_save=200000, conf_thres_res=3, real_height=2.1, std_interval_multiplier=2, do_clustering_before_alignment=False, do_clustering=False, clustering_eps=0.02, clustering_min_points=10, valid_masks=None):
    from slam3r.utils.recon_utils import to_numpy, join

    if per_frame_res is not None:
        # collect the registered point clouds and rgb colors
        pcds = []
        rgbs = []
        pred_frame_num = len(per_frame_res['l2w_pcds'])
        registered_confs = per_frame_res['l2w_confs']
        registered_pcds = per_frame_res['l2w_pcds']
        rgb_imgs = per_frame_res['rgb_imgs']
        for i in range(pred_frame_num):
            registered_pcd = to_numpy(registered_pcds[i])
            if registered_pcd.shape[0] == 3:
                registered_pcd = registered_pcd.transpose(1, 2, 0)
            registered_pcd = registered_pcd.reshape(-1, 3)
            rgb = rgb_imgs[i].reshape(-1, 3)
            pcds.append(registered_pcd)
            rgbs.append(rgb)

        res_pcds = np.concatenate(pcds, axis=0)
        res_rgbs = np.concatenate(rgbs, axis=0)

        pts_count = len(res_pcds)
        valid_ids = np.arange(pts_count)

        # filter out points with gt valid masks
        if valid_masks is not None:
            valid_masks = np.stack(valid_masks, axis=0).reshape(
                -1)  # print('filter out ratio of points by gt valid masks:', 1.-valid_masks.astype(float).mean())
        else:
            valid_masks = np.ones(pts_count, dtype=bool)

        # filter out points with low confidence
        if registered_confs is not None:
            conf_masks = []
            for i in range(len(registered_confs)):
                conf = registered_confs[i]
                conf_mask = (conf > conf_thres_res).reshape(-1).cpu()
                conf_masks.append(conf_mask)
            conf_masks = np.array(torch.cat(conf_masks))
            valid_ids = valid_ids[conf_masks & valid_masks]
            print('ratio of points filered out: {:.2f}%'.format((1. - len(valid_ids) / pts_count) * 100))

        # sample from the resulting pcd consisting of all frames
        n_samples = min(num_points_save, len(valid_ids))
        print(f"resampling {n_samples} points from {len(valid_ids)} points")
        sampled_idx = np.random.choice(valid_ids, n_samples, replace=False)
        sampled_pts = res_pcds[sampled_idx]
        sampled_rgbs = res_rgbs[sampled_idx]

        sampled_pts[..., 1:] *= -1  # flip the axis for better visualization

        # * hsv mapping pcd
        sampled_pts, sampled_rgbs = map_pcd_colors_by_hsv(sampled_pts, sampled_rgbs, palette_rgb)

        # make target and base instance list
        target_instance_list = list(instance_info.keys())
        target_instance_list.remove(base_instance)
        base_pcd_result, target_pcd_result_list, all_aligned_pcd = process_pcd_to_segmented_pcd(
            pcd=sampled_pts, colors=sampled_rgbs, real_height=real_height, base_instance=base_instance, target_instance_list=target_instance_list, instance_info=instance_info, std_interval_multiplier=std_interval_multiplier,
            do_clustering_before_alignment=do_clustering_before_alignment, do_clustering=do_clustering, clustering_eps=clustering_eps, clustering_min_points=clustering_min_points, save_dir=save_dir)

        # world pcd aligned red door
        global_pcd_pts = all_aligned_pcd if all_aligned_pcd is not None else sampled_pts

        # global scene rendering
        save_name = f"recon.glb"
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.PointCloud(vertices=global_pcd_pts, colors=sampled_rgbs / 255.))
        save_path = join(save_dir, save_name)
        # scene.export(save_path)

        def add_bbox_to_scene(pcd_pts, outline_color, is_aligned=True):
            if pcd_pts is None or len(pcd_pts) == 0: return

            base_rgb = np.array(outline_color[:3], dtype=np.uint8)

            # already alingn coordinate
            min_b, max_b = pcd_pts.min(axis=0), pcd_pts.max(axis=0)
            extents = max_b - min_b
            center = (max_b + min_b) / 2.0

            radius = max(extents) * 0.005
            if radius < 0.002: radius = 0.002

            x1, y1, z1 = min_b
            x2, y2, z2 = max_b
            corners = [
                [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
                [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]
            ]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            edge_color = np.array([base_rgb[0], base_rgb[1], base_rgb[2], 255], dtype=np.uint8)

            cylinders = []
            for start_idx, end_idx in edges:
                segment = np.array([corners[start_idx], corners[end_idx]])
                if np.linalg.norm(segment[0] - segment[1]) > 1e-5:
                    try:
                        cyl = trimesh.creation.cylinder(radius=radius, segment=segment)
                        cyl.visual.face_colors = edge_color
                        cylinders.append(cyl)
                    except Exception:
                        continue

            # make bbox with 12 cylinders and add scene
            if cylinders:
                bbox_mesh = trimesh.util.concatenate(cylinders)
                scene.add_geometry(bbox_mesh)

            # vertex highlights
            vertex_pcd = trimesh.PointCloud(vertices=np.array(corners))
            vertex_color = np.array([255, 255, 0], dtype=np.uint8)  # target(yellow)
            if base_rgb[0] == 255 and base_rgb[1] == 0 and base_rgb[2] == 0:
                vertex_color = np.array([255, 0, 0], dtype=np.uint8)  # base(red)

            vertex_pcd.colors = np.tile(vertex_color, (len(corners), 1))
            scene.add_geometry(vertex_pcd)
        # bbox add scene
        if base_pcd_result.aligned_pcd is not None and base_pcd_result.aligned_colors is not None:
            # base object other color
            base_outline_color = np.array([255, 0, 0], dtype=np.uint8)
            add_bbox_to_scene(base_pcd_result.aligned_pcd, base_outline_color)

        for target_instance in target_instance_list:
            target_res = target_pcd_result_list[target_instance]
            if target_res.aligned_pcd is not None and target_res.aligned_colors is not None:
                # target yellow
                target_outline_color = np.array([255, 255, 0], dtype=np.uint8)
                add_bbox_to_scene(target_res.aligned_pcd, target_outline_color)
        scene.export(save_path)
        target_list_radio = gradio.Radio(choices=target_instance_list, value=target_instance_list[0], interactive=True)
        target_pcd_result = target_pcd_result_list[target_instance_list[0]]

        return (
            gradio.update(value=save_path, interactive=True),
            gradio.update(value=base_pcd_result.save_path, interactive=True),
            gradio.update(value=f'<div style="min-height: 10px; text-align: center;">{base_instance}</div>'),
            gradio.update(value=base_pcd_result.height),
            gradio.update(value=target_pcd_result.save_path, interactive=True),
            gradio.update(value=f'<div style="min-height: 10px; text-align: center;">{target_instance_list[0]}</div>'),
            gradio.update(value=target_pcd_result.height),
            gradio.update(value=target_pcd_result.width),
            gradio.update(value=target_pcd_result.depth),
            gradio.update(value=target_pcd_result.volume),
            target_list_radio,
            target_pcd_result_list,
        )

    else:
        return (
            gradio.update(value=None, interactive=False),
            gradio.update(value=None, interactive=False),
            gradio.update(value=f'<div style="min-height: 10px; text-align: center;">base</div>'),
            gradio.update(value=None, interactive=False),
            gradio.update(value=None, interactive=False),
            gradio.update(value=f'<div style="min-height: 10px; text-align: center;">target1</div>'),
            gradio.update(value=None),
            gradio.update(value=None),
            gradio.update(value=None),
            gradio.update(value=None),
            gradio.Radio(choices=[], value=None, interactive=False),
            None,
        )


def change_target_visualization(target_radio, target_pcd_result_list):
    if target_pcd_result_list is not None:
        target_pcd_result = target_pcd_result_list[target_radio]

        return (
            gradio.update(value=target_pcd_result.save_path, interactive=True),
            gradio.update(value=f'<div style="min-height: 10px; text-align: center;">{target_radio}</div>'),
            gradio.update(value=target_pcd_result.height),
            gradio.update(value=target_pcd_result.width),
            gradio.update(value=target_pcd_result.depth),
            gradio.update(value=target_pcd_result.volume),
        )
    else:
        return (
            gradio.update(value=None, interactive=False),
            gradio.update(value=f'<div style="min-height: 10px; text-align: center;">target</div>'),
            gradio.update(value=None),
            gradio.update(value=None),
            gradio.update(value=None),
            gradio.update(value=None),
        )


def display_inputs(images, input_type):
    img_label = "Click or use the left/right arrow keys to browse images"
    vid_label = "Uploaded Video"
    if images is None or len(images) == 0:
        if input_type == "directory":
            vis_img_gal = True
            vis_vid_gal = False
        elif input_type == "images":
            vis_img_gal = True
            vis_vid_gal = False
        elif input_type == "video":
            vis_img_gal = False
            vis_vid_gal = True
        return [
            gradio.update(label=img_label, value=None, visible=vis_img_gal, selected_index=0, scale=2, preview=True, height=300, ),
            gradio.update(label=vid_label, value=None, visible=vis_vid_gal, scale=2, height=300, ),
        ]

    if isinstance(images, str):
        file_path = images
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        if any(file_path.endswith(ext) for ext in video_extensions):
            return [
                gradio.update(label=img_label, value=None, visible=False, selected_index=0, scale=2, preview=True, height=300, ),
                gradio.update(label=vid_label, value=file_path, autoplay=True, visible=True, scale=2, height=300, ),
            ]
        else:
            images_list = os.listdir(file_path)
            return [
                gradio.update(label=img_label, value=images_list, visible=True, selected_index=0, scale=2, preview=True, height=300, ),
                gradio.update(label=vid_label, value=None, visible=False, scale=2, height=300, ),
            ]

    if isinstance(images, list):
        images = sort_file_list(images)
        return [
            gradio.update(label=img_label, value=images, visible=True, selected_index=0, scale=2, preview=True, height=300, ),
            gradio.update(label=vid_label, value=None, visible=False, scale=2, height=300, ),
        ]


def change_inputfile_type(input_type):
    file_count = "directory"
    label = "Select a directory containing images"
    file_types = ["directory"]
    vis_img_gal = True
    vis_vid_gal = False
    if input_type == "directory":
        file_count = "directory"
        label = "Select a directory containing images"
        file_types = ["directory"]
        vis_img_gal = True
        vis_vid_gal = False
    elif input_type == "images":
        file_count = "multiple"
        label = "Upload multiple images"
        file_types = ["image"]
        vis_img_gal = True
        vis_vid_gal = False
    elif input_type == "video":
        file_count = "single"
        label = "Upload a mp4 video"
        file_types = ["video"]
        vis_img_gal = False
        vis_vid_gal = True
    return [
        gradio.update(file_count=file_count, value=None, label=label, file_types=file_types),
        gradio.update(visible=vis_vid_gal),
        gradio.update(value=None, visible=vis_img_gal),
        gradio.update(value=None, visible=vis_vid_gal),
    ]


def change_kf_stride_type(kf_stride):
    max_kf_stride = -1
    inter = False
    value = -1
    if kf_stride == "auto":
        value = -1
        inter = False
        max_kf_stride = -1
    elif kf_stride == "manual setting":
        value = 1
        inter = True
        max_kf_stride = 10
    return gradio.update(value=value, minimum=value, maximum=max_kf_stride, interactive=inter)


def change_buffer_strategy(buffer_strategy):
    value = 100
    vis = True
    if buffer_strategy == "reservoir" or buffer_strategy == "fifo":
        value = 100
        vis = True
    elif buffer_strategy == "unbounded":
        value = 10000
        vis = False
    return gradio.update(value=value, visible=vis)


def segment_scene(XSam_model, sam2_predictor, per_gpu, image, fps, segment_info, vprompt_masks, score_thr, background_list, output_dir):
    if per_gpu:
        torch.cuda.set_device(0)

    try:
        if isinstance(image, list):
            image = [img.name if hasattr(img, 'name') else img for img in image]
        elif os.path.isdir(image):
            image = [os.path.join(image, img) for img in sorted(os.listdir(image)) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        elif isinstance(image, str):
            extracted_frame_dir, frame_width = extract_frames(image, fps, output_dir)
            image = [os.path.join(extracted_frame_dir, img) for img in sorted(os.listdir(extracted_frame_dir))]

        if vprompt_masks == "":
            vprompt_masks = None

        prompt = 'ins: ' + ', '.join([f"{info['name']}" for info in segment_info])
        # print(f"prompt: {prompt}\nid: {[info['id'] for info in segment_info]}\ncolor: {[info['color'] for info in segment_info]}")
        print(f"prompt: {prompt}\nid: {[info['id'] for info in segment_info]}\n")

        print("Instance Segmentation")
        input_frame_idx = 0
        img_path = image[input_frame_idx]
        output_path_per_file = os.path.join(output_dir, os.path.basename(img_path))
        # pil_image = Image.open(img_path)
        pil_image = preprocess_image(img_path, target_size=(1086, 1448))

        llm_input, llm_output, masked_image, pred_labels, saved_path, masks, masks_save_path, pred_ids = XSam_model.run_on_image(
            pil_image, prompt, task_name='genseg', segment_info=segment_info, vprompt_masks=vprompt_masks,
            threshold=score_thr, output_dir=output_path_per_file)
        print(f"llm_input: {llm_input}\nllm_output: {llm_output}")

        save_jpg_dir = os.path.join(output_dir, 'convert_to_jpg')

        if os.path.exists(save_jpg_dir):
            shutil.rmtree(save_jpg_dir)
        os.makedirs(save_jpg_dir, exist_ok=True)
        for idx, img in enumerate(image):
            # pil_image = Image.open(img)
            save_path = os.path.join(save_jpg_dir, f'{idx:08d}.jpg')
            preprocess_image(img, save_path=save_path, target_size=(1086, 1448))
            # pil_image.save(save_path)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2_predictor.init_state(save_jpg_dir)

            for i, mask in enumerate(masks):
                sam2_predictor.add_new_mask(inference_state=state, mask=mask, frame_idx=input_frame_idx, obj_id=i)

            video_segments = {}
            for out_frame_idx, out_object_ids, out_mask_logits in sam2_predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {out_object_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_object_id in enumerate(out_object_ids)}

        # Save masked images
        output_image_path = []
        save_track_dir = os.path.join(output_dir, 'tracked_masks')
        if os.path.exists(save_track_dir):
            shutil.rmtree(save_track_dir)
        os.makedirs(save_track_dir, exist_ok=True)
        import supervision as sv
        colors = sv.ColorPalette.DEFAULT.colors

        # ========================================================
        # for debuging label / origin color mapping
        print("\n" + "=" * 55)
        print("[DEBUG] Label to Original Color Mapping")
        print("-" * 55)
        for j, label in enumerate(pred_labels):
            if label in background_list:
                print(f"   - Label: {label.ljust(15)} | Original RGB: Background ('outside')")
            else:
                orig_color = colors[j % len(colors)]
                pure_rgb = [orig_color.r, orig_color.g, orig_color.b]
                print(f"   - Label: {label.ljust(15)} | Original RGB: {pure_rgb}")
        print("=" * 55 + "\n")
        # ========================================================
        for i in range(len(image)):
            if i not in video_segments:
                continue
            # pil_image = Image.open(image[i])
            pil_image = preprocess_image(image[i], target_size=(1086, 1448))
            inverse_mask = np.ones_like(pil_image)[:, :, 0]

            for obj_id, mask in video_segments[i].items():
                if obj_id >= len(pred_labels):
                    continue

                if pred_labels[obj_id] in background_list:
                    color = 'outside'
                else:
                    orig_color = colors[obj_id % len(colors)]
                    color = [orig_color.r, orig_color.g, orig_color.b]

                mask = mask.squeeze(0)
                if mask.sum() == 0:
                    continue

                pil_image, color_tone = XSam_model.image_scaling_with_mask(pil_image, mask, mode=color)
                inverse_mask = inverse_mask & (mask == 0.)

            pil_image, _ = XSam_model.image_scaling_with_mask(pil_image, inverse_mask, mode='outside')

            save_path = os.path.join(save_track_dir, f'whole_{i:08d}.png')
            pil_image.save(save_path)
            output_image_path.append(save_path)

        segmented_labels = []
        instance_info = {}
        for idx, label in enumerate(pred_labels):
            if label in background_list:
                continue
            instance_idx = 0
            while label + f'-{instance_idx}' in segmented_labels:
                instance_idx += 1
            segmented_labels.append(label + f'-{instance_idx}')

            # ========================================================
            # pure rgb list
            orig_color = colors[idx % len(colors)]
            pure_rgb = [orig_color.r, orig_color.g, orig_color.b]
            instance_info[label + f'-{instance_idx}'] = pure_rgb
            # ========================================================
        segmentation_list = video_segments
        default_dropdown_value = segmented_labels[0] if len(segmented_labels) > 0 else None

        return [
            gradio.update(value=output_image_path, interactive=True),
            gradio.update(value=[], interactive=False),
            gradio.update(value=output_image_path),
            gradio.update(value=[], interactive=False),
            [],
            [],
            segment_info,
            segmented_labels,
            segmentation_list,
            instance_info,
            gradio.Dropdown(choices=segmented_labels, value=default_dropdown_value, interactive=True,
                            label="Base Target"),
        ]

    except Exception as e:
        print(f"Error in segment_scene: {e}")
        return [
            gradio.update(value=[], interactive=False),
            gradio.update(value=[], interactive=False),
            gradio.update(value=[], interactive=False),
            gradio.update(value=[], interactive=False),
            [],
            [],
            segment_info,
            None,
            None,
            None,
            gradio.Dropdown(choices=None, value='', interactive=False, label="Base Target"),
        ]


def initialize_segment_outputs():
    return [
        gradio.update(value=[], interactive=False),
        gradio.update(value=[], interactive=False),
        gradio.update(value=[], interactive=False),
        gradio.update(value=[], interactive=False),
        [],
        [],
        False,
    ]


def on_output_files_changed(active_files, removed_files, update_state, prev_active, prev_removed):
    if update_state:
        return (
            gradio.update(),
            gradio.update(),
            gradio.update(),
            gradio.update(),
            active_files,
            removed_files,
            False,
        )
    else:
        prev_active = [] if prev_active is None else prev_active
        prev_removed = [] if prev_removed is None else prev_removed

        if active_files == prev_active and removed_files == prev_removed:
            return (
                gradio.update(),
                gradio.update(),
                gradio.update(),
                gradio.update(),
                active_files,
                removed_files,
                False,
            )

        unique_active_files = []
        if active_files is not None:
            for file in active_files:
                if file not in unique_active_files:
                    unique_active_files.append(file)

        unique_removed_files = []
        if removed_files is not None:
            for file in removed_files:
                if file not in unique_removed_files:
                    unique_removed_files.append(file)

        moved_to_removed = [x for x in prev_active if x not in unique_active_files]
        moved_to_active = [x for x in prev_removed if x not in unique_removed_files]

        new_active = unique_active_files + moved_to_active
        new_removed = unique_removed_files + moved_to_removed

        new_active = sort_file_list(new_active)

        return (
            gradio.update(value=new_active, interactive=True if new_active != [] else False),
            gradio.update(value=new_removed, interactive=True if new_removed != [] else False),
            gradio.update(value=new_active),
            gradio.update(value=new_removed),
            new_active,
            new_removed,
            True,
        )


def disable_contents(*args):
    group_len = len(args)
    if group_len == 1:
        return gradio.update(interactive=False)
    return [gradio.update(interactive=False) for _ in range(group_len)]


def enable_contents(*args):
    group_len = len(args)
    if group_len == 1:
        return gradio.update(interactive=True)
    return [gradio.update(interactive=True) for _ in range(group_len)]


def update_instance_view(selected_idx, instances_state, real_door_height, base_height_world, tmpdir_name, color_name):
    if not instances_state or not selected_idx:
        return gradio.update(value=None), gradio.update(value=None), gradio.update(value=None), gradio.update(
            value=None), gradio.update(value=None)

    idx = int(selected_idx.split(" ")[1])
    inst_pcd, inst_colors = instances_state[idx]

    if len(inst_pcd) == 0:
        return gradio.update(value=None), gradio.update(value=None), gradio.update(value=None), gradio.update(
            value=None), gradio.update(value=None)

    # min/max aabb
    min_bound = inst_pcd.min(axis=0)
    max_bound = inst_pcd.max(axis=0)
    extents = max_bound - min_bound
    width, height, depth = extents[0], extents[1], extents[2]
    center = (max_bound + min_bound) / 2.0

    # calculation
    real_h, real_w, real_d = get_real_coordinates_from_aligned(height, width, depth, base_height_world,
                                                               real_door_height)
    vol = real_h * real_w * real_d

    save_path = os.path.join(tmpdir_name, f"{color_name}_instance_{idx}.glb")
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=inst_pcd, colors=inst_colors / 255.))

    # visualize yellow bbox
    transform = np.eye(4)
    transform[:3, 3] = center
    bbox_outline = trimesh.path.creation.box_outline(extents=extents, transform=transform)
    bbox_outline.colors = np.tile([255, 255, 0, 255], (len(bbox_outline.entities), 1))
    scene.add_geometry(bbox_outline)
    scene.export(save_path)

    return gradio.update(value=save_path), gradio.update(value=real_h), gradio.update(value=real_w), gradio.update(
        value=real_d), gradio.update(value=vol)

