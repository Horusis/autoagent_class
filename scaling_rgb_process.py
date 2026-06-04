import os
import numpy as np
from PIL import Image
import cv2



def image_scaling_with_mask(image, mask, mode='R_door'):
    input_is_pil = isinstance(image, Image.Image)

    image_np = np.array(image).copy()
    mask_np = np.array(mask)

    # bool / 0~1 / 0~255 마스크 모두 대응
    region = mask_np > 0
    if not np.any(region):
        return Image.fromarray(image_np) if input_is_pil else image_np

    region_pixels = image_np[region].astype(np.float32)

    # 원본 밝기를 이용해서 명암 유지
    gray = (
        0.299 * region_pixels[:, 0]
        + 0.587 * region_pixels[:, 1]
        + 0.114 * region_pixels[:, 2]
    )

    # 마스크 내부에서만 대비를 조금 늘려서 물체감 유지
    g_min, g_max = gray.min(), gray.max()
    if g_max > g_min:
        gray = (gray - g_min) / (g_max - g_min)
        gray = 40.0 + gray * 215.0
    else:
        gray = np.clip(gray, 0, 255)

    gray = gray.astype(np.uint8)

    if mode == 'R':
        red_tone = np.stack([
            gray,
            np.zeros_like(gray),
            np.zeros_like(gray)
        ], axis=-1)
        image_np[region] = red_tone

    elif mode == 'G':
        green_tone = np.stack([
            np.zeros_like(gray),
            gray,
            np.zeros_like(gray)
        ], axis=-1)
        image_np[region] = green_tone

    elif mode == 'B':
        blue_tone = np.stack([
            np.zeros_like(gray),
            np.zeros_like(gray),
            gray
        ], axis=-1)
        image_np[region] = blue_tone

    elif mode == 'outside':
        # grayscale + 원본 명암 유지
        gray_tone = np.stack([gray, gray, gray], axis=-1)
        image_np[region] = gray_tone

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return Image.fromarray(image_np) if input_is_pil else image_np




if __name__ == "__main__":
    image_dir = "/home/ccat/PycharmProjects/auto_drive/X-SAM/datas/multi_co_vis"
    output_dir = "/home/ccat/PycharmProjects/auto_drive/X-SAM/datas/multi_co_vis_scaled"
    os.makedirs(output_dir, exist_ok=True)
    for image_name in os.listdir(image_dir):
        if not os.path.isdir(os.path.join(image_dir, image_name)):
            continue
        image_path = os.path.join(image_dir, image_name)

        image = Image.open(os.path.join(image_path, 'origin.png')).convert('RGB')

        door_mask_path = os.path.join(image_path, '00_door.png')
        chair_mask_path = os.path.join(image_path, '01_chair.png')
        refri_mask_path = os.path.join(image_path, '06_refrigerator.png')
        door_exist = os.path.exists(door_mask_path)
        chair_exist = os.path.exists(chair_mask_path)
        refri_exist = os.path.exists(refri_mask_path)

        door_mask = cv2.imread(door_mask_path, cv2.IMREAD_GRAYSCALE) if door_exist else np.zeros(image.size[::-1], dtype=np.uint8)
        chair_mask = cv2.imread(chair_mask_path, cv2.IMREAD_GRAYSCALE) if chair_exist else np.zeros(image.size[::-1], dtype=np.uint8)
        refri_mask = cv2.imread(refri_mask_path, cv2.IMREAD_GRAYSCALE) if refri_exist else np.zeros(image.size[::-1], dtype=np.uint8)

        overall_mask = cv2.bitwise_or(door_mask, chair_mask)
        overall_mask = cv2.bitwise_or(overall_mask, refri_mask)
        inverse_overall_mask = cv2.bitwise_not(overall_mask)

        result = image_scaling_with_mask(image, door_mask, mode='R') if door_exist else image
        result = image_scaling_with_mask(result, chair_mask, mode='G') if chair_exist else result
        result = image_scaling_with_mask(result, refri_mask, mode='B') if refri_exist else result
        result = image_scaling_with_mask(result, inverse_overall_mask, mode='outside')

        result.save(os.path.join(output_dir, f"{image_name}_scaled.png"))


