import open3d as o3d
import numpy as np
import trimesh
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def load_glb_as_pcd(file_path):

    # 1. trimesh로 glb 파일 로드
    scene = trimesh.load(file_path)

    points = []
    colors = []

    # glb 내부에 객체가 여러 개일 수 있으므로 순회 (단일 객체여도 정상 작동)
    geometries = scene.geometry.values() if isinstance(scene, trimesh.Scene) else [scene]

    for geom in geometries:
        # 정점 데이터가 없으면 스킵
        if not hasattr(geom, 'vertices') or len(geom.vertices) == 0:
            continue

        points.append(geom.vertices)

        # 2. 색상 추출
        if hasattr(geom, 'colors') and geom.colors is not None and len(geom.colors) > 0:
            # 순수 PointCloud로 읽힌 경우
            color_data = geom.colors[:, :3] / 255.0
            colors.append(color_data)

        elif hasattr(geom, 'visual') and hasattr(geom.visual,
                                                 'vertex_colors') and geom.visual.vertex_colors is not None and len(
                geom.visual.vertex_colors) > 0:
            # Mesh 형태로 읽혔지만 정점 색상이 있는 경우
            color_data = geom.visual.vertex_colors[:, :3] / 255.0
            colors.append(color_data)

        else:
            # 색상이 전혀 없는 경우 기본값(흰색)
            colors.append(np.ones((len(geom.vertices), 3)))

    if not points:
        raise ValueError("GLB 파일 안에서 포인트 데이터를 찾을 수 없습니다. (포맷이 호환되지 않을 수 있습니다)")

    # 모든 데이터를 하나의 Numpy 배열로 합침
    points = np.vstack(points)
    colors = np.vstack(colors)

    # 3. Open3D PointCloud 객체로 변환
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def process_and_cluster_pcd(pcd_path, reference_colors_rgb):

    # Point Cloud 로드
    print("Loading Point Cloud...")
    pcd = load_glb_as_pcd(pcd_path)

    # 1: 아티팩트 및 노이즈 제거 (Pre-processing)
    print("Step 1: Removing outliers using SOR...")
    pcd_sor, ind_sor = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=4.0) # std : 판정기준 / nb : 참고 주변점 개수

    print("Step 1.5: Removing spatial noise using global DBSCAN...")
    labels_global = np.array(pcd_sor.cluster_dbscan(eps=0.05, min_points=15, print_progress=False)) # eps : 탐색반경 / min : 군집크기

    valid_mask = labels_global >= 0
    pcd_clean = pcd_sor.select_by_index(np.where(valid_mask)[0].tolist())
    o3d.visualization.draw_geometries([pcd_clean], window_name="Filtered PCD")

    # 2: HSV 색공간 변환 및 채도(Saturation) 기반 흑백 필터링
    print("Step 2: Mapping colors using HSV Hue with Saturation Filter...")
    points_rgb = np.asarray(pcd_clean.colors)
    ref_colors_norm = reference_colors_rgb / 255.0

    # RGB -> HSV 변환
    points_hsv = mcolors.rgb_to_hsv(points_rgb)
    ref_hsv = mcolors.rgb_to_hsv(ref_colors_norm)

    # 채도(Saturation) 필터 생성
    # 흑백 벽면을 걸러내기 위한 기준. 0.15~0.20 사이가 적당
    SATURATION_THRESHOLD = 0.15
    gray_bg_mask = points_hsv[:, 1] < SATURATION_THRESHOLD

    # 거리 계산 로직
    h_points = points_hsv[:, 0:1]
    h_refs = ref_hsv[:, 0:1].T
    h_diff = np.abs(h_points - h_refs)
    h_dist = np.minimum(h_diff, 1.0 - h_diff)

    s_points = points_hsv[:, 1:2]
    s_refs = ref_hsv[:, 1:2].T
    s_dist = np.abs(s_points - s_refs)

    total_dist = (h_dist * 5.0) + (s_dist * 1.0)

    min_distances = np.min(total_dist, axis=1)
    color_ids = np.argmin(total_dist, axis=1)

    # 1. 컬러 임계값 적용 (섞인 색상 차단)
    COLOR_THRESHOLD = 1.3
    out_of_bounds_mask = min_distances >= COLOR_THRESHOLD
    color_ids[out_of_bounds_mask] = -1

    # 2. 흑백 필터 덮어씌우기 (채도가 낮은 벽/바닥을 최종적으로 -1로 확정)
    color_ids[gray_bg_mask] = -1

    # Step 3: 각 색상 그룹 내에서 Instance 분할 및 배경/노이즈 통합
    print("Step 3: Extracting instances and pruning spatial noise...")
    points = np.asarray(pcd_clean.points)

    final_instances = []
    global_instance_id = 0
    gray_points_list = []  # 노이즈 및 배경을 모두 모을 리스트

    #  추출에 성공한 색상 ID를 담을 집합(Set)
    extracted_color_ids = set()

    # 3-1. 유효한 21가지 색상 객체들 분할
    for color_idx in range(len(reference_colors_rgb)):
        class_mask = (color_ids == color_idx)
        if not np.any(class_mask):
            continue

        class_points = points[class_mask]

        if len(class_points) < 200:
            gray_points_list.append(class_points)  # 크기가 작으면 회색 리스트로
            continue

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(class_points)
        local_labels = np.array(temp_pcd.cluster_dbscan(eps=0.05, min_points=20, print_progress=False)) # 애매한 dbscan 포인트 <- 수정할 수 있으면 배제하는게 좋을듯?

        unique_labels = set(local_labels)
        for label in unique_labels:
            inst_mask = (local_labels == label)
            cluster_points = class_points[inst_mask]

            if label == -1 or len(cluster_points) < 300:
                gray_points_list.append(cluster_points)  # 노이즈 파편은 회색 리스트로
                continue

            inst_pcd = o3d.geometry.PointCloud()
            inst_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            inst_pcd.colors = o3d.utility.Vector3dVector(np.tile(ref_colors_norm[color_idx], (len(cluster_points), 1)))

            final_instances.append(inst_pcd)
            global_instance_id += 1

            #  인스턴스가 하나라도 생성되었다면 해당 색상 ID 기록
            extracted_color_ids.add(color_idx)

    # 3-2.배경(-1) 포인트와 탈락한 노이즈(gray_points_list) 통합
    bg_mask = (color_ids == -1)
    if np.any(bg_mask):
        gray_points_list.append(points[bg_mask])

    if len(gray_points_list) > 0:
        # 리스트에 모인 모든 회색 포인트들을 하나로 합침
        all_gray_points = np.vstack(gray_points_list)

        #  벽면이나 바닥 전체를 포괄하는 배경이므로, 구멍이 뚫리지 않도록 DBSCAN을 생략하고 통째로 렌더링
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(all_gray_points)
        bg_pcd.colors = o3d.utility.Vector3dVector(np.full((len(all_gray_points), 3), 0.5))

        # final_instances.append(bg_pcd) # 여기 주석을 살리냐 지우느냐 따라 backgroud 회색 pcd를 포함할건지 아닐지 조절 가능
        # 배경(-1) 존재 기록
        extracted_color_ids.add(-1)

    print(f"Total instances extracted (including 1 background): {len(extracted_color_ids)}") # 추출된 컬러가 곧 instance
    return final_instances, list(extracted_color_ids)

def show_legend(extracted_ids, palette_rgb):
    # 범례 창 크기 설정
    fig, ax = plt.subplots(figsize=(3, 6))
    ax.axis('off')  # 축 숨기기

    y_pos = 0.95
    step = 0.08

    # 배경(-1)이 추출되었는지 먼저 확인하고 상단에 그리기
    if -1 in extracted_ids:
        rect = patches.Rectangle((0.1, y_pos), 0.2, 0.05, facecolor=(0.5, 0.5, 0.5))
        ax.add_patch(rect)
        ax.text(0.4, y_pos + 0.025, 'Background', verticalalignment='center', fontsize=12, fontweight='bold')
        y_pos -= step

    # 나머지 0~20 클래스 그리기
    for idx in sorted(extracted_ids):
        if idx == -1:
            continue

        color = palette_rgb[idx] / 255.0
        rect = patches.Rectangle((0.1, y_pos), 0.2, 0.05, facecolor=color)
        ax.add_patch(rect)
        ax.text(0.4, y_pos + 0.025, f'Class {idx}', verticalalignment='center', fontsize=12)
        y_pos -= step

    plt.title('Extracted Classes', fontweight='bold', pad=20)

    legend_path = 'legend.png'
    plt.savefig(legend_path, bbox_inches='tight', dpi=150)

    # Matplotlib 메모리 해제 (매우 중요)
    plt.close(fig)

    # 윈도우 기본 이미지 뷰어로 파일 실행
    try:
        os.startfile(legend_path)
    except AttributeError:
        #  Mac/Linux에서 돌릴 경우를 대비한 예외 처리
        import subprocess
        import sys
        if sys.platform == "darwin":
            subprocess.call(["open", legend_path])
        else:
            subprocess.call(["xdg-open", legend_path])

if __name__ == "__main__":
    # 팔레트 기준색상 21가지 하드코딩으로 매핑
    palette_rgb = np.array([
        [163, 81, 251],[64, 222, 138],[255, 64, 64],
        [255, 161, 160],[255, 118, 51],[255, 182, 51], [209, 212, 53],[27, 150, 64]
    ,[76, 251, 18],[148, 207, 26],[0, 214, 193],[46,156,170],[0,196,255],[54,71,151],[102,117,255],[0,25,239],[134,58,255],[83,0,135],[205,58,255],[255,151,202],[255,57,201]], dtype = np.uint8)

    pcd_path = r'C:\Users\SMJ\PycharmProjects\instance_clustering\recon2.glb'
    # 반환값 2개
    instances, extracted_ids = process_and_cluster_pcd(pcd_path, palette_rgb)

    # 1. Matplotlib 범례 창 띄우기 (뒤에서 실행 상태 유지)
    show_legend(extracted_ids, palette_rgb)

    # 2. Open3D 시각화 창 띄우기
    o3d.visualization.draw_geometries(instances, window_name="3D Instances")

