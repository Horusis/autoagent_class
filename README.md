# 3D Object Dimension and Volume Estimation from Multi-View Images

**AIC6056 자율에이전트시각표현학습(AAVRL)** 수업 프로젝트 
**Team VML:** 한상범, 오재석, 서민재, 김다운

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![SAM 2](https://img.shields.io/badge/Model-SAM%202-green.svg)]()
[![XSAM](https://img.shields.io/badge/Model-XSAM-red.svg)]()
[![SLAM3R](https://img.shields.io/badge/Model-SLAM3R-orange.svg)]()
[![Open3D](https://img.shields.io/badge/Library-Open3D-lightgrey.svg)]()

##  Project Overview
본 프로젝트는 사진 이미지(Multi-view)를 활용하여 현실 세계 물체의 3차원 크기(가로, 세로, 깊이) 및 부피를 추정하는 파이프라인을 제안합니다. 
2D 이미지 상의 Instance Segmentation 결과를 비디오 전체에 걸쳐 추적(Tracking)하고, 이를 3D Point Cloud로 리프팅(Lifting)한 뒤, 현실 좌표계 기반의 축 정렬 및 스케일 보정을 거쳐 최종적인 3D Bounding Box와 실제 사이즈를 도출합니다.

##  Core Pipeline

우리 팀의 파이프라인은 다음 4가지 주요 단계로 구성됩니다:

### 1. 2D Instance & Video Segmentation
* **Initial Segmentation:** `XSAM`을 사용하여 첫 프레임(또는 키 프레임)에 대해 초기 인스턴스 세그멘테이션을 수행합니다.
* **Instance Tracking:** 앞 단계에서 추출된 세그멘테이션 마스크를 `SAM 2`의 프롬프트로 입력하여 Video Segmentation을 진행합니다. 이를 통해 전체 Scene(Multi-view images)에 걸쳐 동일 객체에 대한 일관된 Instance ID를 유지합니다.

### 2. 3D Lifting (Image to 3D)
* **Point Cloud Generation:** 분할(Segmented)된 Multi-view 이미지를 바탕으로 `SLAM3R`을 적용하여 3D Point Cloud(PCD) 데이터를 생성합니다.

### 3. Instance Extraction in 3D Space
* **Color Mapping:** 2D Segmentation Mask에 사용된 색상을 HSV color space를 활용하여 3D PCD 상에 단색으로 매핑합니다.
* **Separation:** Segmentation Class와 매핑된 색상 정보를 기반으로 각 Instance를 추출하고, 전체 PCD에서 개별 Object를 분리(Crop)합니다.

### 4. Axis Alignment & Absolute Scale Calibration
* **PCA/SVD Axis Alignment:** 현실 좌표계와 3D 공간의 축을 정렬하기 위해 문(Door)이나 TV와 같이 수직/수평이 뚜렷한 **Base Object**를 설정합니다. 분리된 Base Object의 Point Cloud에 대해 **SVD(특이값 분해)**를 수행하여 다음과 같이 전체 공간의 축을 정렬합니다:
  * 가장 긴 축(1st Principal Component) $\rightarrow$ 높이(Height, Y-axis)
  * 두 번째로 긴 축(2nd Principal Component) $\rightarrow$ 너비(Width, X-axis)
* **Bounding Box Generation:** 정렬된 축을 기준으로 분리된 모든 Object들의 3D Bounding Box를 생성하여 상대적 크기를 측정합니다.
* **Real-World Scale Calculation:** Base Object의 '실제 현실 높이(Real-world Height)'를 시스템에 입력하면, 3D 공간 상의 길이와의 비율(Scale factor)을 계산하여 모든 BBox의 실제 가로, 세로, 깊이 및 부피를 도출합니다.

##  Environment & Prerequisites

원활한 모델 추론 및 Point Cloud 처리를 위해 Linux 환경과 충분한 VRAM을 갖춘 GPU를 권장합니다.
* **OS:** Ubuntu 20.04 / 22.04
* **Hardware:** NVIDIA GPU (e.g., RTX 3090 24GB or higher recommended for SLAM3R & SAM 2)
* **Dependencies:**
* 자세한 사항은 enviroment_clean.yaml 을 확인해주세요!

## Quick start
다운로드와 환경설정을 마치신 후에 SAM2 의 체크포인트를 따로 다운로드 받아주셔야 합니다.
이후 아래코드를 통해 gradio UI 로 실행해주세요
gradio run_overall.py

## Data preparation
실행된 gradio 에 준비하신 custom image 를 드레그로 입력하시면 됩니다

Acknowledgments
본 프로젝트는 한양대학교 대학원 AIC6056 자율에이전트시각표현학습(AAVRL) 수업의 일환으로 진행되었습니다.

References: 
* SAM 2: Segment Anything in Images and Videos
* SLAM3R
* XSAM
