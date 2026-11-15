"""
main.py - OpenCV 기반 실시간 영상 움직임 감지, 다중 객체 추적 및 특정 물체 재인식 애플리케이션 (+ 기호 AI 계약 검증 적용)

분석 대상 영상: /content/IMG_0591.MOV (또는 웹캠 스트림)
구현 기능:
1. 영상 입출력 및 제어: cv2.VideoCapture, 실시간 FPS 계산/표시, 키보드 제어 (q: 종료, s: 스냅샷, p: 일시정지)
2. 배경 차분: MOG2 (createBackgroundSubtractorMOG2) 및 absdiff 차영상 지원, 학습률(learning rate) 설정, 모폴로지(열림/닫힘)
3. 객체 감지: 윤곽선(Contour) 추출, 면적 필터링, 바운딩 박스 및 중심점 계산
4. 다중 객체 추적: tracker.py의 Tracker 클래스 연동 (Greedy / Hungarian / IoU / Kalman Filter), ID 및 이동 궤적 시각화
5. 특정 물체 재인식: ORB / SIFT 특징점 매칭 기반 타겟 물체 검출 시 "[TARGET DETECTED] (Target Found)" 표시
6. 보너스 과제 (침입 감지 구역): 마우스 드래그 구역 지정 및 중심점 진입 시 경고 표시 + 로그 파일(intrusion_log.txt) 기록
7. 기호 AI (Symbolic AI): @symbolic_contract 기반 사전/사후 조건 검증으로 잘못된 이미지/좌표/임계값 입력 오류를 원천 차단
"""

import argparse
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from symbolic_verifier import symbolic_contract, SymbolicContractError
from tracker import Tracker

BASE_DIR = "/content" if os.path.exists("/content/IMG_0591.MOV") else os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(BASE_DIR, "IMG_0591.MOV")
DEFAULT_TARGET = os.path.join(BASE_DIR, "target.png")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output_tracked.mp4")
DEFAULT_LOG = os.path.join(BASE_DIR, "intrusion_log.txt")


class FeatureTargetMatcher:  # "ORB 또는 SIFT 특징점 매칭을 통해 등록된 특정 물체(타겟 이미지)를 영상 내에서 재인식하는 클래스"
    """미리 등록된 타겟 물체 이미지를 로드하여 매 프레임 ORB/SIFT 특징점 매칭 및 호모그래피 검출을 수행하는 클래스"""

    def __init__(
        self,
        target_image_path: str,
        method: str = "orb",
        min_match_count: int = 18,
    ):  # "타겟 물체 이미지를 읽고 ORB 또는 SIFT 특징점 추출기와 BFMatcher를 초기화하는 생성자 메소드"
        """타겟 이미지를 로드하고 지정된 알고리즘(ORB 또는 SIFT)으로 기준 특징점과 디스크립터를 추출하는 메소드"""
        self.method = method.lower().strip()
        self.min_match_count = int(min_match_count)

        # 기호 AI 사전 검증: 타겟 이미지 존재 및 유효성 확인
        self.target_img = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
        if self.target_img is None or self.target_img.size == 0:
            raise FileNotFoundError(f"타겟 이미지를 찾을 수 없거나 손상되었습니다: {target_image_path}")

        self.target_gray = cv2.cvtColor(self.target_img, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.target_gray.shape[:2]

        # SIFT 또는 ORB 특징점 추출기 및 거리 측정 방식(L2 노름 vs 해밍 거리) 설정
        if self.method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=800)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.ratio_thresh = 0.72
        elif self.method == "orb":
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.ratio_thresh = 0.75
        else:
            raise SymbolicContractError(f"지원하지 않는 특징점 알고리즘입니다: {self.method}")

        # 타겟 물체의 특징점(Keypoints)과 디스크립터(Descriptors) 사전 계산
        self.kp_target, self.des_target = self.detector.detectAndCompute(
            self.target_gray, None
        )

    @symbolic_contract(
        role="현재 그레이스케일 프레임에서 타겟 물체의 특징점을 매칭하여 검출 여부와 투영 좌표를 반환하는 메소드",
        pre=lambda self, frame_gray: isinstance(frame_gray, np.ndarray) and frame_gray.ndim == 2 and frame_gray.size > 0,
        post=lambda res: isinstance(res, tuple) and len(res) == 4 and isinstance(res[0], bool) and res[1] >= 0 and res[2] >= 0,
    )
    def match_frame(
        self, frame_gray: np.ndarray
    ) -> Tuple[bool, int, int, Optional[np.ndarray]]:  # "매 프레임 특징점을 추출하고 Lowe's Ratio Test 및 RANSAC 호모그래피로 타겟 물체 등장 여부를 판별하는 메소드"
        """현재 프레임(Grayscale)에서 타겟 물체의 특징점을 매칭하고 (발견여부, 매칭수, 전체특징점수, 외곽선좌표)를 반환하는 메소드"""
        if self.des_target is None or len(self.kp_target) == 0:
            return False, 0, 0, None

        # 현재 프레임에서 특징점 및 디스크립터 추출
        kp_frame, des_frame = self.detector.detectAndCompute(frame_gray, None)
        if des_frame is None or len(kp_frame) < 4:
            return False, 0, len(self.kp_target), None

        # k-NN (k=2) 매칭 후 Lowe's Ratio Test로 모호한 매칭 제거
        raw_matches = self.matcher.knnMatch(self.des_target, des_frame, k=2)
        good_matches = []
        for m_n in raw_matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        found = len(good_matches) >= self.min_match_count
        corners_projected = None

        # 충분한 매칭점이 확보된 경우 RANSAC 기반 호모그래피(Homography) 행렬을 추정하여 기하학적 일관성 검증
        if found:
            src_pts = np.float32(
                [self.kp_target[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                inliers = int(mask.ravel().sum()) if mask is not None else 0
                if inliers >= max(10, self.min_match_count // 2):
                    pts = np.float32(
                        [
                            [0, 0],
                            [0, self.h - 1],
                            [self.w - 1, self.h - 1],
                            [self.w - 1, 0],
                        ]
                    ).reshape(-1, 1, 2)
                    corners_projected = np.int32(cv2.perspectiveTransform(pts, H))
                else:
                    found = False
            else:
                found = False

        return found, len(good_matches), len(self.kp_target), corners_projected


class IntrusionZoneMonitor:  # "[보너스 과제 2] 화면 내 사각형 침입 감지 구역을 마우스로 설정하고 객체 중심점 진입 시 경고 및 로그를 남기는 클래스"
    """화면 내 특정 영역(사각형)을 감시하여 추적 객체의 중심점이 들어오면 경고 발생 및 파일 로깅을 수행하는 클래스"""

    def __init__(
        self,
        default_rect: Tuple[int, int, int, int] = (930, 180, 1740, 840),
        log_path: str = "/content/intrusion_log.txt",
    ):  # "기본 침입 감지 구역 좌표(x1, y1, x2, y2)와 로그 파일 경로를 초기화하는 생성자 메소드"
        """침입 감지 영역 좌표와 마우스 드래그 상태, 로그 파일을 초기화하는 생성자 메소드"""
        self.rect: Tuple[int, int, int, int] = default_rect
        self.drawing: bool = False
        self.start_pt: Tuple[int, int] = (0, 0)
        self.log_path = log_path
        self.logged_events: List[str] = []

        # 로그 파일 초기화 및 컬럼 헤더 기록
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== Intrusion Detection Log ===\n")
            f.write("Timestamp | Frame | Object_ID | Centroid(x,y) | Zone(x1,y1,x2,y2)\n")

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: object
    ) -> None:  # "마우스 좌클릭 드래그 이벤트를 처리하여 사용자가 화면에 침입 감지 사각형 영역을 직접 지정하도록 하는 콜백 메소드"
        """OpenCV 창에서 마우스 드래그로 침입 감지 사각형(rect)을 실시간 갱신하는 콜백 메소드"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            x1, y1 = min(self.start_pt[0], x), min(self.start_pt[1], y)
            x2, y2 = max(self.start_pt[0], x), max(self.start_pt[1], y)
            self.rect = (x1, y1, x2, y2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.start_pt[0], x), min(self.start_pt[1], y)
            x2, y2 = max(self.start_pt[0], x), max(self.start_pt[1], y)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                self.rect = (x1, y1, x2, y2)

    @symbolic_contract(
        role="추적 중인 객체들의 중심점이 침입 감지 구역 내부에 진입했는지 판별하고 로그 파일에 기록하는 메소드",
        pre=lambda self, frame_idx, tracked_objects: frame_idx >= 0 and isinstance(tracked_objects, dict),
        post=lambda res: isinstance(res, list),
    )
    def check_and_log(
        self, frame_idx: int, tracked_objects: dict
    ) -> List[int]:  # "객체 중심점(cx, cy)이 사각형 감지 구역 안에 들어왔는지 검사하고 침입한 객체 ID 리스트를 반환 및 로깅하는 메소드"
        """객체 중심점이 침입 감지 구역 내에 들어왔는지 확인하고 로그를 기록하는 메소드"""
        x1, y1, x2, y2 = self.rect
        intruding_ids: List[int] = []
        for obj_id, info in tracked_objects.items():
            cx, cy = info["centroid"]
            # 현재 프레임에서 실제로 검출(disappeared == 0)된 객체의 중심점이 구역 내부에 있는지 확인
            if x1 <= cx <= x2 and y1 <= cy <= y2 and info["disappeared"] == 0:
                intruding_ids.append(obj_id)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                entry = f"{ts} | Frame {frame_idx:04d} | ID:{obj_id} | ({cx}, {cy}) | ({x1},{y1},{x2},{y2})"
                self.logged_events.append(entry)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
        return intruding_ids


@symbolic_contract(
    role="타겟 이미지 파일이 없을 경우 원본 영상의 첫 프레임에서 특정 물체(신발/오브젝트) 영역을 크롭하여 생성하는 함수",
    pre=lambda video_path, target_path=DEFAULT_TARGET: isinstance(video_path, str) and len(video_path) > 0,
    post=lambda res: isinstance(res, str) and os.path.exists(res),
)
def ensure_default_target_image(
    video_path: str, target_path: str = DEFAULT_TARGET
) -> str:  # "특징점 매칭에 사용할 타겟 물체 이미지(target.png)를 준비하고 경로를 반환하는 함수"
    """영상(IMG_0591.MOV)의 첫 프레임에서 기준 신발 물체 영역을 타겟 이미지로 추출·저장하는 함수"""
    if os.path.exists(target_path):
        return target_path

    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    cap.release()
    if not ret or frame0 is None:
        raise RuntimeError(f"비디오에서 첫 프레임을 읽을 수 없습니다: {video_path}")

    # IMG_0591.MOV 신발 물체 영역 추출 (y: 150~820, x: 100~460)
    h, w = frame0.shape[:2]
    y1, y2 = int(h * 0.08), int(h * 0.43)
    x1, x2 = int(w * 0.09), int(w * 0.43)
    crop = frame0[y1:y2, x1:x2]
    cv2.imwrite(target_path, crop)
    return target_path


@symbolic_contract(
    role="배경 차분(MOG2 또는 absdiff)과 모폴로지 연산, 윤곽선 면적 필터링을 통해 움직이는 객체의 바운딩 박스를 검출하는 함수",
    pre=lambda frame, bg_subtractor, ref_gray, bg_method="mog2", learning_rate=0.002, min_area=15000.0: isinstance(frame, np.ndarray) and frame.ndim == 3 and min_area > 0,
    post=lambda res: isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], list) and isinstance(res[1], np.ndarray),
)
def detect_moving_objects(
    frame: np.ndarray,
    bg_subtractor: Optional[cv2.BackgroundSubtractor],
    ref_gray: Optional[np.ndarray],
    bg_method: str = "mog2",
    learning_rate: float = 0.002,
    min_area: float = 15000.0,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:  # "배경 차분 및 모폴로지(열림/닫힘)를 수행하여 전경 객체들의 바운딩 박스 리스트와 마스크를 반환하는 함수"
    """배경 차분과 모폴로지 연산을 적용하여 움직이는 객체의 바운딩 박스 리스트와 전경 마스크를 반환하는 함수"""
    if bg_method == "absdiff":
        # 직접 차영상(absdiff) 방식: 기준 배경 프레임과 현재 프레임 간의 절대 밝기 차이 계산
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (11, 11), 0
        )
        diff = cv2.absdiff(ref_gray, gray)
        _, fg_mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    else:
        # MOG2 가우시안 혼합 배경 모델링 방식: 그림자(127)를 제외하고 순수 전경(255)만 이진화
        raw_mask = bg_subtractor.apply(frame, learningRate=learning_rate)
        _, fg_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

    # 모폴로지 연산: 열림(MORPH_OPEN)으로 미세 노이즈 제거 -> 닫힘(MORPH_CLOSE)으로 손가락 등 떨어진 부위를 한 덩어리로 결합
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))

    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # 외곽 윤곽선(Contour) 추출 및 최소 면적(min_area) 필터링
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

    # 좌측에서 우측 순서로 정렬하여 일관성 유지
    bboxes.sort(key=lambda b: b[0])
    return bboxes, fg_mask


@symbolic_contract(
    role="비디오 스트림을 프레임 단위로 읽어 객체 감지, 다중 객체 추적, 특징점 재인식, 침입 감지를 통합 실행하는 메인 파이프라인 함수",
    post=lambda res: isinstance(res, dict) and res.get("total_frames", 0) > 0,
)
def run_video_analysis(
    video_source: str = DEFAULT_VIDEO,
    target_path: str = DEFAULT_TARGET,
    output_path: Optional[str] = DEFAULT_OUTPUT,
    bg_method: str = "mog2",
    learning_rate: float = 0.002,
    match_mode: str = "greedy",
    use_kalman: bool = False,
    feature_method: str = "orb",
    min_area: float = 15000.0,
    headless: bool = True,
) -> dict:  # "비디오 파일 또는 웹캠을 입력받아 전체 컴퓨터 비전 추적 파이프라인을 실행하고 결과 영상 및 통계를 생성하는 함수"
    """비디오 분석 파이프라인을 실행하고 결과 오버레이 영상 저장 및 정량 통계를 반환하는 함수"""
    target_path = ensure_default_target_image(video_source, target_path)

    cap = cv2.VideoCapture(0 if video_source == "0" else video_source)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 소스를 열 수 없습니다: {video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

    # 배경 차분기, 다중 객체 추적기, 특징점 매칭기, 침입 감지 모니터 객체 생성
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=36, detectShadows=True
    )
    tracker = Tracker(
        max_disappeared=12,
        max_distance=260.0,
        min_iou=0.05,
        max_trajectory_len=90,
        mode=match_mode,
        use_kalman=use_kalman,
    )
    target_matcher = FeatureTargetMatcher(
        target_image_path=target_path,
        method=feature_method,
        min_match_count=18,
    )
    # 세로 영상(1080x1920) 및 가로 영상에 모두 대응하는 침입 감지 구역 설정
    default_zone = (
        (int(width * 0.1), int(height * 0.2), int(width * 0.9), int(height * 0.85))
        if width < height
        else (930, 180, 1740, 840)
    )
    intrusion_monitor = IntrusionZoneMonitor(
        default_rect=default_zone,
        log_path=DEFAULT_LOG,
    )

    window_name = "OpenCV Real-Time Object Tracking & Re-ID"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, intrusion_monitor.mouse_callback)

    ref_gray = None
    frame_idx = 0
    paused = False
    snapshot_count = 0

    fps_history: List[float] = []
    target_detected_frames = 0
    intrusion_frames = 0
    frame_logs: List[dict] = []

    while True:
        t_start = time.perf_counter()
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if ref_gray is None:
                ref_gray = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (11, 11), 0
                )

            display = frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1단계: 배경 차분 및 움직이는 객체 바운딩 박스 검출
            bboxes, _ = detect_moving_objects(
                frame=frame,
                bg_subtractor=bg_subtractor,
                ref_gray=ref_gray,
                bg_method=bg_method,
                learning_rate=learning_rate,
                min_area=min_area,
            )

            # 2단계: 다중 객체 추적기(Tracker)로 ID 부여 및 이동 궤적 업데이트
            tracked = tracker.update(bboxes)

            # 3단계: 특징점 매칭(ORB/SIFT)으로 등록된 특정 물체(Apple Watch) 재인식
            target_found, good_matches, total_kp, target_poly = (
                target_matcher.match_frame(frame_gray)
            )
            if target_found:
                target_detected_frames += 1

            # 4단계: 침입 감지 구역 진입 여부 확인 및 로그 기록
            intruding_ids = intrusion_monitor.check_and_log(frame_idx, tracked)
            if intruding_ids:
                intrusion_frames += 1

            # --- 화면 시각화 오버레이 ---
            # (a) 침입 감지 구역 사각형 표시
            zx1, zy1, zx2, zy2 = intrusion_monitor.rect
            zone_color = (0, 0, 255) if intruding_ids else (255, 200, 0)
            cv2.rectangle(display, (zx1, zy1), (zx2, zy2), zone_color, 2)
            cv2.putText(
                display,
                "Intrusion Zone (Bonus)",
                (zx1 + 8, zy1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                zone_color,
                2,
            )

            # (b) 특정 물체 재인식 성공 시 "[TARGET DETECTED] Target Found" 표시
            if target_found:
                if target_poly is not None:
                    cv2.polylines(
                        display, [target_poly], True, (255, 255, 0), 3, cv2.LINE_AA
                    )
                cv2.putText(
                    display,
                    f"[TARGET DETECTED] Target Found ({feature_method.upper()}: {good_matches}/{total_kp})",
                    (30, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (0, 255, 255),
                    3,
                )

            # (c) 각 추적 객체의 초록색 바운딩 박스, 고유 ID, 중심점, 빨간색 이동 궤적선 표시
            for obj_id, info in tracked.items():
                if info["disappeared"] > 0:
                    continue
                x, y, w, h = info["bbox"]
                cx, cy = info["centroid"]
                traj = info["trajectory"]

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
                label = f"ID:{obj_id}"
                cv2.rectangle(
                    display, (x, max(0, y - 36)), (x + 110, y), (0, 255, 0), -1
                )
                cv2.putText(
                    display,
                    label,
                    (x + 8, max(26, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 0),
                    2,
                )

                # 중심점 표시 (빨간색 원)
                cv2.circle(display, (cx, cy), 7, (0, 0, 255), -1)

                # 칼만 필터 예측 좌표 표시 (하늘색 원)
                if use_kalman and "predicted" in info:
                    px, py = info["predicted"]
                    cv2.circle(display, (px, py), 6, (255, 255, 0), 2)

                # 객체의 이동 경로(Trajectory)를 빨간색 선으로 연결하여 시각화
                for k in range(1, len(traj)):
                    if traj[k - 1] is None or traj[k] is None:
                        continue
                    thickness = max(2, int(np.sqrt(k / float(len(traj))) * 5))
                    cv2.line(
                        display,
                        traj[k - 1],
                        traj[k],
                        (0, 0, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

            # (d) 침입 감지 시 상단 경고 메시지 출력
            if intruding_ids:
                cv2.putText(
                    display,
                    f"WARNING: INTRUSION DETECTED (ID: {intruding_ids})",
                    (width // 2 - 280, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

            # (e) 현재 FPS 및 조작 안내 문구 표시
            elapsed = max(1e-6, time.perf_counter() - t_start)
            current_fps = 1.0 / elapsed
            fps_history.append(current_fps)

            cv2.putText(
                display,
                f"FPS: {current_fps:.1f} | Frame: {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} | Mode: {match_mode.upper()}",
                (30, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (50, 255, 50),
                2,
            )
            cv2.putText(
                display,
                "Press 'q' to quit, 's' to save snapshot, 'p' to pause",
                (30, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (220, 220, 220),
                2,
            )

            if writer is not None:
                writer.write(display)

            frame_logs.append(
                {
                    "frame": frame_idx,
                    "num_detections": len(bboxes),
                    "active_ids": [
                        oid for oid, inf in tracked.items() if inf["disappeared"] == 0
                    ],
                    "all_ids": list(tracked.keys()),
                    "centroids": {
                        oid: inf["centroid"]
                        for oid, inf in tracked.items()
                        if inf["disappeared"] == 0
                    },
                    "target_found": target_found,
                    "good_matches": good_matches,
                    "intruding_ids": intruding_ids,
                }
            )
            frame_idx += 1

        # 키보드 제어 이벤트 처리 (q: 종료, s: 스냅샷 저장, p: 일시정지 토글)
        if not headless:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_name = f"/content/snapshot_{snapshot_count:03d}_frame{frame_idx:04d}.png"
                cv2.imwrite(snap_name, display)
                snapshot_count += 1
            elif key == ord("p"):
                paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()

    return {
        "total_frames": frame_idx,
        "avg_fps": float(np.mean(fps_history)) if fps_history else 0.0,
        "total_unique_ids": tracker.next_object_id - 1,
        "target_detected_frames": target_detected_frames,
        "intrusion_frames": intrusion_frames,
        "frame_logs": frame_logs,
    }


def parse_args() -> argparse.Namespace:  # "커맨드라인 실행 인자(비디오 경로, 학습률, 매칭 알고리즘 등)를 파싱하여 반환하는 함수"
    """애플리케이션 실행을 위한 명령행 인자(CLI Arguments)를 파싱하는 함수"""
    parser = argparse.ArgumentParser(
        description="OpenCV Real-Time Object Tracking & Feature Re-ID Application"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_VIDEO,
        help="입력 비디오 파일 경로 또는 '0'(웹캠)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="재인식할 타겟 물체 이미지 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="분석 결과 오버레이 비디오 저장 경로",
    )
    parser.add_argument(
        "--bg-method",
        type=str,
        default="mog2",
        choices=["mog2", "absdiff"],
        help="배경 차분 알고리즘 선택 (mog2 또는 absdiff)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.002,
        help="MOG2 배경 학습률 (예: 0.001, 0.01, 0.05)",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="greedy",
        choices=["greedy", "hungarian", "iou"],
        help="객체 매칭 알고리즘 선택 (greedy, hungarian, iou)",
    )
    parser.add_argument(
        "--use-kalman",
        action="store_true",
        help="Kalman Filter 위치 예측 활성화 여부",
    )
    parser.add_argument(
        "--feature-method",
        type=str,
        default="orb",
        choices=["orb", "sift"],
        help="특징점 추출 알고리즘 선택 (orb 또는 sift)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=15000.0,
        help="윤곽선 최소 면적 임계값",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="GUI 창(cv2.imshow) 출력 활성화 (로컬 데스크톱 환경용)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_video_analysis(
        video_source=args.video,
        target_path=args.target,
        output_path=args.output,
        bg_method=args.bg_method,
        learning_rate=args.learning_rate,
        match_mode=args.match_mode,
        use_kalman=args.use_kalman,
        feature_method=args.feature_method,
        min_area=args.min_area,
        headless=not args.gui,
    )
    print("=== Video Analysis Summary ===")
    print(f"Processed Frames      : {summary['total_frames']}")
    print(f"Average FPS           : {summary['avg_fps']:.2f}")
    print(f"Total Assigned IDs    : {summary['total_unique_ids']}")
    print(f"Target Detected Frames: {summary['target_detected_frames']}/{summary['total_frames']}")
    print(f"Intrusion Alert Frames: {summary['intrusion_frames']}/{summary['total_frames']}")
