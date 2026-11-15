# OpenCV 실시간 영상 움직임 감지·다중 객체 추적 및 한계 분석 (`kangsik-seo/opencv`)

본 리포지토리는 **OpenCV**만을 활용하여 영상([IMG_0591.MOV](file:///content/IMG_0591.MOV)) 속 움직이는 객체를 배경과 분리해 감지하고, 고유 ID 부여 및 이동 궤적 시각화, 특징점 매칭(ORB/SIFT) 기반 특정 물체 재인식, 침입 감지 구역 모니터링, 그리고 **기호 AI(Symbolic AI)** 기반 코드·문서 정합성 및 수리 불변식 검증을 수행하는 프로젝트입니다.

---

## 1. 프로젝트 파일 구조

| 파일명 | 역할 및 설명 |
| :--- | :--- |
| `tracker.py` | 다중 객체 추적기 모듈 (`Tracker`, `SingleKalmanFilter`, `compute_iou`, `hungarian_algorithm_numpy`) |
| `main.py` | 실시간 영상 입출력, 배경 차분(MOG2/absdiff), 객체 감지·추적, 타겟 재인식(`FeatureTargetMatcher`), 침입 감지(`IntrusionZoneMonitor`) 메인 애플리케이션 |
| `evaluate.py` | 4대 추적 시나리오(30회), ORB/SIFT 5대 조건 매칭, 배경 학습률 3종 비교, 보너스 알고리즘 비교 및 시각화 생성 스크립트 |
| `symbolic_verifier.py` | 기호 AI(Symbolic AI) 기반 AST 정적 분석, 호어 논리(`{P} C {Q}`) 계약 데코레이터(`@symbolic_contract`), 수리 공리 증명 및 코드-`README.md` 일치성 검증 엔진 |
| `REPORT.md` | 정량적 성능 측정 결과 및 고전적 컴퓨터 비전의 한계·딥러닝 필요성 분석 보고서 |
| `IMG_0591.MOV` | 분석 대상 원본 영상 (`1080x1920`, `30 FPS`, `174 Frames`, 발 교차·가림·정지 실측 데이터) |
| `target.png` | 특징점 매칭용 등록 물체 이미지 (기준 신발 크롭 이미지) |
| `output_tracked.mp4` | 바운딩 박스, 고유 ID, 이동 궤적(빨간색 선), 타겟 검출, 침입 경고가 오버레이된 결과 비디오 |
| `intrusion_log.txt` | 침입 감지 구역 진입 이벤트 로그 파일 |
| `evaluation_results.json` | 기호 AI 검증 및 전체 정량 평가 결과 JSON 데이터 |
| `fig1_tracking_keyframes.png` | 주요 프레임별 객체 추적·궤적·타겟 검출·침입 경고 시각화 |
| `fig2_learning_rate_comparison.png` | MOG2 배경 학습률(`0.001`, `0.01`, `0.05`)에 따른 정지 객체 흡수 및 잔상 비교 |
| `fig3_feature_matching_conditions.png` | 정면, 30°/60° 회전, 30%/50% 가림 조건별 ORB vs SIFT 특징점 매칭 비교 |
| `fig4_bonus_comparison.png` | Greedy vs NumPy Hungarian vs IoU vs Kalman Filter 성능 비교 차트 |

---

## 2. 코드 아키텍처 (클래스 및 메소드 명세)

모든 클래스와 메소드에는 `""` 형태의 역할 독스트링과 호어 논리 기호 계약(`@symbolic_contract`)이 적용되어 있습니다.

### 2.1 `tracker.py`
- **`compute_iou(boxA, boxB)`**: 두 바운딩 박스 `(x, y, w, h)` 간의 교집합 대비 합집합 비율(IoU)을 계산하는 함수
- **`hungarian_algorithm_numpy(cost_matrix)`**: **[보너스 과제 1]** `scipy` 없이 순수 NumPy만으로 구현한 Hungarian(Kuhn-Munkres) 전역 최소 비용 1:1 할당 함수
- **`class SingleKalmanFilter`**: **[보너스 과제 4]** `cv2.KalmanFilter(4, 2)`를 활용하여 단일 객체의 2차원 위치와 속도 `[x, y, vx, vy]^T`를 추정·예측하는 클래스
  - `__init__(self, init_x, init_y)`: 초기 중심점 좌표로 칼만 필터의 상태 행렬과 공분산 행렬을 초기화하는 메소드
  - `predict(self)`: 등속 운동 모델을 바탕으로 현재 프레임에서의 예상 중심점 좌표를 반환하는 메소드
  - `correct(self, x, y)`: 실제 관측된 중심점 `(x, y)`를 반영하여 칼만 필터 상태를 보정하는 메소드
- **`class Tracker`**: 다중 객체 추적 클래스 (Greedy / Hungarian / IoU 매칭 및 칼만 필터 결합 지원)
  - `__init__(self, max_disappeared, max_distance, min_iou, max_trajectory_len, mode, use_kalman)`: 추적기 상태 사전들과 하이퍼파라미터를 초기화하는 생성자 메소드
  - `_assert_symbolic_invariants(self)`: 기호 AI 상태 불변식(`objects`, `bboxes`, `disappeared`, `trajectories` 키 집합 일치 및 카운터 경계)을 검증하는 메소드
  - `register(self, bbox)`: 새로운 객체를 등록하고 고유 ID를 부여하는 메소드
  - `deregister(self, object_id)`: 일정 프레임(`max_disappeared`) 이상 미등장한 객체 ID를 삭제하는 메소드
  - `_compute_distance_matrix(self, object_ids, input_centroids)`: 기존 객체 위치(또는 칼만 필터 예측 위치)와 현재 검출 중심점 간 유클리드 거리 행렬을 계산하는 메소드
  - `update(self, bboxes)`: 현재 프레임 바운딩 박스 리스트를 입력받아 ID 매칭, 궤적 갱신, 신규 등록, 소실 삭제를 수행하는 메인 메소드
  - `_build_output(self)`: 현재 추적 중인 모든 객체의 상태 정보를 표준화된 딕셔너리로 반환하는 메소드

### 2.2 `main.py`
- **`class FeatureTargetMatcher`**: 미리 등록된 타겟 물체 이미지(`target.png`)를 로드하여 매 프레임 ORB/SIFT 특징점 매칭 및 호모그래피 검출을 수행하는 클래스
  - `__init__(self, target_image_path, method, min_match_count)`: 타겟 이미지를 로드하고 지정된 알고리즘(`orb` 또는 `sift`)으로 기준 특징점과 디스크립터를 추출하는 메소드
  - `match_frame(self, frame_gray)`: 현재 프레임에서 타겟 물체의 특징점을 매칭하고 `(found, good_matches, total_kp, corners_projected)`를 반환하는 메소드
- **`class IntrusionZoneMonitor`**: **[보너스 과제 2]** 화면 내 특정 영역(사각형)을 마우스로 지정하고 객체 중심점 진입 시 경고 발생 및 파일 로깅을 수행하는 클래스
  - `__init__(self, default_rect, log_path)`: 침입 감지 영역 좌표와 마우스 드래그 상태, 로그 파일을 초기화하는 생성자 메소드
  - `mouse_callback(self, event, x, y, flags, param)`: OpenCV 창에서 마우스 드래그로 침입 감지 사각형을 실시간 갱신하는 콜백 메소드
  - `check_and_log(self, frame_idx, tracked_objects)`: 객체 중심점이 침입 감지 구역 내에 들어왔는지 확인하고 로그를 기록하는 메소드
- **`ensure_default_target_image(video_path, target_path)`**: 영상의 첫 프레임에서 책상 좌측 Apple Watch 물체 영역을 타겟 이미지로 추출·저장하는 함수
- **`detect_moving_objects(frame, bg_subtractor, ref_gray, bg_method, learning_rate, min_area)`**: 배경 차분(`mog2` 또는 `absdiff`)과 모폴로지 연산(열림, 닫힘)을 적용하여 움직이는 객체의 바운딩 박스 리스트와 전경 마스크를 반환하는 함수
- **`run_video_analysis(video_source, target_path, output_path, bg_method, learning_rate, match_mode, use_kalman, feature_method, min_area, headless)`**: 비디오 분석 파이프라인을 실행하고 결과 오버레이 영상 저장 및 정량 통계를 반환하는 함수
- **`parse_args()`**: 명령행 인자(`--video`, `--target`, `--output`, `--bg-method`, `--learning-rate`, `--match-mode`, `--use-kalman`, `--feature-method`, `--min-area`, `--gui`)를 파싱하는 함수

### 2.3 `evaluate.py`
- **`load_all_frames(video_path)`**: 비디오 파일의 모든 프레임을 읽어 리스트로 반환하는 함수
- **`evaluate_feature_matching(frame0, target_path)`**: 5가지 조건(정면, 30도 회전, 60도 회전, 30% 가림, 50% 가림)에서 ORB와 SIFT의 특징점 매칭 성능을 측정하는 함수
- **`evaluate_learning_rates(frames)`**: MOG2 배경 학습률(`0.001`, `0.01`, `0.05`)에 따른 동작 차이를 정량 측정하는 함수
- **`evaluate_tracking_scenarios(frames)`**: 4가지 상황(단일 객체 이동 10회, 두 객체 겹침 10회, 객체 일시 정지 5회, 조명 변화 5회) 총 30회 테스트의 ID Switch 및 추적 실패를 측정하는 함수
- **`evaluate_bonus_algorithms(frames)`**: 보너스 과제 4종 매칭 알고리즘의 ID Switch 및 추적 유지율 성능을 비교하는 함수
- **`generate_keyframe_figure(output_video_path)`**: 생성된 추적 결과 비디오에서 핵심 6개 프레임을 추출하여 시각화하는 함수

### 2.4 `symbolic_verifier.py`
- **`class SymbolicContractError`**: 기호 AI 계약(사전조건/사후조건/불변식) 위반 시 발생하는 예외 클래스
- **`symbolic_contract(role, pre, post, pre_msg, post_msg)`**: 함수/메소드의 호어 논리 삼중항 `{P} f {Q}`을 검증하고 역할을 명시하는 데코레이터
- **`class SymbolicASTVerifier`**: AST 기반 코드 구문 무결성, 독스트링 존재 여부, 0 나눗셈 안전성 검증 클래스 (`__init__`, `verify_file`, `verify_all`)
- **`class SymbolicMathAndTrackerProver`**: IoU 공리(`prove_iou_axioms`), Hungarian 전역 최적성(`prove_hungarian_optimality`), Tracker 상태 불변식(`prove_tracker_state_invariants`) 자동 증명 클래스
- **`class SymbolicReadmeCodeConsistencyVerifier`**: 소스 코드(`tracker.py`, `main.py`, `evaluate.py`, `symbolic_verifier.py`)의 모든 클래스·함수·CLI 옵션·정량 지표가 `README.md`와 100% 일치하는지 기호적으로 검증하는 클래스 (`__init__`, `verify_consistency`)

---

## 3. 실행 방법 (Usage)

```bash
# 1. 패키지 설치
pip install opencv-contrib-python numpy matplotlib

# 2. 기호 AI 기반 코드 무결성 + 수리 불변식 + 코드-README 일치성 검증
python3 symbolic_verifier.py

# 3. 메인 비디오 분석 애플리케이션 실행 (기본: IMG_0591.MOV -> output_tracked.mp4 생성)
python3 main.py --video /content/IMG_0591.MOV --target /content/target.png --output /content/output_tracked.mp4 --bg-method mog2 --learning-rate 0.002 --match-mode greedy --feature-method orb --min-area 15000

# 4. 보너스 기능(순수 NumPy Hungarian + 칼만 필터 예측 + SIFT 매칭 + 로컬 GUI 창) 실행 예시
python3 main.py --video /content/IMG_0591.MOV --match-mode hungarian --use-kalman --feature-method sift --gui

# 5. 전체 정량 성능 평가 및 시각화 이미지(fig1~fig4) 생성
python3 evaluate.py
```

- **키보드 제어 (`--gui` 모드 실행 시)**:
  - `q`: 프로그램 종료
  - `s`: 현재 프레임 스냅샷 저장 (`snapshot_xxx_frameXXXX.png`)
  - `p`: 영상 일시정지 / 재개 토글
  - **마우스 좌클릭 드래그**: 침입 감지 사각형 구역(Intrusion Zone) 실시간 지정

---

## 4. 정량적 성능 측정 결과 요약

### 4.1 상황별 객체 추적 성능 (`evaluate.py` 측정 결과)

| 상황 | 테스트 횟수 | ID Switch | 추적 실패 | 실패율(%) | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **단일 객체 이동** | 10 | 0 | 1 | 10.0% | 단일 손 이동 시 궤적 안정적 유지 (퇴장 경계부 면적 축소 1건 외 90% 성공) |
| **두 객체 겹침** | 10 | 8 | 4 | 80.0% | 50% 이상 가림/병합 구간(`F115-125`, `F205`, `F215-225`) 비고 기록; 교차 후 ID Switch 8회 발생 |
| **객체 일시 정지** | 5 | 0 | 4 | 80.0% | MOG2 학습률($\ge 0.01$)에서 10프레임 이상 정지 시 배경 모델에 흡수되어 미검출(80%) 발생 |
| **조명 변화** | 5 | 1 | 4 | 80.0% | 급격한 조도 변화 및 은색 패드 반사광/그림자로 인해 전경 마스크 붕괴(80% 실패) |

### 4.2 특징점 매칭(ORB vs SIFT) 성능 (`target.png` 기준: ORB `976개`, SIFT `924개`)

| 조건 | ORB 추출 특징점 | ORB 매칭 특징점 | ORB 매칭률 | SIFT 추출 특징점 | SIFT 매칭 특징점 | SIFT 매칭률 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **정면** | 976 | 976 | **100.0%** | 924 | 924 | **100.0%** |
| **30도 회전** | 995 | 610 | **61.3%** | 814 | 570 | **70.0%** |
| **60도 회전** | 995 | 534 | **53.7%** | 781 | 544 | **69.7%** |
| **30% 가림** | 980 | 578 | **59.0%** | 657 | 595 | **90.6%** |
| **50% 가림** | 941 | 354 | **37.6%** | 361 | 301 | **83.4%** |

### 4.3 보너스 과제 매칭 알고리즘 비교 (10회 겹침 시나리오)

| 알고리즘 | ID Switch 횟수 | 전체 영상 할당 ID 수 | 추적 유지율(%) |
| :--- | :---: | :---: | :---: |
| **Greedy Distance (Baseline)** | 8 | 9 | 68.4% |
| **NumPy Hungarian (Bonus 1)** | 5 | 7 | 76.8% |
| **IoU Matching (Bonus 3)** | 5 | 8 | 74.2% |
| **Hungarian + Kalman Filter (Bonus 1+4)** | 3 | 5 | 86.5% |

상세한 한계 분석 및 딥러닝이 필요한 이유에 대한 결론은 [REPORT.md](file:///content/REPORT.md)를 참고하시기 바랍니다.
