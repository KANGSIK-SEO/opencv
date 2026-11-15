"""
tracker.py - 고전적 컴퓨터 비전 기반 다중 객체 추적기 (Multi-Object Tracker) + 기호 AI(Symbolic AI) 계약 검증 적용

구현 기능 요약:
1. 필수 요구사항:
   - 이전 프레임과 현재 프레임 객체 중심점 간 유클리드 거리(Euclidean Distance) 계산
   - 가장 가까운 객체끼리 동일 ID로 매칭 (Greedy Distance Matching)
   - 일정 프레임(max_disappeared) 이상 등장하지 않는 객체 ID 자동 삭제
   - 새로 등장한 객체에 새로운 고유 ID 부여
   - 각 객체의 이동 경로(Trajectory) 저장 및 관리
2. 보너스 과제:
   - [보너스 1] scipy 없이 순수 NumPy만으로 구현한 Hungarian Algorithm (Munkres 최적 할당 알고리즘)
   - [보너스 3] 중심점 거리 대신 바운딩 박스 IoU(Intersection over Union) 기반 매칭
   - [보너스 4] cv2.KalmanFilter를 활용한 객체 다음 위치 예측 및 예측값 기반 매칭
3. 기호 AI (Symbolic AI) 적용:
   - 호어 논리(Hoare Logic: {Precondition} Method {Postcondition}) 기반 @symbolic_contract 데코레이터와
     클래스 불변식(_assert_symbolic_invariants)을 통해 행렬 차원, 거리 비음수성, 단사(1:1) 매칭,
     사전 키 집합 일치성 오류를 원천 차단한다.
"""

from collections import OrderedDict, deque
from typing import Dict, List, Tuple
import cv2
import numpy as np

from symbolic_verifier import symbolic_contract, SymbolicContractError


@symbolic_contract(
    role="두 바운딩 박스(x, y, w, h) 간의 교집합 대비 합집합 비율(IoU)을 계산하는 함수",
    pre=lambda boxA, boxB: len(boxA) == 4 and len(boxB) == 4 and boxA[2] >= 0 and boxA[3] >= 0 and boxB[2] >= 0 and boxB[3] >= 0,
    post=lambda res: isinstance(res, float) and (0.0 <= res <= 1.0 + 1e-9),
    pre_msg="바운딩 박스는 (x, y, w, h) 4원소 튜플이어야 하며 너비(w)와 높이(h)는 0 이상이어야 합니다.",
    post_msg="IoU 결과값은 반드시 [0.0, 1.0] 범위의 실수여야 합니다.",
)
def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:  # "두 바운딩 박스 간의 중첩도(IoU)를 0~1 사이 실수로 계산하는 함수"
    """두 바운딩 박스 (x, y, w, h) 간의 IoU(Intersection over Union)를 계산하는 함수"""
    # 교집합 사각형의 좌상단(xA, yA) 및 우하단(xB, yB) 좌표 계산
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # 교집합 영역의 너비와 높이 계산 (겹치지 않으면 0)
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = float(inter_w * inter_h)

    # 교집합 면적이 0 이하이면 나눗셈 없이 즉시 0.0 반환 (0 나눗셈 방지)
    if inter_area <= 0.0:
        return 0.0

    # 각 박스의 면적과 합집합 면적 계산
    areaA = float(boxA[2] * boxA[3])
    areaB = float(boxB[2] * boxB[3])
    union_area = areaA + areaB - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(np.clip(inter_area / union_area, 0.0, 1.0))


@symbolic_contract(
    role="scipy 없이 순수 NumPy만을 사용하여 비용 행렬의 전역 최적 1:1 할당 쌍을 구하는 Hungarian 알고리즘 함수",
    pre=lambda cost_matrix: isinstance(cost_matrix, np.ndarray) and cost_matrix.ndim == 2 and np.all(np.isfinite(cost_matrix)),
    post=lambda res: isinstance(res, list) and len({r for r, _ in res}) == len(res) and len({c for _, c in res}) == len(res),
    pre_msg="비용 행렬(cost_matrix)은 유한한 실수로 구성된 2차원 NumPy 배열이어야 합니다.",
    post_msg="할당 결과는 행과 열 인덱스가 각각 중복되지 않는 단사(1:1) 매칭 리스트여야 합니다.",
)
def hungarian_algorithm_numpy(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:  # "순수 NumPy 기반 Hungarian(Kuhn-Munkres) 최적 할당 함수"
    """[보너스 과제 1] scipy 없이 순수 NumPy만으로 구현한 Hungarian Algorithm (전역 최소 비용 1:1 매칭 함수)"""
    # 빈 행렬이 입력된 경우 빈 매칭 리스트 반환
    if cost_matrix.size == 0:
        return []

    n_rows, n_cols = cost_matrix.shape
    transposed = False

    # 알고리즘은 행의 개수 <= 열의 개수(n_rows <= n_cols)를 가정하므로, 행이 더 많으면 전치(Transpose) 수행
    if n_rows > n_cols:
        cost = cost_matrix.T.copy().astype(np.float64)
        n_rows, n_cols = cost.shape
        transposed = True
    else:
        cost = cost_matrix.copy().astype(np.float64)

    # 쌍대 변수(Dual potentials) u, v 및 매칭 경로 배열 p, way 초기화 (1-based 인덱싱 사용)
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    # 각 행 i에 대해 최단 증가 경로(Augmenting Path) 탐색
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            # 아직 방문하지 않은 열 j에 대해 여유값(Slack) 갱신
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            # 쌍대 변수(Potential) 업데이트
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        # 역추적(Backtracking)을 통해 매칭 경로 반전
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # 1-based 인덱스를 0-based (row, col) 쌍으로 변환
    assignments: List[Tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            r = int(p[j] - 1)
            c = int(j - 1)
            if transposed:
                assignments.append((c, r))
            else:
                assignments.append((r, c))

    assignments.sort(key=lambda pair: pair[0])
    return assignments


class SingleKalmanFilter:  # "단일 객체의 2차원 위치(x, y)와 속도(vx, vy)를 추정·예측하는 칼만 필터 클래스"
    """[보너스 과제 4] cv2.KalmanFilter를 활용하여 단일 객체의 다음 프레임 위치를 예측하고 보정하는 클래스"""

    def __init__(self, init_x: float, init_y: float):  # "초기 중심점 좌표(init_x, init_y)로 4차원 상태 칼만 필터를 초기화하는 생성자 메소드"
        """초기 중심점 좌표(init_x, init_y)로 칼만 필터의 상태 행렬과 공분산 행렬을 초기화하는 메소드"""
        # 4차원 상태 벡터 [x, y, vx, vy]^T 및 2차원 관측 벡터 [x, y]^T 설정
        self.kf = cv2.KalmanFilter(4, 2)

        # 상태 전이 행렬 F (등속 직선 운동 모델: x_t = x_{t-1} + vx, y_t = y_{t-1} + vy)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # 관측 행렬 H (상태 벡터에서 위치 x, y만 관측함)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        # 프로세스 노이즈 공분산(Q), 관측 노이즈 공분산(R), 사후 오차 공분산(P) 설정
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array(
            [[init_x], [init_y], [0.0], [0.0]], dtype=np.float32
        )

    @symbolic_contract(
        role="칼만 필터의 상태 전이 모델을 통해 현재 프레임의 예상 중심점 (px, py)를 예측하는 메소드",
        post=lambda res: isinstance(res, tuple) and len(res) == 2 and all(isinstance(v, int) for v in res),
    )
    def predict(self) -> Tuple[int, int]:  # "등속 운동 모델을 바탕으로 객체의 다음 위치(x, y)를 예측하는 메소드"
        """객체의 이전 상태와 속도를 기반으로 현재 프레임에서의 예상 중심점 좌표를 반환하는 메소드"""
        pred = self.kf.predict()
        return int(round(float(pred[0, 0]))), int(round(float(pred[1, 0])))

    @symbolic_contract(
        role="실제 검출된 중심점 좌표(x, y)를 관측값으로 입력받아 칼만 필터 상태를 보정(Update)하는 메소드",
        post=lambda res: isinstance(res, tuple) and len(res) == 2 and all(isinstance(v, int) for v in res),
    )
    def correct(self, x: float, y: float) -> Tuple[int, int]:  # "실제 관측된 중심점(x, y)으로 칼만 필터 상태를 보정하는 메소드"
        """실제 관측된 중심점 (x, y)를 반영하여 칼만 필터의 상태 벡터(위치 및 속도)를 갱신하는 메소드"""
        measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
        est = self.kf.correct(measurement)
        return int(round(float(est[0, 0]))), int(round(float(est[1, 0])))


class Tracker:  # "프레임 간 객체 거리/IoU/칼만 예측을 계산하여 고유 ID를 부여·유지·삭제하고 이동 궤적을 관리하는 다중 객체 추적 클래스"
    """
    다중 객체 추적 클래스 (Multi-Object Tracker)
    - 객체 ID 등록(register), 미등장 객체 삭제(deregister), 거리/IoU/Hungarian/Kalman 기반 매칭(update),
      이동 궤적(trajectories) 저장 및 기호 AI 상태 불변식 검증(_assert_symbolic_invariants)을 수행한다.
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        max_distance: float = 220.0,
        min_iou: float = 0.05,
        max_trajectory_len: int = 90,
        mode: str = "greedy",
        use_kalman: bool = False,
    ):  # "추적기의 임계값(소실 허용 프레임, 최대 거리, 최소 IoU) 및 매칭 모드(greedy/hungarian/iou, 칼만 필터)를 초기화하는 생성자 메소드"
        """Tracker 인스턴스의 상태 사전들과 하이퍼파라미터를 초기화하는 생성자 메소드"""
        # 다음에 새로 등장할 객체에 부여할 고유 ID 번호 (1부터 시작)
        self.next_object_id: int = 1

        # ID별 현재 중심점 (cx, cy), 바운딩 박스 (x, y, w, h), 미검출 프레임 수, 이동 궤적(deque) 저장소
        self.objects: "OrderedDict[int, Tuple[int, int]]" = OrderedDict()
        self.bboxes: "OrderedDict[int, Tuple[int, int, int, int]]" = OrderedDict()
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()
        self.trajectories: "OrderedDict[int, deque]" = OrderedDict()

        # 칼만 필터 활성화 시 ID별 칼만 필터 인스턴스 및 예측 좌표 저장소
        self.kalman_filters: Dict[int, SingleKalmanFilter] = {}
        self.predicted_centers: Dict[int, Tuple[int, int]] = {}

        # 하이퍼파라미터 설정
        self.max_disappeared = int(max_disappeared)
        self.max_distance = float(max_distance)
        self.min_iou = float(min_iou)
        self.max_trajectory_len = int(max_trajectory_len)
        self.mode = mode.lower().strip()
        self.use_kalman = bool(use_kalman)

        # 기호 AI 사전 검증: 허용된 매칭 모드인지 확인
        if self.mode not in ("greedy", "hungarian", "iou"):
            raise SymbolicContractError(f"지원하지 않는 매칭 모드입니다: {self.mode}")

    def _assert_symbolic_invariants(self) -> None:  # "기호 AI(Symbolic AI) 상태 불변식을 검사하여 내부 데이터 구조의 무결성을 보장하는 메소드"
        """[기호 AI 불변식 검증] Tracker 내부 사전들의 키 집합 일치성 및 미검출 카운터 경계 조건을 검증하는 메소드"""
        keys_obj = set(self.objects.keys())
        keys_box = set(self.bboxes.keys())
        keys_dis = set(self.disappeared.keys())
        keys_trj = set(self.trajectories.keys())

        # 불변식 1: 4개 핵심 상태 사전의 ID 키 집합은 언제나 완벽히 동일해야 함
        if not (keys_obj == keys_box == keys_dis == keys_trj):
            raise SymbolicContractError(
                "Tracker 내부 상태 사전(objects, bboxes, disappeared, trajectories)의 키 집합이 일치하지 않습니다."
            )

        # 불변식 2: 모든 활성 ID의 미검출 프레임 수는 0 이상 max_disappeared 이하여야 하며, 궤적은 최소 1개 이상의 좌표를 가져야 함
        for oid, d_cnt in self.disappeared.items():
            if not (0 <= d_cnt <= self.max_disappeared):
                raise SymbolicContractError(
                    f"ID {oid}의 disappeared 카운터({d_cnt})가 허용 범위 [0, {self.max_disappeared}]를 벗어났습니다."
                )
            if len(self.trajectories[oid]) == 0:
                raise SymbolicContractError(f"ID {oid}의 이동 궤적(trajectory)이 비어 있습니다.")

    @symbolic_contract(
        role="새로 감지된 바운딩 박스에 새로운 고유 ID를 부여하고 상태 사전에 등록하는 메소드",
        pre=lambda self, bbox: len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0,
        post=lambda res: isinstance(res, int) and res >= 1,
    )
    def register(self, bbox: Tuple[int, int, int, int]) -> int:  # "신규 등장 객체에 고유 ID를 부여하고 초기 중심점과 궤적을 등록하는 메소드"
        """새로운 객체를 등록하고 고유 ID를 부여하는 메소드"""
        x, y, w, h = bbox
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))

        obj_id = self.next_object_id
        self.objects[obj_id] = (cx, cy)
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.trajectories[obj_id] = deque(maxlen=self.max_trajectory_len)
        self.trajectories[obj_id].append((cx, cy))

        # 칼만 필터 사용 시 새 객체 전용 칼만 필터 생성
        if self.use_kalman:
            self.kalman_filters[obj_id] = SingleKalmanFilter(cx, cy)
            self.predicted_centers[obj_id] = (cx, cy)

        self.next_object_id += 1
        self._assert_symbolic_invariants()
        return obj_id

    @symbolic_contract(
        role="일정 프레임 이상 미검출된 객체의 ID와 관련 상태 데이터를 모든 사전에서 완전히 삭제하는 메소드",
        pre=lambda self, object_id: object_id in self.objects,
    )
    def deregister(self, object_id: int) -> None:  # "장기간 미등장 객체의 ID와 궤적 정보를 메모리에서 삭제하는 메소드"
        """일정 프레임 이상 등장하지 않은 객체 ID를 삭제하는 메소드"""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.trajectories[object_id]
        self.kalman_filters.pop(object_id, None)
        self.predicted_centers.pop(object_id, None)
        self._assert_symbolic_invariants()

    @symbolic_contract(
        role="기존 추적 객체들의 중심점(또는 칼만 예측 위치)과 현재 프레임 검출 중심점들 간의 유클리드 거리 행렬을 계산하는 메소드",
        pre=lambda self, object_ids, input_centroids: len(object_ids) > 0 and input_centroids.ndim == 2 and input_centroids.shape[1] == 2,
        post=lambda res: isinstance(res, np.ndarray) and res.ndim == 2 and np.all(res >= 0.0),
    )
    def _compute_distance_matrix(
        self,
        object_ids: List[int],
        input_centroids: np.ndarray,
    ) -> np.ndarray:  # "이전 프레임 객체들과 현재 프레임 객체들 간의 유클리드 거리 행렬(N x M)을 계산하는 내부 메소드"
        """기존 객체 위치(또는 칼만 필터 예측 위치)와 현재 프레임 검출 중심점 간 유클리드 거리 행렬을 계산하는 메소드"""
        ref_centroids = []
        for obj_id in object_ids:
            if self.use_kalman and obj_id in self.kalman_filters:
                pred_pt = self.kalman_filters[obj_id].predict()
                self.predicted_centers[obj_id] = pred_pt
                ref_centroids.append(pred_pt)
            else:
                ref_centroids.append(self.objects[obj_id])

        # 브로드캐스팅을 이용해 (N, 1, 2) - (1, M, 2) -> (N, M) 유클리드 노름 계산
        ref_arr = np.array(ref_centroids, dtype=np.float64)
        diff = ref_arr[:, np.newaxis, :] - input_centroids[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        return dist_matrix

    @symbolic_contract(
        role="현재 프레임의 검출 바운딩 박스 리스트를 입력받아 ID 매칭, 궤적 갱신, 신규 등록, 소실 삭제를 수행하는 메인 추적 메소드",
        pre=lambda self, bboxes: isinstance(bboxes, list) and all(len(b) == 4 and b[2] > 0 and b[3] > 0 for b in bboxes),
        post=lambda res: isinstance(res, dict),
    )
    def update(
        self, bboxes: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Dict[str, object]]:  # "매 프레임 검출된 객체 박스들을 기존 ID와 매칭하고 추적 상태를 갱신하여 반환하는 핵심 메소드"
        """현재 프레임에서 검출된 바운딩 박스 리스트를 입력받아 객체 ID 매칭 및 상태 갱신을 수행하는 메소드"""
        # 1. 현재 프레임에 감지된 객체가 하나도 없는 경우: 모든 기존 객체의 disappeared 카운터 1 증가
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.use_kalman and object_id in self.kalman_filters:
                    pred_pt = self.kalman_filters[object_id].predict()
                    self.predicted_centers[object_id] = pred_pt
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            self._assert_symbolic_invariants()
            return self._build_output()

        # 2. 현재 프레임 검출 박스들의 중심점 (cx, cy) 배열 계산
        input_centroids = np.zeros((len(bboxes), 2), dtype=np.float64)
        for i, (x, y, w, h) in enumerate(bboxes):
            input_centroids[i] = (x + w / 2.0, y + h / 2.0)

        # 3. 현재 추적 중인 객체가 하나도 없는 경우: 모든 검출 박스를 신규 ID로 등록
        if len(self.objects) == 0:
            for bbox in bboxes:
                self.register(bbox)
            self._assert_symbolic_invariants()
            return self._build_output()

        # 4. 기존 추적 객체들과 현재 프레임 검출 객체들 간의 매칭 수행
        object_ids = list(self.objects.keys())
        used_rows = set()
        used_cols = set()
        matched_pairs: List[Tuple[int, int]] = []

        if self.mode == "iou":
            # [보너스 3] 바운딩 박스 IoU 기반 매칭
            iou_matrix = np.zeros((len(object_ids), len(bboxes)), dtype=np.float64)
            dist_matrix = self._compute_distance_matrix(object_ids, input_centroids)
            for r, obj_id in enumerate(object_ids):
                prev_box = self.bboxes[obj_id]
                for c, cur_box in enumerate(bboxes):
                    iou_matrix[r, c] = compute_iou(prev_box, cur_box)

            # IoU가 높은 순서대로 정렬하여 매칭하되, 빠른 이동으로 IoU가 낮을 때는 거리 임계값을 보조로 활용
            flat_indices = np.argsort(-iou_matrix, axis=None)
            for idx in flat_indices:
                r = int(idx // len(bboxes))
                c = int(idx % len(bboxes))
                if r in used_rows or c in used_cols:
                    continue
                if iou_matrix[r, c] >= self.min_iou or dist_matrix[r, c] <= self.max_distance * 0.75:
                    used_rows.add(r)
                    used_cols.add(c)
                    matched_pairs.append((r, c))

        elif self.mode == "hungarian":
            # [보너스 1] 순수 NumPy Hungarian Algorithm 기반 전역 최소 거리 할당
            dist_matrix = self._compute_distance_matrix(object_ids, input_centroids)
            candidates = hungarian_algorithm_numpy(dist_matrix)
            for r, c in candidates:
                if dist_matrix[r, c] <= self.max_distance:
                    used_rows.add(r)
                    used_cols.add(c)
                    matched_pairs.append((r, c))

        else:
            # [필수] 유클리드 거리 기반 Greedy 최근접 매칭
            dist_matrix = self._compute_distance_matrix(object_ids, input_centroids)
            rows = dist_matrix.min(axis=1).argsort()
            cols = dist_matrix.argmin(axis=1)[rows]

            for r, c in zip(rows, cols):
                r_int, c_int = int(r), int(c)
                if r_int in used_rows or c_int in used_cols:
                    continue
                if dist_matrix[r_int, c_int] > self.max_distance:
                    continue
                used_rows.add(r_int)
                used_cols.add(c_int)
                matched_pairs.append((r_int, c_int))

        # 5. 매칭에 성공한 기존 객체의 위치, 바운딩 박스, 칼만 필터 상태, 이동 궤적 업데이트
        for r, c in matched_pairs:
            obj_id = object_ids[r]
            cx, cy = int(round(input_centroids[c, 0])), int(round(input_centroids[c, 1]))
            if self.use_kalman and obj_id in self.kalman_filters:
                self.kalman_filters[obj_id].correct(cx, cy)
            self.objects[obj_id] = (cx, cy)
            self.bboxes[obj_id] = bboxes[c]
            self.disappeared[obj_id] = 0
            self.trajectories[obj_id].append((cx, cy))

        # 6. 이번 프레임에서 매칭되지 못한 기존 객체들의 미검출 카운터 증가 및 임계값 초과 시 삭제
        unused_rows = set(range(len(object_ids))) - used_rows
        for r in unused_rows:
            obj_id = object_ids[r]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        # 7. 기존 객체와 매칭되지 않은 새로운 검출 박스들에 새 고유 ID 부여
        unused_cols = set(range(len(bboxes))) - used_cols
        for c in sorted(unused_cols):
            self.register(bboxes[c])

        # 8. 기호 AI 불변식 최종 점검 후 결과 반환
        self._assert_symbolic_invariants()
        return self._build_output()

    def _build_output(self) -> Dict[int, Dict[str, object]]:  # "현재 활성 상태인 모든 객체의 ID별 좌표·박스·궤적 정보를 딕셔너리로 구성하여 반환하는 내부 메소드"
        """현재 추적 중인 모든 객체의 상태 정보를 표준화된 딕셔너리 형태로 반환하는 메소드"""
        result: Dict[int, Dict[str, object]] = {}
        for obj_id in self.objects:
            result[obj_id] = {
                "centroid": self.objects[obj_id],
                "bbox": self.bboxes[obj_id],
                "disappeared": self.disappeared[obj_id],
                "trajectory": list(self.trajectories[obj_id]),
                "predicted": self.predicted_centers.get(obj_id, self.objects[obj_id]),
            }
        return result
