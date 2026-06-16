# VisionSuiteTrain

**통합 학습 코어** — 하나의 config(train/test manifest)로 여러 아키텍처를 학습하고,
[**VisionSuiteCore**](https://github.com/CVKim/VisionSuiteCore) 가 그대로 추론할 수 있는
**ONNX 컨트랙트 + model.yaml manifest** 로 내보낸다.

```
라벨(labelme/coco/yolo) → [Train] 학습 → [Export] VSC 컨트랙트(ONNX + manifest)
        → VisionSuiteCore: .trt 빌드 → 추론/Shadow A·B/Threshold sweep → 배포
        → (피드백: 미검/과검 데이터 재라벨 → 재학습)   ← 풀 MLOps 루프
```

## 설계 원칙
- **Python / PyTorch** (학습 시점). VisionSuiteCore 는 C++/TensorRT 배포 DLL — **런타임이 분리**돼 있고,
  둘은 **코드가 아니라 "export 컨트랙트"** 로 연결된다.
- **재구현하지 않는다.** 성숙한 trainer(ultralytics · timm · smp · D-FINE · RF-DETR)를
  **통합 config + 통합 export** 뒤로 wrapping 한다. 통합의 핵심 가치는 ① config 스키마 ② export 컨트랙트.
- **config-driven**: task/arch/dataset/gpu/hyperparams 를 yaml 로 지정 → 코드 수정 없이 학습.

## 지원 목표 (단계적)
| task | arch (1차) | arch (확장) | export 컨트랙트 |
|---|---|---|---|
| detection | YOLOv8 | YOLOv7 · D-FINE · RF-DETR | `[B, 4+NC, N]` (eYolov8Hbb) |
| classification | EfficientNet | — | `[B, NC, 1, 1]` |
| segmentation | DeepLabV3 | — | `[B, C, H, W]` |

→ detection 은 export 를 동일 컨트랙트로 통일하므로, VisionSuiteCore 의 **단일 핸들러**(`type: yolov8_hbb`)로
  v7/v8/D-FINE/RF-DETR 를 전부 추론 가능.

## 구조 (계획)
```
configs/   train/*.yaml  test/*.yaml          # 학습/평가 manifest
data/      adapters/ (labelme|coco|yolo|mask) # → 통일 Dataset
trainers/  yolo / dfine / rf_detr / timm_cls / smp_seg  # 어댑터(성숙 trainer wrapping)
export/    to_vsc.py                          # arch → VSC ONNX 컨트랙트 + model.yaml
train.py   test.py                            # python train.py --config configs/train/X.yaml --gpus 0,1
```

## 사용 (계획)
```bash
python train.py --config configs/train/det_yolov8.yaml --gpus 0,1
python export.py --ckpt runs/.../best.pt --to vsc --out exports/model_A   # ONNX + model.yaml
# → exports/model_A 를 VisionSuiteCore configs/models 에 넣고 추론
```

## 라이선스 / 작성
- 단독 author. 회사 코드네임·모델·데이터 커밋 금지(`_local_` 프리픽스, gitignore).
- 브랜치: `master`(안정) · `dev`(통합) · `feat/*`(작업) → 검증 후 머지.

> 상태: 초기 스캐폴드 단계. 자세한 설계는 `docs/ARCHITECTURE.md`(예정).
