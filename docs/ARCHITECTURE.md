# VisionSuiteTrain — 아키텍처

> 설계 출처: 멀티에이전트 설계 하네스(6영역 병렬 연구 → 합성). VisionSuiteCore(VSC) 코드 근거.

## 1. 정체성: VSC의 "생산자(producer)"
VisionSuiteTrain(VST)은 PyTorch 학습 코어로, **VisionSuiteCore가 추론하는 아티팩트를 emit**한다:
- `model.onnx` — 가중치(ONNX, fp32). TRT 빌드는 VSC가 수행.
- `model.yaml` — VSC 런타임 config(ModelConfig 매핑).
- `<NAME>_manifest.yaml` — I/O 계약(environment/inputs/outputs/preprocessing/task).

두 repo는 **코드가 아니라 "export 컨트랙트"로만 연결**된다.

## 2. 통일 컨트랙트 (canonical 3-tensor)
모든 arch의 ONNX 출력을 VSC 단일 핸들러가 먹는 형태로 통일한다:

| task | ONNX 출력 | 규약 |
|---|---|---|
| **DET** | `[B, 4+NC, A]` channel-first | ch0-3=cx,cy,w,h(letterbox 입력-px), ch4..=per-class **sigmoid 적용** 점수, objectness 없음, **NMS/top-k export 금지**(VSC가 conf+NMS) |
| **CLS** | `[B, NC]` (=`[B,NC,1,1]` 동일취급) | **raw logit** export, `cls_activation`은 manifest에 명시(이중 softmax 회피) |
| **SEG** | `[B, C, H, W]` channel-first | argmax는 VSC C++ 산출. multi_channel=채널값 [0,1](그래프에 sigmoid/softmax) / one_channel=raw score map. index0=background |

⚠ **ONNX 입력은 정규화 안 된 NCHW float32** — 그래프 내부 normalize 금지(VSC가 BGR→RGB + /255 + 표준화 수행).
⚠ `NC == len(names) == manifest.label_map 길이 == decision.names 길이` (단일 진실원 = `dataset.names[]`, 0-base, 순서=학습 라벨 id).

## 3. 구조 (어댑터 registry)
```
(task, arch) ──registry──▶ BaseTrainer 어댑터  ─prepare_data→ train → export_to_vsc
```
- **BaseTrainer**(ABC): `prepare_data()` / `train()` / `export_to_vsc()` 3-메서드 강제.
- **registry**: `@register_trainer(task, *archs)` 자동 등록 → `build_trainer(cfg)`.
- **VscExporter**: ONNX export + manifest/model.yaml writer + **fail-fast 정합 assert**.
  manifest↔model.yaml 입력 일치 · `label_map keys==range(len(names))` · `NC(onnx introspect)==len(names)`,
  그리고 **실제 ONNX 아티팩트** 대조(입력 C/H/W · opset · io 이름 — `_assert_against_onnx`).
- **데이터**: labelme JSON → **IR**(Sample/Region) Reader + task별 Writer(yolo_txt/mask_png/imagefolder). format↔task 직교 분리.

## 4. 어댑터별 정합 (1차/2차)
| arch | task | 정합 | 범위 |
|---|---|---|---|
| **yolov8_hbb** | det | **drop-in** (ultralytics export `nms=False` → `[1,4+nc,A]` 정확 일치) | 1차 |
| **efficientnet** | cls | **drop-in** (timm `[B,NC]` logit) | 1차 |
| **deeplab3pp** | seg | 작음 (단일출력 Wrap + 채널 [0,1] 활성화) | 1차 |
| yolov7_hbb | det | raw v7 export 불일치(anchor-major+obj) → v8 헤드 재학습 또는 변환 | 2차 |
| dfine_hbb | det | **GAP 큼** — DETR 출력(labels/boxes/scores) → VSC **신규 핸들러 필요** | 2차 |
| rfdetr_hbb | det | **GAP 큼** — dets/labels → 그래프 패치 또는 신규 핸들러 | 2차 |

## 5. 디렉토리
```
configs/train|test/*.yaml
src/visionsuitetrain/
  config/{schema,canonical}.py   registry.py
  trainers/{base,yolo_hbb,efficientnet,deeplab}.py
  data/{ir,validate,split}.py  data/readers/labelme.py  data/writers/{yolo_txt,mask_png,imagefolder}.py
  export/{base_exporter,manifest,model_yaml,onnx_io}.py
  cli.py
tests/
```

## 6. 풀 루프
```
labelme → VST.train → VST.export(onnx+model.yaml+manifest)
   → VSC: TRT 빌드 → 추론 / Shadow A·B / threshold_sweep → 배포
   → 미검·과검 데이터 재라벨 → 재학습
```

## 7. 미결 (open questions — 진행하며 확정)
- yolov7/OBB/D-FINE/RF-DETR 2차 우선순위 (D-FINE/RF-DETR은 VSC 신규 핸들러 선결).
- CLS 정규화: timm 변종 mean/std 분기(imagenet_std vs global_std).
- SEG export 모드 기본값(multi_channel sigmoid/softmax vs one_channel).
- 전처리 보간 byte-identical vs 근사.
- dynamic batch export 기본값.
- SEG 겹침(그리기 순서) 우선순위 정책 (circle/rect/polygon 렌더·circle aabb 는 반영됨).
