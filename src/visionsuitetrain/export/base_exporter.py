"""VscExporter — ONNX + manifest + model.yaml 를 쓰고 export 종료 시 fail-fast 정합 assert.

정합 위반은 export 를 중단(런타임 로드 에러를 선제 차단). 어댑터는 ONNX 만 만들고
이 클래스에 output_shape 를 넘겨 manifest/model.yaml 동시 생성 + 검증을 위임한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from ..config.schema import TrainConfig
from .manifest import build_manifest
from .model_yaml import build_model_yaml


class VscExporter:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.names = list(cfg.dataset.names)

    def write(self, onnx_path: str | Path, output_shape: list, *,
              weights_name: str = "model.onnx",
              nms_conf_vector: Optional[list[float]] = None,
              thresholds: Optional[dict[str, float]] = None) -> dict[str, Path]:
        out_dir = Path(self.cfg.run.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = build_manifest(self.cfg, output_shape, thresholds=thresholds)
        model_yaml = build_model_yaml(self.cfg, weights=weights_name,
                                      nms_conf_vector=nms_conf_vector)
        # 실제 ONNX 아티팩트까지 대조(입력 C/H/W·opset·io 이름)
        self.assert_consistency(manifest, model_yaml, output_shape, onnx_path=onnx_path)

        name = self.cfg.run.name
        man_p = out_dir / f"{name}_manifest.yaml"
        my_p = out_dir / "model.yaml"
        man_p.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
        my_p.write_text(yaml.safe_dump(model_yaml, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        return {"onnx": Path(onnx_path), "model_yaml": my_p, "manifest": man_p}

    # ── fail-fast 정합 검증 ──
    def assert_consistency(self, manifest: dict, model_yaml: dict, output_shape: list,
                           onnx_path: Optional[str | Path] = None) -> None:
        cfg, names = self.cfg, self.names
        in_name = cfg.export.io_names.input

        ishape = manifest["inputs"][in_name]["shape"]      # [1, C, H, W]
        m_in = model_yaml["model"]["input"]
        if ishape[2:] != [m_in["h"], m_in["w"]]:
            raise AssertionError(
                f"input shape 불일치: manifest {ishape[2:]} vs model.yaml [{m_in['h']},{m_in['w']}]")

        lm = manifest["task"]["label_map"]
        if sorted(lm.keys()) != list(range(len(names))):
            raise AssertionError(f"label_map 키가 0..{len(names)-1} 아님: {sorted(lm.keys())}")

        if manifest["environment"]["opset"] != cfg.export.opset:
            raise AssertionError("manifest opset 불일치")

        nc = self._infer_nc(output_shape)
        if nc is not None and nc != len(names):
            raise AssertionError(f"ONNX 출력 NC({nc}) != len(names)({len(names)})")

        # 실제 ONNX 아티팩트와 cfg 대조(cfg-vs-cfg tautology 보완 — 입력 dim·opset·io 이름)
        if onnx_path is not None:
            self._assert_against_onnx(onnx_path)

    def _assert_against_onnx(self, onnx_path: str | Path) -> None:
        from .onnx_io import (introspect_input_shape, introspect_io_names,
                              introspect_opset)
        cfg = self.cfg
        real_in = introspect_input_shape(onnx_path)        # [N, C, H, W] 기대
        if len(real_in) == 4:
            for got, exp, nm in ((real_in[1], cfg.export.input.c, "C"),
                                 (real_in[2], cfg.export.input.h, "H"),
                                 (real_in[3], cfg.export.input.w, "W")):
                if got not in (-1, exp):
                    raise AssertionError(f"ONNX 실제 입력 {nm}({got}) != config({exp})")
        real_op = introspect_opset(onnx_path)
        if real_op is not None and real_op != cfg.export.opset:
            raise AssertionError(f"ONNX 실제 opset({real_op}) != config({cfg.export.opset})")
        i_name, o_name = introspect_io_names(onnx_path)
        if i_name not in (None, cfg.export.io_names.input):
            raise AssertionError(f"ONNX 입력명({i_name}) != io_names.input({cfg.export.io_names.input})")
        if o_name not in (None, cfg.export.io_names.output):
            raise AssertionError(f"ONNX 출력명({o_name}) != io_names.output({cfg.export.io_names.output})")

    def _infer_nc(self, output_shape: list) -> Optional[int]:
        if not output_shape:
            return None
        task = self.cfg.task
        try:
            if task == "hbbdetection":        # [B, 4+NC, A]
                v = output_shape[1]
                return None if v < 0 else v - 4
            if task == "obbdetection":        # [B, 5+NC, A]
                v = output_shape[1]
                return None if v < 0 else v - 5
            if task == "classification":      # [B, NC] 또는 [B, NC, 1, 1] 동일취급
                v = output_shape[1] if (len(output_shape) > 2
                                        and all(d == 1 for d in output_shape[2:])) \
                    else output_shape[-1]
                return None if v < 0 else v
            if task == "segmentation":        # [B, C, H, W]
                v = output_shape[1]
                return None if v < 0 else v
        except (IndexError, TypeError):
            return None
        return None
