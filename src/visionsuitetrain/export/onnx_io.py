"""ONNX io rename(data/output) + 출력 shape introspect. (onnx 는 lazy import)"""
from __future__ import annotations

from pathlib import Path


def rename_io(onnx_path: str | Path, input_name: str = "data",
              output_name: str = "output") -> None:
    """ONNX 그래프의 첫 입력/출력 노드명을 VSC 컨트랙트(data/output)로 변경(in-place)."""
    import onnx  # lazy
    m = onnx.load(str(onnx_path))
    g = m.graph
    if g.input:
        old = g.input[0].name
        g.input[0].name = input_name
        for node in g.node:
            node.input[:] = [input_name if x == old else x for x in node.input]
    if g.output:
        old = g.output[0].name
        g.output[0].name = output_name
        for node in g.node:
            node.output[:] = [output_name if x == old else x for x in node.output]
    onnx.save(m, str(onnx_path))


def introspect_output_shape(onnx_path: str | Path, output_index: int = 0) -> list:
    """첫 출력 텐서의 shape([d0,d1,...]) — symbolic dim 은 -1. NC/A 검증용."""
    import onnx  # lazy
    m = onnx.load(str(onnx_path))
    out = m.graph.output[output_index]
    dims = out.type.tensor_type.shape.dim
    shape = []
    for d in dims:
        shape.append(int(d.dim_value) if d.HasField("dim_value") else -1)
    return shape
