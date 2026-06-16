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


def introspect_input_shape(onnx_path: str | Path, input_index: int = 0) -> list:
    """첫 입력 텐서의 shape([d0,d1,...]) — symbolic dim 은 -1. 실제 C/H/W 검증용."""
    import onnx  # lazy
    m = onnx.load(str(onnx_path))
    inp = m.graph.input[input_index]
    dims = inp.type.tensor_type.shape.dim
    return [int(d.dim_value) if d.HasField("dim_value") else -1 for d in dims]


def introspect_opset(onnx_path: str | Path) -> int | None:
    """실제 ONNX 의 default(ai.onnx) opset 버전. config opset 과 대조용."""
    import onnx  # lazy
    m = onnx.load(str(onnx_path))
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            return int(op.version)
    return int(m.opset_import[0].version) if m.opset_import else None


def ensure_channel_first_det(onnx_path: str | Path, output_index: int = 0,
                             force: bool = False) -> bool:
    """det 3D 출력을 channel-first([1,4+NC,A])로 Transpose(perm=[0,2,1]) 삽입(in-place).

    DETR(ultralytics RT-DETR)는 query-major([1,A,4+NC], dim 이 symbolic 일 수 있음)로 export →
    VSC yolov8_hbb 컨트랙트([1,4+NC,A])에 맞춤. force=True 면 무조건 transpose(DETR 어댑터가 호출),
    그 외엔 concrete dims 에서 dims[1]>dims[2] 일 때만. 삽입 후 shape inference 로 출력 dim 재계산.
    """
    import onnx  # lazy
    from onnx import helper
    m = onnx.load(str(onnx_path))
    out = m.graph.output[output_index]
    sh = out.type.tensor_type.shape
    dims = [d.dim_value if d.HasField("dim_value") else -1 for d in sh.dim]
    if len(dims) != 3:
        return False
    if not force and not (dims[1] > dims[2] >= 0):
        return False                       # heuristic: concrete query-major 만
    old = out.name
    src = old + "_pretp"
    for node in m.graph.node:
        node.output[:] = [src if x == old else x for x in node.output]
    m.graph.node.append(helper.make_node("Transpose", [src], [old], perm=[0, 2, 1]))
    # 출력 dim: concrete 면 [d0,d2,d1]로 교환, symbolic 이면 clear(rank3 유지). shape-infer 미사용(native segfault 회피)
    if all(d >= 0 for d in dims):
        for i, v in enumerate((dims[0], dims[2], dims[1])):
            sh.dim[i].Clear(); sh.dim[i].dim_value = v
    else:
        for d in sh.dim:
            d.Clear()
    onnx.save(m, str(onnx_path))
    return True


def introspect_io_names(onnx_path: str | Path) -> tuple[str | None, str | None]:
    """(첫 입력명, 첫 출력명) — io_names(data/output) 정합 검증용."""
    import onnx  # lazy
    m = onnx.load(str(onnx_path))
    in_name = m.graph.input[0].name if m.graph.input else None
    out_name = m.graph.output[0].name if m.graph.output else None
    return in_name, out_name
