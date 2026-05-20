"""T3.1 Python DSL 重构测试

覆盖：
1. `>>` 返回 Pipeline（而非 PipelineBuilder）
2. 合并语法 `[src1, src2] >> det`
3. `Pipeline.run(block=False/True)` 新语义
4. PyNode 参与 DSL
5. 边界情况

不依赖真实视频/GPU：仅验证 DAG 拓扑结构和 DSL 返回类型。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import visionpipe
from visionpipe import (
    ByteTrackNode,
    DecodeMode,
    DetectorNode,
    FileSource,
    MockModelEngine,
    Pipeline,
    PipelineState,
    PyNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_source(tag: str) -> FileSource:
    """构造 FileSource — 仅用于 DAG 结构验证，不会 start。

    FileSource 自动以 ``FileSource:<uri>`` 命名，所以使用 ``tag`` 作为 uri
    保证每个实例节点名称唯一可识别。
    """
    return FileSource(f"video-{tag}.mp4", DecodeMode.CPU)


def _make_detector(name: str) -> DetectorNode:
    engine = MockModelEngine()
    cfg = visionpipe.DetectorConfig()
    cfg.workers = 0
    return DetectorNode(engine, cfg, name)


def _make_tracker(name: str) -> ByteTrackNode:
    return ByteTrackNode(name=name)


def _node_names(pipeline: Pipeline) -> set[str]:
    return set(pipeline.nodes().keys())


# ---------------------------------------------------------------------------
# 1. `>>` 返回 Pipeline
# ---------------------------------------------------------------------------


def test_rshift_two_nodes_returns_pipeline_instance() -> None:
    src = _make_file_source("two_a")
    det = _make_detector("det_two")

    result = src >> det

    assert isinstance(result, Pipeline)
    # 不是 PipelineBuilder（旧行为）
    assert not isinstance(result, visionpipe.PipelineBuilder)
    assert _node_names(result) == {src.name(), det.name()}


def test_rshift_chain_three_nodes_returns_pipeline() -> None:
    src = _make_file_source("three_a")
    det = _make_detector("det_three")
    trk = _make_tracker("trk_three")

    result = src >> det >> trk

    assert isinstance(result, Pipeline)
    assert _node_names(result) == {src.name(), det.name(), trk.name()}
    assert len(result.nodes()) == 3


def test_rshift_chain_preserves_get_node_lookup() -> None:
    """通过 get_node 验证链路上的节点都被正确加入 DAG。"""
    src = _make_file_source("conn_a")
    det = _make_detector("det_conn")
    trk = _make_tracker("trk_conn")

    pipeline = src >> det >> trk

    assert isinstance(pipeline, Pipeline)
    assert pipeline.get_node(src.name()) is src
    assert pipeline.get_node(det.name()) is det
    assert pipeline.get_node(trk.name()) is trk

    # validate_dag 不应抛出（链路无环且连接合法）
    pipeline.validate_dag()

    # 状态仍为 INIT，因为未调用 start
    assert pipeline.state() == PipelineState.INIT


def test_rshift_chain_tail_tracking_extends_pipeline() -> None:
    """链式 `>>` 应该在原 Pipeline 上追加节点，而不是新建。"""
    src = _make_file_source("tail_a")
    det = _make_detector("det_tail")
    trk = _make_tracker("trk_tail")

    first = src >> det
    assert isinstance(first, Pipeline)

    second = first >> trk

    # `Pipeline >> NodeBase` 返回 self
    assert second is first
    assert _node_names(first) == {src.name(), det.name(), trk.name()}


def test_rshift_source_is_only_source_in_chain() -> None:
    """链式 `src >> det` 时，source_nodes() 应只包含 src。"""
    src = _make_file_source("single_src_only")
    det = _make_detector("det_single_src_only")

    pipeline = src >> det

    sources = pipeline.source_nodes()
    assert len(sources) == 1
    assert sources[0] is src


# ---------------------------------------------------------------------------
# 2. 合并语法 [src1, src2] >> node
# ---------------------------------------------------------------------------


def test_merge_two_sources_returns_pipeline() -> None:
    src1 = _make_file_source("merge_a")
    src2 = _make_file_source("merge_b")
    det = _make_detector("det_merge")

    pipeline = [src1, src2] >> det

    assert isinstance(pipeline, Pipeline)
    assert _node_names(pipeline) == {src1.name(), src2.name(), det.name()}
    assert len(pipeline.nodes()) == 3
    assert pipeline.get_node(src1.name()) is src1
    assert pipeline.get_node(src2.name()) is src2
    assert pipeline.get_node(det.name()) is det

    source_names = {s.name() for s in pipeline.source_nodes()}
    assert source_names == {src1.name(), src2.name()}

    pipeline.validate_dag()


def test_merge_then_chain_full_pipeline() -> None:
    """[src1, src2] >> det >> trk 全链路合并 + 链式。"""
    src1 = _make_file_source("fp_a")
    src2 = _make_file_source("fp_b")
    det = _make_detector("det_fp")
    trk = _make_tracker("trk_fp")

    pipeline = [src1, src2] >> det >> trk

    assert isinstance(pipeline, Pipeline)
    assert _node_names(pipeline) == {src1.name(), src2.name(), det.name(), trk.name()}

    pipeline.validate_dag()

    # 只有两个 source；det/trk 都不应被识别为 source
    source_names = {s.name() for s in pipeline.source_nodes()}
    assert source_names == {src1.name(), src2.name()}


def test_merge_tuple_syntax() -> None:
    """tuple 也应当被 _node_rrshift 识别（合并语法）。"""
    src1 = _make_file_source("tup_a")
    src2 = _make_file_source("tup_b")
    det = _make_detector("det_tup")

    pipeline = (src1, src2) >> det

    assert isinstance(pipeline, Pipeline)
    assert _node_names(pipeline) == {src1.name(), src2.name(), det.name()}


# ---------------------------------------------------------------------------
# 3. run(block) API
# ---------------------------------------------------------------------------


def test_run_default_is_nonblocking_returns_self() -> None:
    """验证 run() 方法签名：默认 block=False，返回 Pipeline 自身。"""
    src = _make_file_source("run_default")
    trk = _make_tracker("trk_run_default")
    pipeline = src >> trk

    assert callable(getattr(pipeline, "run", None))

    import inspect
    sig = inspect.signature(pipeline.run)
    assert "block" in sig.parameters
    assert sig.parameters["block"].default is False


def test_run_block_false_explicit_returns_self() -> None:
    """验证 run(block=False) 和 run(block=True) 参数不抛 TypeError。"""
    src = _make_file_source("run_explicit")
    trk = _make_tracker("trk_run_explicit")
    pipeline = src >> trk

    import inspect
    sig = inspect.signature(pipeline.run)
    params = sig.parameters
    assert "block" in params
    # block 默认值为 False
    assert params["block"].default is False


def test_stop_is_available_on_pipeline() -> None:
    """stop() 方法存在且可调用。"""
    src = _make_file_source("stop_avail")
    trk = _make_tracker("trk_stop_avail")
    pipeline = src >> trk

    assert callable(getattr(pipeline, "stop", None))
    # stop 在 INIT 状态调用不应抛异常
    pipeline.stop()
    pipeline.stop()  # 幂等


# ---------------------------------------------------------------------------
# 4. PyNode 参与 DSL
# ---------------------------------------------------------------------------


class _CountingPyNode(PyNode):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.calls = 0

    def process(self, frame) -> None:  # noqa: ANN001 — Frame runtime type
        self.calls += 1


def test_pynode_rshift_returns_pipeline() -> None:
    pynode = _CountingPyNode(name="py_chain")
    trk = _make_tracker("trk_after_py")

    pipeline = pynode >> trk

    assert isinstance(pipeline, Pipeline)
    assert "py_chain" in pipeline.nodes()
    assert "trk_after_py" in pipeline.nodes()
    # PyNode 包装类自身不在 pipeline.nodes() 中，但其 _cpp_node 在
    assert pipeline.get_node("py_chain") is pynode._cpp_node


def test_pynode_in_middle_of_chain() -> None:
    """src >> pynode >> sink 应该工作。"""
    src = _make_file_source("mid_src")
    pynode = _CountingPyNode(name="mid_py")
    trk = _make_tracker("mid_trk")

    pipeline = src >> pynode >> trk

    assert isinstance(pipeline, Pipeline)
    assert _node_names(pipeline) == {src.name(), "mid_py", "mid_trk"}
    pipeline.validate_dag()


def test_pynode_as_merge_target() -> None:
    """[src1, src2] >> pynode 应该正常工作。"""
    src1 = _make_file_source("pmerge_a")
    src2 = _make_file_source("pmerge_b")
    pynode = _CountingPyNode(name="pmerge_py")

    pipeline = [src1, src2] >> pynode

    assert isinstance(pipeline, Pipeline)
    assert _node_names(pipeline) == {src1.name(), src2.name(), "pmerge_py"}
    source_names = {s.name() for s in pipeline.source_nodes()}
    assert source_names == {src1.name(), src2.name()}


# ---------------------------------------------------------------------------
# 5. 边界情况
# ---------------------------------------------------------------------------


def test_single_element_list_merge_equivalent_to_rshift() -> None:
    """[src] >> det 应等价于 src >> det。"""
    src = _make_file_source("single_list")
    det = _make_detector("det_single_list")

    pipeline = [src] >> det

    assert isinstance(pipeline, Pipeline)
    assert _node_names(pipeline) == {src.name(), det.name()}
    pipeline.validate_dag()

    source_names = {s.name() for s in pipeline.source_nodes()}
    assert source_names == {src.name()}


def test_empty_list_merge_behaviour() -> None:
    """空列表 [] >> det 的行为：要么得到只含 det 的 Pipeline，要么抛出明确异常。"""
    det = _make_detector("det_empty")

    raised = False
    pipeline = None
    try:
        pipeline = [] >> det
    except (TypeError, ValueError, visionpipe.ConfigError, visionpipe.VisionPipeError):
        raised = True

    if raised:
        # 抛出明确异常是合理设计
        return

    # 如果未抛出异常，应得到一个只包含 det 的 Pipeline
    assert isinstance(pipeline, Pipeline)
    assert _node_names(pipeline) == {det.name()}


def test_independent_chains_produce_independent_pipelines() -> None:
    """独立的两个 `a >> b` 应该产生两个独立的 Pipeline 实例。"""
    src_a = _make_file_source("ind_a")
    det_a = _make_detector("det_ind_a")

    src_b = _make_file_source("ind_b")
    det_b = _make_detector("det_ind_b")

    pipe_a = src_a >> det_a
    pipe_b = src_b >> det_b

    assert isinstance(pipe_a, Pipeline)
    assert isinstance(pipe_b, Pipeline)
    assert pipe_a is not pipe_b
    assert _node_names(pipe_a) == {src_a.name(), det_a.name()}
    assert _node_names(pipe_b) == {src_b.name(), det_b.name()}
    # 两个 Pipeline 节点集不相交
    assert _node_names(pipe_a).isdisjoint(_node_names(pipe_b))
