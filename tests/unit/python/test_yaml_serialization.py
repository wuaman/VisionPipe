"""Tests for T3.3 YAML serialization/deserialization.

Tests cover:
1. NodeSpec normal path: create valid NodeSpec for all allowed types
2. NodeSpec boundary: empty params dict, all valid type values
3. NodeSpec error path: invalid type string triggers ValidationError
4. PipelineSpec normal path: nodes only, nodes + edges
5. EdgeSpec validation: edges referencing non-existent nodes trigger ValidationError
6. YAML round-trip: model_dump -> yaml.dump -> yaml.safe_load -> model_validate
7. load_yaml normal path: write valid YAML, load returns correct PipelineSpec
8. load_yaml error path: invalid node type in YAML triggers ValidationError
9. Empty node list: PipelineSpec allows nodes=[] with edges=[]
10. Default values: default_queue_capacity=16, default_overflow_policy="DROP_OLDEST"

All tests run without GPU and without importing the main visionpipe package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from pydantic import ValidationError

from visionpipe.serialization import (
    EdgeSpec,
    NodeSpec,
    PipelineSpec,
    load_yaml,
)


# ---------------------------------------------------------------------------
# 1. NodeSpec normal path
# ---------------------------------------------------------------------------


class TestNodeSpecNormalPath:
    """Create valid NodeSpec instances for all supported node types."""

    @pytest.mark.parametrize(
        "node_type",
        [
            "file_source",
            "rtsp_source",
            "detector",
            "classifier",
            "segment",
            "bytetrack",
            "py_node",
        ],
    )
    def test_valid_node_types(self, node_type: str) -> None:
        spec = NodeSpec(name=f"test_{node_type}", type=node_type, params={"key": "val"})
        assert spec.name == f"test_{node_type}"
        assert spec.type == node_type
        assert spec.params == {"key": "val"}

    def test_node_with_complex_params(self) -> None:
        params = {
            "uri": "/path/to/video.mp4",
            "gpu_device": 0,
            "score_threshold": 0.5,
            "nested": {"a": 1, "b": [2, 3]},
        }
        spec = NodeSpec(name="det1", type="detector", params=params)
        assert spec.params == params


# ---------------------------------------------------------------------------
# 2. NodeSpec boundary values
# ---------------------------------------------------------------------------


class TestNodeSpecBoundary:
    """Boundary cases: empty params, minimum valid configurations."""

    def test_empty_params_default(self) -> None:
        spec = NodeSpec(name="src", type="file_source")
        assert spec.params == {}

    def test_explicit_empty_params(self) -> None:
        spec = NodeSpec(name="src", type="file_source", params={})
        assert spec.params == {}

    def test_all_valid_types_boundary(self) -> None:
        valid_types = [
            "file_source",
            "rtsp_source",
            "detector",
            "classifier",
            "segment",
            "bytetrack",
            "py_node",
        ]
        for t in valid_types:
            spec = NodeSpec(name="n", type=t)
            assert spec.type == t


# ---------------------------------------------------------------------------
# 3. NodeSpec error path
# ---------------------------------------------------------------------------


class TestNodeSpecErrorPath:
    """Invalid type strings must trigger ValidationError."""

    @pytest.mark.parametrize(
        "invalid_type",
        [
            "invalid_node",
            "FileSource",
            "DETECTOR",
            "file-source",
            "",
            "source",
            "infer",
            "unknown",
        ],
    )
    def test_invalid_type_raises_validation_error(self, invalid_type: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            NodeSpec(name="bad_node", type=invalid_type)
        assert "node type" in str(exc_info.value).lower() or "input" in str(exc_info.value).lower()

    def test_none_type_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            NodeSpec(name="bad_node", type=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. PipelineSpec normal path
# ---------------------------------------------------------------------------


class TestPipelineSpecNormalPath:
    """Valid PipelineSpec with nodes only and with nodes + edges."""

    def test_nodes_only_no_edges(self) -> None:
        nodes = [
            NodeSpec(name="src", type="file_source", params={"uri": "test.mp4"}),
            NodeSpec(name="det", type="detector"),
        ]
        spec = PipelineSpec(name="my_pipeline", nodes=nodes)
        assert spec.name == "my_pipeline"
        assert len(spec.nodes) == 2
        assert spec.edges == []

    def test_nodes_with_edges(self) -> None:
        nodes = [
            NodeSpec(name="src", type="file_source"),
            NodeSpec(name="det", type="detector"),
            NodeSpec(name="track", type="bytetrack"),
        ]
        edges = [
            EdgeSpec(from_node="src", to_node="det"),
            EdgeSpec(from_node="det", to_node="track"),
        ]
        spec = PipelineSpec(name="pipe", nodes=nodes, edges=edges)
        assert len(spec.edges) == 2
        assert spec.edges[0].from_node == "src"
        assert spec.edges[0].to_node == "det"
        assert spec.edges[1].from_node == "det"
        assert spec.edges[1].to_node == "track"

    def test_custom_pipeline_config(self) -> None:
        nodes = [NodeSpec(name="src", type="file_source")]
        spec = PipelineSpec(
            name="custom",
            id="pipe-001",
            default_queue_capacity=32,
            default_overflow_policy="BLOCK",
            nodes=nodes,
        )
        assert spec.id == "pipe-001"
        assert spec.default_queue_capacity == 32
        assert spec.default_overflow_policy == "BLOCK"


# ---------------------------------------------------------------------------
# 5. EdgeSpec edge reference validation
# ---------------------------------------------------------------------------


class TestEdgeSpecValidation:
    """Edges referencing non-existent nodes must raise ValidationError."""

    def test_edge_unknown_source_node(self) -> None:
        nodes = [
            NodeSpec(name="src", type="file_source"),
            NodeSpec(name="det", type="detector"),
        ]
        edges = [EdgeSpec(from_node="nonexistent", to_node="det")]
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(name="pipe", nodes=nodes, edges=edges)
        assert "nonexistent" in str(exc_info.value)

    def test_edge_unknown_target_node(self) -> None:
        nodes = [
            NodeSpec(name="src", type="file_source"),
            NodeSpec(name="det", type="detector"),
        ]
        edges = [EdgeSpec(from_node="src", to_node="missing_node")]
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(name="pipe", nodes=nodes, edges=edges)
        assert "missing_node" in str(exc_info.value)

    def test_edge_both_nodes_unknown(self) -> None:
        nodes = [NodeSpec(name="src", type="file_source")]
        edges = [EdgeSpec(from_node="ghost_a", to_node="ghost_b")]
        with pytest.raises(ValidationError) as exc_info:
            PipelineSpec(name="pipe", nodes=nodes, edges=edges)
        # At least the first unknown node should be mentioned
        assert "ghost_a" in str(exc_info.value)

    def test_valid_edges_pass(self) -> None:
        nodes = [
            NodeSpec(name="a", type="file_source"),
            NodeSpec(name="b", type="detector"),
        ]
        edges = [EdgeSpec(from_node="a", to_node="b")]
        spec = PipelineSpec(name="pipe", nodes=nodes, edges=edges)
        assert len(spec.edges) == 1


# ---------------------------------------------------------------------------
# 6. YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    """PipelineSpec -> model_dump -> yaml.dump -> yaml.safe_load -> model_validate."""

    def test_round_trip_simple_pipeline(self) -> None:
        original = PipelineSpec(
            name="roundtrip",
            id="rt-001",
            default_queue_capacity=16,
            default_overflow_policy="DROP_OLDEST",
            nodes=[
                NodeSpec(name="src", type="file_source", params={"uri": "/tmp/v.mp4"}),
                NodeSpec(name="det", type="detector", params={"score_threshold": 0.5}),
            ],
            edges=[EdgeSpec(from_node="src", to_node="det")],
        )

        # Serialize
        data = original.model_dump()
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False)

        # Deserialize
        loaded_data = yaml.safe_load(yaml_str)
        restored = PipelineSpec.model_validate(loaded_data)

        assert restored == original

    def test_round_trip_complex_params(self) -> None:
        original = PipelineSpec(
            name="complex",
            nodes=[
                NodeSpec(
                    name="det",
                    type="detector",
                    params={
                        "input_width": 640,
                        "input_height": 640,
                        "score_threshold": 0.25,
                        "nms_threshold": 0.45,
                        "max_detections": 100,
                        "workers": 2,
                    },
                ),
                NodeSpec(
                    name="track",
                    type="bytetrack",
                    params={
                        "track_thresh": 0.5,
                        "track_buffer": 30,
                        "match_thresh": 0.8,
                        "frame_rate": 30,
                    },
                ),
            ],
            edges=[EdgeSpec(from_node="det", to_node="track")],
        )

        data = original.model_dump()
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False)
        loaded_data = yaml.safe_load(yaml_str)
        restored = PipelineSpec.model_validate(loaded_data)

        assert restored == original

    def test_round_trip_no_edges(self) -> None:
        original = PipelineSpec(
            name="solo",
            nodes=[NodeSpec(name="src", type="rtsp_source", params={"uri": "rtsp://x"})],
        )
        data = original.model_dump()
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False)
        loaded_data = yaml.safe_load(yaml_str)
        restored = PipelineSpec.model_validate(loaded_data)

        assert restored == original


# ---------------------------------------------------------------------------
# 7. load_yaml normal path
# ---------------------------------------------------------------------------


class TestLoadYamlNormalPath:
    """Write valid YAML file and load with load_yaml."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "loaded_pipe",
            "id": "lp-001",
            "default_queue_capacity": 16,
            "default_overflow_policy": "DROP_OLDEST",
            "nodes": [
                {"name": "src", "type": "file_source", "params": {"uri": "video.mp4"}},
                {"name": "det", "type": "detector", "params": {"workers": 2}},
            ],
            "edges": [{"from_node": "src", "to_node": "det"}],
        }
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml.dump(yaml_content, allow_unicode=True, sort_keys=False))

        spec = load_yaml(yaml_file)

        assert spec.name == "loaded_pipe"
        assert spec.id == "lp-001"
        assert len(spec.nodes) == 2
        assert spec.nodes[0].name == "src"
        assert spec.nodes[0].type == "file_source"
        assert spec.nodes[0].params == {"uri": "video.mp4"}
        assert spec.nodes[1].name == "det"
        assert spec.nodes[1].type == "detector"
        assert spec.nodes[1].params == {"workers": 2}
        assert len(spec.edges) == 1
        assert spec.edges[0].from_node == "src"
        assert spec.edges[0].to_node == "det"

    def test_load_yaml_with_string_path(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "strpath",
            "nodes": [{"name": "n1", "type": "py_node", "params": {}}],
            "edges": [],
        }
        yaml_file = tmp_path / "pipe2.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        spec = load_yaml(str(yaml_file))

        assert spec.name == "strpath"
        assert len(spec.nodes) == 1
        assert spec.nodes[0].type == "py_node"

    def test_load_yaml_minimal(self, tmp_path: Path) -> None:
        """Minimal valid YAML with only required fields."""
        yaml_content = {"name": "min", "nodes": [{"name": "a", "type": "classifier"}]}
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        spec = load_yaml(yaml_file)

        assert spec.name == "min"
        assert spec.default_queue_capacity == 16
        assert spec.default_overflow_policy == "DROP_OLDEST"
        assert spec.edges == []
        assert spec.nodes[0].params == {}


# ---------------------------------------------------------------------------
# 8. load_yaml error path
# ---------------------------------------------------------------------------


class TestLoadYamlErrorPath:
    """Invalid YAML content must trigger ValidationError."""

    def test_invalid_node_type_in_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "bad_pipe",
            "nodes": [
                {"name": "src", "type": "invalid_type_xyz", "params": {}},
            ],
            "edges": [],
        }
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValidationError) as exc_info:
            load_yaml(yaml_file)
        assert "invalid_type_xyz" in str(exc_info.value)

    def test_missing_node_name_in_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "bad_pipe",
            "nodes": [
                {"type": "detector", "params": {}},
            ],
        }
        yaml_file = tmp_path / "no_name.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValidationError):
            load_yaml(yaml_file)

    def test_edge_referencing_nonexistent_node_in_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "bad_edges",
            "nodes": [{"name": "src", "type": "file_source"}],
            "edges": [{"from_node": "src", "to_node": "phantom"}],
        }
        yaml_file = tmp_path / "bad_edges.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValidationError) as exc_info:
            load_yaml(yaml_file)
        assert "phantom" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. Empty node list
# ---------------------------------------------------------------------------


class TestEmptyNodeList:
    """PipelineSpec should allow empty nodes and edges lists."""

    def test_empty_nodes_and_edges(self) -> None:
        spec = PipelineSpec(name="empty", nodes=[], edges=[])
        assert spec.nodes == []
        assert spec.edges == []

    def test_empty_nodes_default_edges(self) -> None:
        spec = PipelineSpec(name="empty2", nodes=[])
        assert spec.nodes == []
        assert spec.edges == []


# ---------------------------------------------------------------------------
# 10. Default value validation
# ---------------------------------------------------------------------------


class TestDefaultValues:
    """Verify PipelineSpec defaults are correctly applied."""

    def test_default_queue_capacity(self) -> None:
        spec = PipelineSpec(name="defaults", nodes=[])
        assert spec.default_queue_capacity == 16

    def test_default_overflow_policy(self) -> None:
        spec = PipelineSpec(name="defaults", nodes=[])
        assert spec.default_overflow_policy == "DROP_OLDEST"

    def test_default_id_empty_string(self) -> None:
        spec = PipelineSpec(name="defaults", nodes=[])
        assert spec.id == ""

    def test_default_name(self) -> None:
        spec = PipelineSpec(nodes=[])
        assert spec.name == "pipeline"

    def test_overriding_defaults(self) -> None:
        spec = PipelineSpec(
            name="custom",
            id="id-123",
            default_queue_capacity=64,
            default_overflow_policy="BLOCK",
            nodes=[],
        )
        assert spec.default_queue_capacity == 64
        assert spec.default_overflow_policy == "BLOCK"
        assert spec.id == "id-123"


# ---------------------------------------------------------------------------
# T3.3 新增：CustomNode YAML 支持测试
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. NodeSpec custom_node 正常路径
# ---------------------------------------------------------------------------


class TestNodeSpecCustomNodeNormalPath:
    """custom_node 类型 NodeSpec 在 module/class_name/process_mode 合法时可正常创建。"""

    def test_custom_node_subprocess_mode(self) -> None:
        spec = NodeSpec(
            name="my_custom",
            type="custom_node",
            params={"threshold": 0.5},
            module="my_pkg.my_mod",
            class_name="MyCustomNode",
            process_mode="subprocess",
        )
        assert spec.name == "my_custom"
        assert spec.type == "custom_node"
        assert spec.params == {"threshold": 0.5}
        assert spec.module == "my_pkg.my_mod"
        assert spec.class_name == "MyCustomNode"
        assert spec.process_mode == "subprocess"

    def test_custom_node_inline_mode(self) -> None:
        spec = NodeSpec(
            name="my_inline",
            type="custom_node",
            params={"a": 1, "b": "text"},
            module="some.module",
            class_name="InlineNode",
            process_mode="inline",
        )
        assert spec.process_mode == "inline"
        assert spec.module == "some.module"
        assert spec.class_name == "InlineNode"

    def test_custom_node_with_nested_params(self) -> None:
        params = {"cfg": {"a": [1, 2, 3], "b": True}, "name": "x"}
        spec = NodeSpec(
            name="nested",
            type="custom_node",
            params=params,
            module="pkg.mod",
            class_name="Cls",
            process_mode="subprocess",
        )
        assert spec.params == params


# ---------------------------------------------------------------------------
# 12. NodeSpec custom_node 边界值
# ---------------------------------------------------------------------------


class TestNodeSpecCustomNodeBoundary:
    """边界：process_mode=None（使用默认）、params 为空。"""

    def test_custom_node_process_mode_none(self) -> None:
        spec = NodeSpec(
            name="default_mode",
            type="custom_node",
            module="pkg.mod",
            class_name="Cls",
        )
        assert spec.process_mode is None

    def test_custom_node_empty_params(self) -> None:
        spec = NodeSpec(
            name="empty_params",
            type="custom_node",
            params={},
            module="pkg.mod",
            class_name="Cls",
        )
        assert spec.params == {}

    def test_custom_node_default_params(self) -> None:
        """未传 params 时使用默认空 dict。"""
        spec = NodeSpec(
            name="no_params",
            type="custom_node",
            module="pkg.mod",
            class_name="Cls",
        )
        assert spec.params == {}


# ---------------------------------------------------------------------------
# 13. NodeSpec custom_node 错误路径
# ---------------------------------------------------------------------------


class TestNodeSpecCustomNodeErrorPath:
    """custom_node 缺少必填字段或字段非法时抛 ValidationError。"""

    def test_missing_module_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            NodeSpec(
                name="bad",
                type="custom_node",
                class_name="Cls",
            )
        assert "module" in str(exc_info.value)

    def test_missing_class_name_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            NodeSpec(
                name="bad",
                type="custom_node",
                module="pkg.mod",
            )
        assert "class_name" in str(exc_info.value)

    def test_missing_both_module_and_class_name_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            NodeSpec(name="bad", type="custom_node")
        msg = str(exc_info.value)
        assert "module" in msg and "class_name" in msg

    def test_empty_string_module_raises(self) -> None:
        with pytest.raises(ValidationError):
            NodeSpec(
                name="bad",
                type="custom_node",
                module="",
                class_name="Cls",
            )

    def test_empty_string_class_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            NodeSpec(
                name="bad",
                type="custom_node",
                module="pkg.mod",
                class_name="",
            )

    @pytest.mark.parametrize(
        "bad_mode",
        ["thread", "async", "SUBPROCESS", "INLINE", "fork", "process"],
    )
    def test_invalid_process_mode_raises(self, bad_mode: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            NodeSpec(
                name="bad",
                type="custom_node",
                module="pkg.mod",
                class_name="Cls",
                process_mode=bad_mode,
            )
        assert "process_mode" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 14. annotator 类型是合法 NodeType
# ---------------------------------------------------------------------------


class TestAnnotatorNodeType:
    """annotator 是合法 NodeType，普通节点不要求 module/class_name。"""

    def test_annotator_valid_type(self) -> None:
        spec = NodeSpec(name="anno", type="annotator", params={"draw_detections": True})
        assert spec.type == "annotator"
        assert spec.params == {"draw_detections": True}

    def test_annotator_no_params(self) -> None:
        spec = NodeSpec(name="anno", type="annotator")
        assert spec.type == "annotator"
        assert spec.params == {}
        assert spec.module is None
        assert spec.class_name is None
        assert spec.process_mode is None


# ---------------------------------------------------------------------------
# 15. YAML round-trip with custom_node
# ---------------------------------------------------------------------------


class TestYamlRoundTripCustomNode:
    """含 custom_node 的 PipelineSpec model_dump → yaml → model_validate 往返一致。"""

    def test_round_trip_custom_node(self) -> None:
        original = PipelineSpec(
            name="cn_pipe",
            id="cn-001",
            nodes=[
                NodeSpec(name="src", type="file_source", params={"uri": "v.mp4"}),
                NodeSpec(
                    name="cn",
                    type="custom_node",
                    params={"threshold": 0.7, "label": "x"},
                    module="my_pkg.cn",
                    class_name="MyCN",
                    process_mode="subprocess",
                ),
            ],
            edges=[EdgeSpec(from_node="src", to_node="cn")],
        )

        data = original.model_dump()
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False)
        loaded_data = yaml.safe_load(yaml_str)
        restored = PipelineSpec.model_validate(loaded_data)

        assert restored == original
        assert restored.nodes[1].module == "my_pkg.cn"
        assert restored.nodes[1].class_name == "MyCN"
        assert restored.nodes[1].process_mode == "subprocess"

    def test_round_trip_custom_node_inline(self) -> None:
        original = PipelineSpec(
            name="cn_inline",
            nodes=[
                NodeSpec(
                    name="cn",
                    type="custom_node",
                    params={},
                    module="pkg.mod",
                    class_name="Cls",
                    process_mode="inline",
                ),
            ],
        )
        data = original.model_dump()
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False)
        restored = PipelineSpec.model_validate(yaml.safe_load(yaml_str))
        assert restored == original
        assert restored.nodes[0].process_mode == "inline"

    def test_round_trip_custom_node_default_process_mode(self) -> None:
        """process_mode 未设置时往返保持 None。"""
        original = PipelineSpec(
            name="cn_def",
            nodes=[
                NodeSpec(
                    name="cn",
                    type="custom_node",
                    module="pkg.mod",
                    class_name="Cls",
                ),
            ],
        )
        data = original.model_dump()
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False)
        restored = PipelineSpec.model_validate(yaml.safe_load(yaml_str))
        assert restored == original
        assert restored.nodes[0].process_mode is None


# ---------------------------------------------------------------------------
# 16. load_yaml with custom_node
# ---------------------------------------------------------------------------


class TestLoadYamlCustomNode:
    """写含 custom_node 的 YAML 文件，load_yaml 返回正确的 PipelineSpec。"""

    def test_load_yaml_with_custom_node(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "cn_load",
            "id": "cn-load-001",
            "nodes": [
                {"name": "src", "type": "file_source", "params": {"uri": "v.mp4"}},
                {
                    "name": "cn",
                    "type": "custom_node",
                    "params": {"k": 1},
                    "module": "my_pkg.my_mod",
                    "class_name": "MyClass",
                    "process_mode": "subprocess",
                },
            ],
            "edges": [{"from_node": "src", "to_node": "cn"}],
        }
        yaml_file = tmp_path / "cn.yaml"
        yaml_file.write_text(yaml.dump(yaml_content, allow_unicode=True, sort_keys=False))

        spec = load_yaml(yaml_file)

        assert spec.name == "cn_load"
        assert len(spec.nodes) == 2
        cn_spec = spec.nodes[1]
        assert cn_spec.name == "cn"
        assert cn_spec.type == "custom_node"
        assert cn_spec.params == {"k": 1}
        assert cn_spec.module == "my_pkg.my_mod"
        assert cn_spec.class_name == "MyClass"
        assert cn_spec.process_mode == "subprocess"

    def test_load_yaml_custom_node_missing_module_fails(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "bad_cn",
            "nodes": [
                {
                    "name": "cn",
                    "type": "custom_node",
                    "params": {},
                    "class_name": "MyClass",
                },
            ],
        }
        yaml_file = tmp_path / "bad_cn.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValidationError) as exc_info:
            load_yaml(yaml_file)
        assert "module" in str(exc_info.value)

    def test_load_yaml_custom_node_invalid_process_mode_fails(self, tmp_path: Path) -> None:
        yaml_content = {
            "name": "bad_cn",
            "nodes": [
                {
                    "name": "cn",
                    "type": "custom_node",
                    "module": "pkg.mod",
                    "class_name": "Cls",
                    "process_mode": "thread",
                },
            ],
        }
        yaml_file = tmp_path / "bad_mode.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        with pytest.raises(ValidationError) as exc_info:
            load_yaml(yaml_file)
        assert "process_mode" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 17. _import_custom_node 正常/错误路径
# ---------------------------------------------------------------------------


# 模块级 stub：用于 _import_custom_node 测试
# 注意：此模块路径必须可通过 importlib 解析到本测试模块
class _StubCustomNode:
    """测试用 CustomNode 替身——绕过 C++ 节点构建。

    _import_custom_node 调用 cls(name=..., process_mode=..., **params)，
    本类直接接收这些参数并记录，不构建任何 C++ 资源。
    """

    def __init__(
        self,
        name: str = "stub",
        process_mode: str = "subprocess",
        **kwargs,
    ) -> None:
        self.name = name
        self.process_mode = process_mode
        self.kwargs = kwargs


class _StubWithExtra(_StubCustomNode):
    """另一个 stub，验证类名解析正确。"""

    def __init__(self, name: str = "x", process_mode: str = "subprocess", **kwargs) -> None:
        super().__init__(name=name, process_mode=process_mode, **kwargs)
        self.extra = kwargs.get("extra_value")


class TestImportCustomNodeNormalPath:
    """_import_custom_node 可从 module/class_name 自动导入并实例化。"""

    def test_import_and_instantiate(self) -> None:
        from visionpipe.serialization import _import_custom_node

        spec = NodeSpec(
            name="stub_node",
            type="custom_node",
            params={"threshold": 0.5, "label": "cat"},
            module=__name__,
            class_name="_StubCustomNode",
            process_mode="inline",
        )
        instance = _import_custom_node(spec)

        assert isinstance(instance, _StubCustomNode)
        assert instance.name == "stub_node"
        assert instance.process_mode == "inline"
        assert instance.kwargs == {"threshold": 0.5, "label": "cat"}

    def test_import_default_process_mode_subprocess(self) -> None:
        """spec.process_mode 为 None 时，传给构造函数的应为 'subprocess'。"""
        from visionpipe.serialization import _import_custom_node

        spec = NodeSpec(
            name="default_mode",
            type="custom_node",
            params={},
            module=__name__,
            class_name="_StubCustomNode",
        )
        instance = _import_custom_node(spec)

        assert instance.process_mode == "subprocess"
        assert instance.kwargs == {}

    def test_import_different_class(self) -> None:
        """class_name 解析必须命中对应类。"""
        from visionpipe.serialization import _import_custom_node

        spec = NodeSpec(
            name="extra_node",
            type="custom_node",
            params={"extra_value": 42},
            module=__name__,
            class_name="_StubWithExtra",
            process_mode="subprocess",
        )
        instance = _import_custom_node(spec)

        assert isinstance(instance, _StubWithExtra)
        assert instance.extra == 42


class TestImportCustomNodeErrorPath:
    """_import_custom_node 错误路径：module/class 不存在。"""

    def test_module_not_found_raises_import_error(self) -> None:
        from visionpipe.serialization import _import_custom_node

        spec = NodeSpec(
            name="bad",
            type="custom_node",
            module="this_module_definitely_does_not_exist_xyz_123",
            class_name="WhateverClass",
        )
        with pytest.raises(ImportError):
            _import_custom_node(spec)

    def test_class_not_found_in_module_raises_attribute_error(self) -> None:
        from visionpipe.serialization import _import_custom_node

        spec = NodeSpec(
            name="bad",
            type="custom_node",
            module=__name__,
            class_name="NonExistentClassName_xyz",
        )
        with pytest.raises(AttributeError):
            _import_custom_node(spec)


# ---------------------------------------------------------------------------
# 18. 非 custom_node 类型不要求 module/class_name
# ---------------------------------------------------------------------------


class TestNonCustomNodeTypeNoExtraFields:
    """非 custom_node 类型不要求 module/class_name，且默认为 None。"""

    def test_file_source_no_module(self) -> None:
        spec = NodeSpec(name="src", type="file_source", module=None)
        assert spec.module is None
        assert spec.class_name is None
        assert spec.type == "file_source"

    def test_detector_module_none(self) -> None:
        spec = NodeSpec(name="det", type="detector")
        assert spec.module is None
        assert spec.class_name is None
        assert spec.process_mode is None

    def test_non_custom_node_with_module_set_is_allowed(self) -> None:
        """非 custom_node 类型即便额外指定了 module/class_name 也不报错（字段是可选的）。"""
        spec = NodeSpec(
            name="src",
            type="file_source",
            module="ignored.module",
            class_name="IgnoredClass",
        )
        assert spec.type == "file_source"
        assert spec.module == "ignored.module"
        assert spec.class_name == "IgnoredClass"

    @pytest.mark.parametrize(
        "node_type",
        [
            "file_source",
            "rtsp_source",
            "detector",
            "classifier",
            "segment",
            "bytetrack",
            "annotator",
            "py_node",
            "json_result_sink",
            "mjpeg_sink",
            "webrtc_sink",
        ],
    )
    def test_all_non_custom_types_without_module(self, node_type: str) -> None:
        spec = NodeSpec(name=f"n_{node_type}", type=node_type)
        assert spec.module is None
        assert spec.class_name is None
