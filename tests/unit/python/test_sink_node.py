"""T4.4 单元测试：SinkNode Python 绑定 — enabled 机制与继承关系"""

import visionpipe as vp


class TestSinkNodeHierarchy:
    """SinkNode 继承关系验证"""

    def test_sink_node_exists(self):
        assert hasattr(vp, "SinkNode")

    def test_json_result_sink_is_sink_node(self):
        assert issubclass(vp.JsonResultSink, vp.SinkNode)

    def test_mjpeg_sink_is_sink_node(self):
        assert issubclass(vp.MjpegSink, vp.SinkNode)

    def test_webrtc_sink_is_sink_node(self):
        assert issubclass(vp.WebRTCSink, vp.SinkNode)

    def test_sink_node_is_node_base(self):
        assert issubclass(vp.SinkNode, vp.NodeBase)

    def test_json_result_sink_is_sink_flag(self):
        sink = vp.JsonResultSink()
        assert sink.is_sink() is True
        assert sink.is_source() is False

    def test_mjpeg_sink_is_sink_flag(self):
        sink = vp.MjpegSink()
        assert sink.is_sink() is True
        assert sink.is_source() is False


class TestSinkNodeDefaultEnabled:
    """默认 enabled 状态验证"""

    def test_json_result_sink_default_enabled(self):
        sink = vp.JsonResultSink()
        assert sink.enabled() is True

    def test_mjpeg_sink_default_disabled(self):
        sink = vp.MjpegSink()
        assert sink.enabled() is False


class TestSinkNodeSetEnabled:
    """enabled 切换验证"""

    def test_set_enabled_true(self):
        sink = vp.MjpegSink()
        assert sink.enabled() is False
        sink.set_enabled(True)
        assert sink.enabled() is True

    def test_set_enabled_false(self):
        sink = vp.JsonResultSink()
        assert sink.enabled() is True
        sink.set_enabled(False)
        assert sink.enabled() is False

    def test_toggle_multiple_times(self):
        sink = vp.JsonResultSink()
        for _ in range(10):
            sink.set_enabled(False)
            assert sink.enabled() is False
            sink.set_enabled(True)
            assert sink.enabled() is True
