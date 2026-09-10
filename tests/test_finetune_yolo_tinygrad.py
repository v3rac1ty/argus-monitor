"""Unit tests for training/finetune_yolo_tinygrad.py's pure/CLI logic: `parse_args` defaults
and its --batch>64 rejection, `assert_device_guard`, `find_pre_softmax_output_name`,
`make_batch_dynamic`, `select_trainable_names`, and `replace_matching_initializers` (the ONNX
weight write-back).

No GPU, no dataset, and no real ONNX model file required -- every ONNX graph used here is a
tiny synthetic one built in-memory with `onnx.helper`/`onnx.numpy_helper`. Following
training/train_tinygrad.py's and its test suite's precedent, the actual fine-tuning loop
(train()/main(), the streamed per-batch Tensor construction, the OnnxRunner forward/backward
pass, and the onnxruntime-vs-tinygrad verification step) is exercised only by a real run, never
by this suite.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from training.finetune_yolo_tinygrad import (
    DEFAULT_DATA_DIR,
    DEFAULT_ONNX_PATH,
    DEFAULT_OUT_PATH,
    MAX_BATCH_SIZE,
    assert_device_guard,
    find_pre_softmax_output_name,
    make_batch_dynamic,
    onnx_initializer_name_from_state_dict_key,
    parse_args,
    replace_matching_initializers,
    select_trainable_names,
    write_finetuned_onnx,
)


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.onnx == DEFAULT_ONNX_PATH
        assert args.data == DEFAULT_DATA_DIR
        assert args.epochs == 40
        assert args.batch == 32
        assert args.imgsz == 320
        assert args.lr == pytest.approx(1e-4)
        assert args.seed == 1337
        assert args.patience == 8
        assert args.out == DEFAULT_OUT_PATH
        assert args.no_device_guard is False

    def test_no_device_guard_flag(self):
        args = parse_args(["--no-device-guard"])
        assert args.no_device_guard is True

    def test_batch_at_max_is_allowed(self):
        args = parse_args(["--batch", str(MAX_BATCH_SIZE)])
        assert args.batch == MAX_BATCH_SIZE

    def test_batch_above_max_is_rejected(self):
        with pytest.raises(ValueError):
            parse_args(["--batch", str(MAX_BATCH_SIZE + 1)])

    def test_batch_well_above_max_is_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            parse_args(["--batch", "128"])
        assert "64" in str(excinfo.value)

    def test_overrides(self):
        args = parse_args(
            [
                "--onnx", "models/other.onnx",
                "--data", "datasets/other",
                "--epochs", "5",
                "--batch", "16",
                "--imgsz", "224",
                "--lr", "1e-3",
                "--seed", "7",
                "--patience", "3",
                "--out", "models/other_finetuned.onnx",
            ]
        )
        assert str(args.onnx) == "models/other.onnx"
        assert str(args.data) == "datasets/other"
        assert args.epochs == 5
        assert args.batch == 16
        assert args.imgsz == 224
        assert args.lr == pytest.approx(1e-3)
        assert args.seed == 7
        assert args.patience == 3
        assert str(args.out) == "models/other_finetuned.onnx"


# --------------------------------------------------------------------------
# assert_device_guard
# --------------------------------------------------------------------------


class TestAssertDeviceGuard:
    def test_passes_on_nv(self):
        assert_device_guard("NV")  # must not raise

    def test_raises_and_names_the_device_on_metal(self):
        with pytest.raises(RuntimeError) as excinfo:
            assert_device_guard("METAL")
        message = str(excinfo.value)
        assert "METAL" in message
        assert "NV" in message

    def test_raises_and_names_the_device_on_cpu(self):
        with pytest.raises(RuntimeError) as excinfo:
            assert_device_guard("CPU")
        assert "CPU" in str(excinfo.value)

    def test_error_mentions_dev_nv_env_var(self):
        with pytest.raises(RuntimeError) as excinfo:
            assert_device_guard("METAL")
        assert "DEV=NV" in str(excinfo.value)

    def test_error_mentions_no_device_guard_escape_hatch(self):
        with pytest.raises(RuntimeError) as excinfo:
            assert_device_guard("METAL")
        assert "--no-device-guard" in str(excinfo.value)


# --------------------------------------------------------------------------
# find_pre_softmax_output_name
# --------------------------------------------------------------------------


def _make_model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=list(initializers))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


class TestFindPreSoftmaxOutputName:
    def test_softmax_final_op_returns_its_input(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
        out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 2])
        gemm_w = numpy_helper.from_array(np.zeros((4, 2), dtype=np.float32), name="w")
        gemm = helper.make_node("MatMul", ["x", "w"], ["logits"])
        softmax = helper.make_node("Softmax", ["logits"], ["output0"])
        model = _make_model([gemm, softmax], [x], [out], [gemm_w])
        assert find_pre_softmax_output_name(model) == "logits"

    def test_no_softmax_returns_declared_output(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
        gemm_w = numpy_helper.from_array(np.zeros((4, 2), dtype=np.float32), name="w")
        gemm = helper.make_node("MatMul", ["x", "w"], ["logits"])
        model = _make_model([gemm], [x], [out], [gemm_w])
        assert find_pre_softmax_output_name(model) == "logits"

    def test_multiple_outputs_raises(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out1 = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 4])
        out2 = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 4])
        identity1 = helper.make_node("Identity", ["x"], ["a"])
        identity2 = helper.make_node("Identity", ["x"], ["b"])
        model = _make_model([identity1, identity2], [x], [out1, out2])
        with pytest.raises(ValueError):
            find_pre_softmax_output_name(model)

    def test_no_producer_found_raises(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        # declared output "mystery" is never produced by any node
        out = helper.make_tensor_value_info("mystery", TensorProto.FLOAT, [1, 4])
        identity = helper.make_node("Identity", ["x"], ["a"])
        model = _make_model([identity], [x], [out])
        with pytest.raises(ValueError):
            find_pre_softmax_output_name(model)


# --------------------------------------------------------------------------
# make_batch_dynamic
# --------------------------------------------------------------------------


class TestMakeBatchDynamic:
    def _model_with_reshape(self, shape_value: np.ndarray):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [int(d) for d in shape_value])
        shape_init = numpy_helper.from_array(shape_value.astype(np.int64), name="shape")
        reshape = helper.make_node("Reshape", ["x", "shape"], ["y"])
        return _make_model([reshape], [x], [out], [shape_init])

    def test_input_batch_dim_becomes_dynamic(self):
        model = self._model_with_reshape(np.array([1, 4]))
        make_batch_dynamic(model)
        dim0 = model.graph.input[0].type.tensor_type.shape.dim[0]
        assert dim0.dim_param == "batch"
        assert dim0.dim_value == 0  # cleared, no longer a fixed literal

    def test_custom_dim_param_name(self):
        model = self._model_with_reshape(np.array([1, 4]))
        make_batch_dynamic(model, dim_param="N")
        assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "N"

    def test_reshape_target_leading_one_becomes_negative_one(self):
        model = self._model_with_reshape(np.array([1, 2, 2]))
        patched = make_batch_dynamic(model)
        assert patched == ["shape"]
        new_shape_init = next(i for i in model.graph.initializer if i.name == "shape")
        assert list(numpy_helper.to_array(new_shape_init)) == [-1, 2, 2]

    def test_reshape_target_not_starting_with_one_is_untouched(self):
        model = self._model_with_reshape(np.array([4, 1]))
        patched = make_batch_dynamic(model)
        assert patched == []
        shape_init = next(i for i in model.graph.initializer if i.name == "shape")
        assert list(numpy_helper.to_array(shape_init)) == [4, 1]

    def test_non_reshape_initializer_never_touched(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        weight = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.float32), name="w")
        add = helper.make_node("Add", ["x", "w"], ["y"])
        model = _make_model([add], [x], [out], [weight])
        patched = make_batch_dynamic(model)
        assert patched == []
        w = next(i for i in model.graph.initializer if i.name == "w")
        assert list(numpy_helper.to_array(w)) == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# select_trainable_names
# --------------------------------------------------------------------------


class TestSelectTrainableNames:
    def test_partitions_by_gradient_flag(self):
        trainable, excluded = select_trainable_names({"a": True, "b": False, "c": True})
        assert trainable == ["a", "c"]
        assert excluded == ["b"]

    def test_all_trainable(self):
        trainable, excluded = select_trainable_names({"a": True, "b": True})
        assert trainable == ["a", "b"]
        assert excluded == []

    def test_all_excluded(self):
        trainable, excluded = select_trainable_names({"a": False, "b": False})
        assert trainable == []
        assert excluded == ["a", "b"]

    def test_empty_input(self):
        assert select_trainable_names({}) == ([], [])

    def test_deterministic_sorted_order(self):
        trainable, excluded = select_trainable_names({"z": True, "a": True, "m": False, "b": False})
        assert trainable == ["a", "z"]
        assert excluded == ["b", "m"]


# --------------------------------------------------------------------------
# onnx_initializer_name_from_state_dict_key
# --------------------------------------------------------------------------


class TestOnnxInitializerNameFromStateDictKey:
    def test_strips_graph_values_prefix(self):
        assert onnx_initializer_name_from_state_dict_key("graph_values.model.0.conv.weight") == "model.0.conv.weight"

    def test_strips_prefix_from_slash_style_name(self):
        key = "graph_values./model.9/m/m.0/attn/Constant_1_output_0"
        assert onnx_initializer_name_from_state_dict_key(key) == "/model.9/m/m.0/attn/Constant_1_output_0"

    def test_missing_prefix_raises(self):
        with pytest.raises(ValueError):
            onnx_initializer_name_from_state_dict_key("model.0.conv.weight")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            onnx_initializer_name_from_state_dict_key("")


# --------------------------------------------------------------------------
# replace_matching_initializers / write_finetuned_onnx
# --------------------------------------------------------------------------


def _model_with_two_initializers():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    w1 = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name="w1")
    w2 = numpy_helper.from_array(np.ones((3,), dtype=np.float32), name="w2")
    add = helper.make_node("Add", ["x", "w1"], ["y"])
    model = _make_model([add], [x], [out], [w1, w2])
    model.metadata_props.append(onnx.StringStringEntryProto(key="names", value="{0: 'failure', 1: 'normal'}"))
    return model


class TestReplaceMatchingInitializers:
    def test_replaces_matching_initializer_value(self):
        model = _model_with_two_initializers()
        new_w1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        replaced = replace_matching_initializers(model, {"w1": new_w1})
        assert replaced == ["w1"]
        w1 = next(i for i in model.graph.initializer if i.name == "w1")
        np.testing.assert_array_equal(numpy_helper.to_array(w1), new_w1)

    def test_untouched_initializer_stays_the_same(self):
        model = _model_with_two_initializers()
        replace_matching_initializers(model, {"w1": np.ones((2, 2), dtype=np.float32)})
        w2 = next(i for i in model.graph.initializer if i.name == "w2")
        np.testing.assert_array_equal(numpy_helper.to_array(w2), np.ones((3,), dtype=np.float32))

    def test_metadata_props_preserved(self):
        model = _model_with_two_initializers()
        replace_matching_initializers(model, {"w1": np.ones((2, 2), dtype=np.float32)})
        names = {p.key: p.value for p in model.metadata_props}
        assert names["names"] == "{0: 'failure', 1: 'normal'}"

    def test_dtype_cast_to_original(self):
        model = _model_with_two_initializers()
        # Pass a float64 array in; the write-back must cast it back to the initializer's
        # original dtype (float32) rather than silently promoting the model's precision.
        new_w1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        replace_matching_initializers(model, {"w1": new_w1})
        w1 = next(i for i in model.graph.initializer if i.name == "w1")
        assert w1.data_type == TensorProto.FLOAT
        arr = numpy_helper.to_array(w1)
        assert arr.dtype == np.float32

    def test_unknown_name_raises_keyerror(self):
        model = _model_with_two_initializers()
        with pytest.raises(KeyError):
            replace_matching_initializers(model, {"does_not_exist": np.zeros((2, 2), dtype=np.float32)})

    def test_shape_mismatch_raises_valueerror(self):
        model = _model_with_two_initializers()
        with pytest.raises(ValueError):
            replace_matching_initializers(model, {"w1": np.zeros((3, 3), dtype=np.float32)})

    def test_returns_only_names_actually_passed(self):
        model = _model_with_two_initializers()
        replaced = replace_matching_initializers(model, {"w1": np.ones((2, 2), dtype=np.float32)})
        assert "w2" not in replaced


class TestWriteFinetunedOnnx:
    def test_writes_file_with_patched_values_and_preserved_metadata(self, tmp_path):
        model = _model_with_two_initializers()
        original_path = tmp_path / "original.onnx"
        onnx.save(model, str(original_path))

        out_path = tmp_path / "nested" / "finetuned.onnx"
        new_w1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        write_finetuned_onnx(original_path, out_path, {"w1": new_w1})

        assert out_path.is_file()
        written = onnx.load(str(out_path))
        w1 = next(i for i in written.graph.initializer if i.name == "w1")
        np.testing.assert_array_equal(numpy_helper.to_array(w1), new_w1)
        names = {p.key: p.value for p in written.metadata_props}
        assert names["names"] == "{0: 'failure', 1: 'normal'}"

    def test_does_not_mutate_the_original_file(self, tmp_path):
        model = _model_with_two_initializers()
        original_path = tmp_path / "original.onnx"
        onnx.save(model, str(original_path))

        out_path = tmp_path / "finetuned.onnx"
        write_finetuned_onnx(original_path, out_path, {"w1": np.ones((2, 2), dtype=np.float32) * 9.0})

        reloaded_original = onnx.load(str(original_path))
        w1 = next(i for i in reloaded_original.graph.initializer if i.name == "w1")
        np.testing.assert_array_equal(numpy_helper.to_array(w1), np.zeros((2, 2), dtype=np.float32))
