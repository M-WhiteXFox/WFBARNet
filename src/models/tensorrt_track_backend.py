from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _torch_dtype_from_trt(dtype: Any) -> torch.dtype:
    import tensorrt as trt  # type: ignore

    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype}")


@dataclass(slots=True)
class TensorRTTrackBackend:
    engine_path: str
    device: str
    engine: Any = field(init=False, repr=False)
    context: Any = field(init=False, repr=False)
    runtime: Any = field(init=False, repr=False)
    input_name: str = field(init=False)
    output_name: str = field(init=False)
    _trt: Any = field(init=False, repr=False)
    _logger: Any = field(init=False, repr=False)
    _use_io_tensor_api: bool = field(init=False)
    _input_index: int = field(init=False, default=-1)
    _output_index: int = field(init=False, default=-1)
    _torch_device: torch.device = field(init=False, repr=False)
    _stream: torch.cuda.Stream = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT engine inference requires a CUDA-capable PyTorch environment.")
        if not self.device.startswith("cuda"):
            raise RuntimeError("TensorRT engine inference requires device='cuda' or 'cuda:N'.")
        if not Path(self.engine_path).is_file():
            raise FileNotFoundError(f"TensorRT engine file not found: {self.engine_path}")

        self._torch_device = torch.device(self.device)
        self._stream = torch.cuda.Stream(device=self._torch_device)

        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:
            raise RuntimeError("TensorRT Python package is required to load .engine models.") from exc

        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        try:
            trt.init_libnvinfer_plugins(self._logger, "")
            self.runtime = trt.Runtime(self._logger)
        except Exception as exc:
            raise RuntimeError(
                "TensorRT Runtime 初始化失败。请确认当前 Python 环境中的 tensorrt 包、CUDA、显卡驱动、"
                "以及系统 PATH 中的 TensorRT DLL/动态库版本彼此匹配。原始错误: "
                f"{exc}"
            ) from exc

        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                "TensorRT engine 反序列化失败。该 engine 可能与当前 TensorRT/CUDA/GPU 架构不兼容，"
                f"请在当前机器环境重新导出 engine: {self.engine_path}"
            )

        self.context = self.engine.create_execution_context()
        self._use_io_tensor_api = hasattr(self.engine, "num_io_tensors")
        self.input_name, self.output_name = self._discover_io_tensors()

    def _discover_io_tensors(self) -> tuple[str, str]:
        trt = self._trt
        if self._use_io_tensor_api:
            inputs: list[str] = []
            outputs: list[str] = []
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    inputs.append(name)
                elif mode == trt.TensorIOMode.OUTPUT:
                    outputs.append(name)
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(
                    f"Track TensorRT engine must expose exactly 1 input and 1 output, got {len(inputs)} inputs and {len(outputs)} outputs."
                )
            return inputs[0], outputs[0]

        inputs = []
        outputs = []
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            if self.engine.binding_is_input(index):
                inputs.append((index, name))
            else:
                outputs.append((index, name))
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Track TensorRT engine must expose exactly 1 input and 1 output, got {len(inputs)} inputs and {len(outputs)} outputs."
            )
        self._input_index, input_name = inputs[0]
        self._output_index, output_name = outputs[0]
        return input_name, output_name

    def _output_shape(self) -> tuple[int, ...]:
        if self._use_io_tensor_api:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        else:
            shape = tuple(int(dim) for dim in self.context.get_binding_shape(self._output_index))
        if any(dim <= 0 for dim in shape):
            raise RuntimeError(f"TensorRT output shape is not fully specified: {shape}")
        return shape

    def _output_dtype(self) -> torch.dtype:
        if self._use_io_tensor_api:
            dtype = self.engine.get_tensor_dtype(self.output_name)
        else:
            dtype = self.engine.get_binding_dtype(self._output_index)
        return _torch_dtype_from_trt(dtype)

    def _input_dtype(self) -> torch.dtype:
        if self._use_io_tensor_api:
            dtype = self.engine.get_tensor_dtype(self.input_name)
        else:
            dtype = self.engine.get_binding_dtype(self._input_index)
        return _torch_dtype_from_trt(dtype)

    def _prepare_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_cuda:
            tensor = tensor.to(self.device, non_blocking=True)
        dtype = self._input_dtype()
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous()

    def stream_context(self) -> Any:
        return torch.cuda.stream(self._stream)

    @torch.inference_mode()
    def infer(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = self._prepare_input(tensor)
        producer_stream = torch.cuda.current_stream(device=tensor.device)
        stream = self._stream
        if producer_stream.cuda_stream != stream.cuda_stream:
            stream.wait_stream(producer_stream)

        with torch.cuda.stream(stream):
            if self._use_io_tensor_api:
                if hasattr(self.context, "set_input_shape"):
                    self.context.set_input_shape(self.input_name, tuple(tensor.shape))
                self.context.set_tensor_address(self.input_name, int(tensor.data_ptr()))
                output = torch.empty(self._output_shape(), device=tensor.device, dtype=self._output_dtype())
                self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
                ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
            else:
                if self.engine.is_shape_binding(self._input_index) or any(dim < 0 for dim in self.engine.get_binding_shape(self._input_index)):
                    self.context.set_binding_shape(self._input_index, tuple(tensor.shape))
                output = torch.empty(self._output_shape(), device=tensor.device, dtype=self._output_dtype())
                bindings = [0] * self.engine.num_bindings
                bindings[self._input_index] = int(tensor.data_ptr())
                bindings[self._output_index] = int(output.data_ptr())
                ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)

        if not ok:
            raise RuntimeError("TensorRT engine execution failed.")
        stream.synchronize()
        return output.float().detach().cpu().numpy()
