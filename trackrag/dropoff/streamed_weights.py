"""Read-only file-backed weights for a pinned, frozen Qwen3-VL checkpoint.

Torch's tensor interface does not enforce mmap.ACCESS_READ. Updating or merging
these base weights can native-crash. The runtime freezes every mapped parameter
and optimizes newly created CUDA LoRA parameters only.
"""

from copy import deepcopy
import json
import math
import mmap
import struct
import warnings

import torch


DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
          "F64": torch.float64, "U8": torch.uint8, "I8": torch.int8,
          "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
          "BOOL": torch.bool}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate checkpoint header field")
        result[key] = value
    return result


class ReadOnlySafeOpen:
    """Small CPU reader used only after pinned checkpoint digest validation.

Unlike Torch's whole-file copy-on-write mapping, this uses a read-only Python
mapping. Tensor views keep its exported buffer alive after the reader is gone.
It deliberately implements only the methods used by the installed HF loader.
"""

    def __init__(self, filename, framework="pt", device="cpu", **kwargs):
        if framework != "pt" or str(device) != "cpu":
            raise ValueError("The streamed reader only supports CPU PyTorch loading")
        with open(filename, "rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated checkpoint header")
            count = struct.unpack("<Q", raw)[0]
            if not 2 <= count <= 16 * 1024**2:
                raise ValueError("Invalid checkpoint header length")
            raw_header = stream.read(count)
            if len(raw_header) != count:
                raise ValueError("Truncated checkpoint header")
            self.header = json.loads(raw_header, object_pairs_hook=_unique_object)
            if not isinstance(self.header, dict):
                raise ValueError("Checkpoint header must be an object")
            self.offset = 8 + count
            self.mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        intervals = []
        try:
            for name, meta in self.header.items():
                if name == "__metadata__":
                    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
                        raise ValueError("Invalid checkpoint metadata")
                    continue
                if not isinstance(meta, dict) or set(meta) != {"dtype", "shape", "data_offsets"}:
                    raise ValueError("Invalid checkpoint tensor metadata")
                shape = meta["shape"]
                offsets = meta["data_offsets"]
                if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                        or not isinstance(offsets, list) or len(offsets) != 2
                        or any(type(x) is not int for x in offsets)
                        or not isinstance(meta["dtype"], str) or meta["dtype"] not in DTYPES):
                    raise ValueError("Invalid checkpoint tensor shape, offsets or dtype")
                first, last = offsets
                size = torch.empty((), dtype=DTYPES[meta["dtype"]]).element_size()
                if not (0 <= first <= last <= len(self.mapping) - self.offset
                        and last - first == math.prod(shape) * size):
                    raise ValueError("Invalid checkpoint tensor bounds")
                if last > first:
                    intervals.append((first, last))
            intervals.sort()
            if any(right > next_left for (_, right), (next_left, _) in zip(intervals, intervals[1:])):
                raise ValueError("Overlapping checkpoint tensors")
        except Exception:
            self.mapping.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Existing tensors/slices own exported views. Closing the mapping here
        # would invalidate those views; release follows their normal lifetime.
        return False

    def keys(self):
        return [key for key in self.header if key != "__metadata__"]

    def metadata(self):
        return self.header.get("__metadata__")

    def get_slice(self, name):
        if name not in self.header or name == "__metadata__":
            raise KeyError(name)
        return _Slice(self, name)

    def get_tensor(self, name):
        return self.get_slice(name)[...]


class _Slice:
    def __init__(self, reader, name):
        self.reader, self.name = reader, name

    def get_shape(self):
        return list(self.reader.header[self.name]["shape"])

    def get_dtype(self):
        return self.reader.header[self.name]["dtype"]

    def __getitem__(self, index):
        meta = self.reader.header[self.name]
        first, last = meta["data_offsets"]
        if first == last:
            return torch.empty(meta["shape"], dtype=DTYPES[meta["dtype"]])[index]
        data = memoryview(self.reader.mapping)[self.reader.offset + first:self.reader.offset + last]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            tensor = torch.frombuffer(data, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"])
        return tensor[index]


def preserve_frozen_cpu_views(module, offload_hook):
    """Restore frozen CPU views without another CPU copy or NF4 meta transfer."""
    frozen = []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad or parameter.device.type != "cpu":
            raise ValueError("Only frozen CPU weights can be file-backed")
        parent_name, _, leaf = name.rpartition(".")
        owner = module.get_submodule(parent_name) if parent_name else module
        metadata = dict(parameter.__dict__)
        if "quant_state" in metadata:
            metadata["quant_state"] = deepcopy(metadata["quant_state"])
        frozen.append((owner, leaf, parameter.data, type(parameter), metadata))
    buffers = []
    for name, value in module.named_buffers():
        parent_name, _, leaf = name.rpartition(".")
        owner = module.get_submodule(parent_name) if parent_name else module
        buffers.append((owner, leaf, value))

    def restore():
        for owner, leaf, data, cls, metadata in frozen:
            if cls.__name__ == "Params4bit":
                accepted = {"quant_state", "blocksize", "compress_statistics", "quant_type",
                            "quant_storage", "module", "bnb_quantized"}
                kwargs = {key: value for key, value in metadata.items() if key in accepted}
                kwargs["quant_state"] = deepcopy(metadata["quant_state"])
                value = cls(data, requires_grad=False, **kwargs)
                for key, field in metadata.items():
                    if key not in accepted:
                        setattr(value, key, field)
                owner._parameters[leaf] = value
                if hasattr(owner, "quant_state"):
                    owner.quant_state = value.quant_state
            else:
                value = torch.nn.Parameter(data, requires_grad=False)
                value.__dict__.update(metadata)
                owner._parameters[leaf] = value
        for owner, leaf, value in buffers:
            owner._buffers[leaf] = value

    offload_hook.offload = restore
    return restore
