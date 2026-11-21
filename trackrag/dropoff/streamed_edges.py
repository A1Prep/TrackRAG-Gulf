"""Frozen vocabulary operations for the explicit low-memory CUDA runtime.

The complete output vocabulary is retained. Its BF16 matrix is read in bounded
row chunks; no approximation, top-k vocabulary or learned-weight substitution is
used. Only gradients with respect to hidden states are returned.
"""

import torch


class CPUEmbedding(torch.nn.Module):
    def __init__(self, original, device):
        super().__init__()
        if original.weight.requires_grad or original.weight.device.type != "cpu":
            raise ValueError("Embedding must be frozen and CPU-backed")
        self.original = original
        self.destination = torch.device(device)
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim

    @property
    def weight(self):
        return self.original.weight

    def forward(self, indices):
        with torch.no_grad():
            return self.original(indices.to("cpu")).to(self.destination)


class FrozenHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, chunk_rows):
        if weight.requires_grad or weight.device.type != "cpu":
            raise ValueError("Streamed head must be frozen and CPU-backed")
        if hidden.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Streamed head supports float32 and bfloat16 hidden states")
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError("Head chunk_rows must be a positive integer")
        if weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
            raise ValueError("Hidden-state and output-projection dimensions differ")
        ctx.save_for_backward(weight)
        ctx.chunk_rows = chunk_rows
        ctx.input_shape = hidden.shape
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = torch.empty((flat.shape[0], weight.shape[0]), device=hidden.device, dtype=hidden.dtype)
        for start in range(0, weight.shape[0], chunk_rows):
            end = min(start + chunk_rows, weight.shape[0])
            block = weight[start:end].to(device=hidden.device, dtype=hidden.dtype)
            logits[:, start:end] = flat @ block.T
            del block
        return logits.reshape(*hidden.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        flat = grad_output.reshape(-1, grad_output.shape[-1])
        grad = torch.zeros((flat.shape[0], weight.shape[1]), device=flat.device, dtype=torch.float32)
        for start in range(0, weight.shape[0], ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, weight.shape[0])
            block = weight[start:end].to(device=flat.device, dtype=torch.float32)
            grad.add_(flat[:, start:end].float() @ block)
            del block
        return grad.to(grad_output.dtype).reshape(ctx.input_shape), None, None


class CPUBackedHead(torch.nn.Module):
    def __init__(self, original, chunk_rows=4096):
        super().__init__()
        if original.bias is not None or original.weight.requires_grad:
            raise ValueError("Streamed head requires a frozen, bias-free output projection")
        if original.weight.device.type != "cpu":
            raise ValueError("Streamed output projection must start on CPU")
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= 8192:
            raise ValueError("Head chunk_rows must be an integer between 1 and 8192")
        self.weight = original.weight
        self.chunk_rows = chunk_rows
        self.in_features = original.in_features
        self.out_features = original.out_features

    def forward(self, hidden):
        return FrozenHeadFunction.apply(hidden, self.weight, self.chunk_rows)
