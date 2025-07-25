# PyTorch Backend

```{note}
Note:
This feature is currently experimental, and the related API is subjected to change in future versions.
```

To enhance the usability of the system and improve developer efficiency, TensorRT-LLM launches a new experimental backend based on PyTorch.

The PyTorch backend of TensorRT-LLM is available in version 0.17 and later. You can try it via importing `tensorrt_llm._torch`.

## Quick Start

Here is a simple example to show how to use `tensorrt_llm.LLM` API with Llama model.

```{literalinclude} ../../examples/llm-api/quickstart_example.py
    :language: python
    :linenos:
```

## Features

- [Sampling](./torch/features/sampling.md)
- [Quantization](./torch/features/quantization.md)
- [Overlap Scheduler](./torch/features/overlap_scheduler.md)
- [Feature Combination Matrix](./torch/features/feature_combination_matrix.md)

## Developer Guide

- [Architecture Overview](./torch/arch_overview.md)
- [Adding a New Model](./torch/adding_new_model.md)
- [Examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/pytorch/README.md)

## Key Components

- [Attention](./torch/attention.md)
- [KV Cache Manager](./torch/kv_cache_manager.md)
- [Scheduler](./torch/scheduler.md)

## Known Issues

- The PyTorch backend on SBSA is incompatible with bare metal environments like Ubuntu 24.04. Please use the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for optimal support on SBSA platforms.
