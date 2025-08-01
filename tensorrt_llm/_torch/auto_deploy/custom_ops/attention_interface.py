"""Attention Interface to handle various attention operators and cache operations.

This module provides an interface between the high-level runtime and cache management system and
the low-level functional attention operators. The interface is designed to provide a homogeneous
object-oriented interface to the high-level runtime via the SequenceInfo dataclass. The SequenceInfo
is also responsible for functionalizing information about the sequence and pass it on the the
various attention interface. The AttentionDescriptor is the main interface to the attention operator
and operates on a purely functional paradigm that is compatible with the torch custom op system.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Type, Union

import torch
from torch._ops import OpOverloadPacket
from torch.export import Dim
from torch.fx import Node


@dataclass
class CacheConfig:
    """A dataclass to hold information how to configure the cache."""

    dtype: Optional[torch.dtype] = None


@dataclass
class SequenceInfo:
    """A dataclass to hold information about how the sequence is laid out and stored in cache.

    We assume the sequence + cache is laid out in the following way. Also note that we differentiate
    between arguments that are originally part of the model/graph and arguments that are needed for
    the attention operator when we switch to cached+flattened attention.

    # ORIGINAL MODEL ARGUMENTS #####################################################################
    - input_ids: [id_0, ..., id_{s_total-1}]
      flattened sequence of [b, 1] or [1, s_total]. We use [b, 1] to denote generate-only batches.
    - position_ids: [pos_0, ..., pos_{s_total-1}]
      flattened sequence of [b, 1] or [1, s_total] indicating absolute position ids for every token
      in the input_ids sequence. We use [b, 1] to denote generate-only batches.

    NOTE: ``input_ids`` and ``position_ids`` are initially expected to be of shape [b, seq_len]
    before we switch to cached+flattened attention.

    # EXTRA ARGUMENTS NEEDED FOR ATTENTION OPERATORS FOR FLATTENED SEQUENCES + CACHES ##############
    - seq_len: [s_0, s_1, ..., s_{b-1}] such that s_total = sum(s_i)
      Describes how long each sequence is. For example,
      input_ids[:s_0] will correspond to sequence 0 in the batch and input_ids[s_0:s_1] will
      correspond to sequence 1 in the batch.
    - input_pos: [pos_0, ..., pos_{b-1}]
      Corresponds to the total number of tokens that has been already been cached for each sequence
      in the batch.
    - cache_loc: [c0, ...., c_{np-1}] where np is total number of pages allocated to describe all
      sequences in the batch.
    - pages_per_seq: [ps_0, ps_1, ..., ps_{b-1}] where ps_i is the number of pages allocated for
      sequence i. Note that, for example, cache_loc[p_0:p_1] will correspond to the pages associated
      with sequence 1 in the batch.

    ################################################################################################

    Here are a couple of notes to emphasize this notation:

    - The total number of allocated token space for sequence i is given by ps_i * page_size. This is
      the total number of tokens that can be cached for each sequence.

    - NOTE: It must hold that pos_i + s_i <= ps_i * page_size for all i in [0, b-1]. Moreover, it is
      the responsibility of the cache manager and/or runtime to ensure sufficient page allocation
      for each sequence.

    """

    ## USE TO INITIALIZE DATA CLASS  ###############################################################
    # max_seq_len corresponds the maximum number of tokens in any sequence. It includes the tokens in the
    # input sequence and the tokens generated by the model.
    max_seq_len: int = 1
    # max_batch_size corresponds to the maximum number of sequences (or requests) that the model can process.
    max_batch_size: int = 1
    # page_size is the granularity with which the cache pages are allocated for a paged kv cache.
    # For an unpaged cache, the page size should be set to max_seq_len.
    # Also note that two sequences in a batch can not share a page.
    page_size: int = 0
    # max_num_tokens is the maximum number of tokens that the model can process across all sequences in the batch.
    # If a batch is composed of context-only requests of input sequence length ISL,
    # then the maximum number of sequences possible in the batch is min (max_batch_size, max_num_tokens // ISL).
    # Similarly, if a batch is composed of generate-only requests,
    # then the maximum number of sequences possible in the batch is min (max_batch_size, max_num_tokens).
    max_num_tokens: Optional[int] = None

    ## [UPDATE WITH CARE] TENSOR FIELDS THAT WILL BE PASSED TO PREPARE_METADATA OP #################
    # input_ids MUST ALWAYS BE THE FIRST FIELD
    input_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, dtype=torch.int))
    position_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, dtype=torch.long))

    seq_len: torch.Tensor = field(default_factory=lambda: torch.ones(1, dtype=torch.int))
    input_pos: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.int))
    cache_loc: torch.Tensor = field(default_factory=lambda: torch.arange(1, dtype=torch.int))
    pages_per_seq: torch.Tensor = field(default_factory=lambda: torch.ones(1, dtype=torch.int))
    ################################################################################################

    ## PRIVATE FIELDS ##############################################################################
    _sequence_lengths: List[int] = field(default_factory=list)
    _num_pages: int = 1

    def __post_init__(self):
        if self.page_size < 1:
            self.page_size = self.max_seq_len

        # NOTE (lucaslie): WAR to address issue when using flashinfer attention with
        # (max_batch_size, max_seq_len) input in trtllm runtime.
        # see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
        max_seq_len_adjusted = self.max_seq_len + 1

        if self.max_num_tokens is None or self.max_num_tokens < 1:
            self.max_num_tokens = self.max_batch_size * max_seq_len_adjusted
        # if the provided max_num_tokens is less than the max_batch_size * max_seq_len,
        # we use the provided max_num_tokens to calculate the number of pages
        total_tokens = min(self.max_num_tokens, self.max_batch_size * max_seq_len_adjusted)
        # Num pages can not be less than max_batch_size.
        self._num_pages = max(
            self.max_batch_size,
            (total_tokens) // self.page_size + (total_tokens % self.page_size > 0),
        )
        self.input_ids = torch.ones(self.max_batch_size, 1, dtype=torch.int)
        self.position_ids = torch.zeros(self.max_batch_size, 1, dtype=torch.long)
        self.seq_len = torch.empty(self.max_batch_size, dtype=torch.int)
        self.input_pos = torch.empty_like(self.seq_len)
        self.cache_loc = torch.empty(self.num_pages, dtype=torch.int)
        self.pages_per_seq = torch.empty_like(self.seq_len)
        assert self.num_pages >= self.max_batch_size, (
            "num_pages must be greater than max_batch_size"
        )
        # dynamic shape descriptors for tensor args
        self._dynamic_shapes: Optional[Tuple[Dict[str, Dim]]] = None

        # keep a list-like object of sequence lengths for simplicity as well
        self._sequence_lengths = [0] * self.max_batch_size

        # indicator if extra args are activated that are needed for cached attention backends
        self._is_cached_attn = False

        # call reset once to initialize the tensors
        self.reset()

    @property
    def device(self) -> torch.device:
        return self.input_pos.device

    @property
    def args(self) -> Tuple[torch.Tensor, ...]:
        args = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                args.append(val)
            if len(args) >= self._num_uncached_attn_args and not self._is_cached_attn:
                break
        return tuple(args)

    @property
    def _num_uncached_attn_args(self) -> int:
        """Return the number of original graph arguments expected by the model."""
        return 2

    @property
    def _cached_attn_arg_names(self) -> List[str]:
        """Return extra arg names for the prepare_metadata op beyond input_ids and position_ids.

        These extra args are needed once we switch from regular attention to inserting cached
        attention ops in the model.
        """
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)][
            self._num_uncached_attn_args :
        ]

    @property
    def dynamic_shapes(self) -> Tuple[Dict[str, Dim]]:
        """Return dynamic shapes of sequence info tensors.

        NOTE: will be lazily initialized since the Dim object is not picklable for multi-processing.
        """
        if self._dynamic_shapes is None:
            # set up shape for input_ids and position_ids
            dynamic_shapes = ({}, {})
            if self.max_batch_size > 1:
                dynamic_shapes[0][0] = Dim("batch_size", max=self.max_batch_size)
            dynamic_shapes[0][1] = Dim("seq_len", max=self.max_seq_len)
            # set up shape for position_ids (same as input_ids)
            dynamic_shapes[1].update(dynamic_shapes[0])
            # set up shape for extra args
            if self._is_cached_attn:
                dynamic_shapes += ({},) * len(self._cached_attn_arg_names)
            self._dynamic_shapes = dynamic_shapes
        return self._dynamic_shapes

    @property
    def num_sequences(self) -> int:
        return len(self._sequence_lengths)

    @property
    def sequence_lengths(self) -> List[int]:
        return self._sequence_lengths

    @property
    def input_positions(self) -> List[int]:
        return self.input_pos[: self.num_sequences].tolist()

    @property
    def is_generate(self) -> bool:
        return all(sl == 1 for sl in self.sequence_lengths)

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @num_pages.setter
    def num_pages(self, value):
        self._num_pages = value
        # update the cache_loc tensor
        self.cache_loc.resize_(value)

    @property
    def is_paged(self) -> bool:
        return self.page_size < self.max_seq_len

    @property
    def page_assignments(self) -> List[List[int]]:
        """Return the page assignments for each sequence."""
        pages_per_seq = self.pages_per_seq[: self.num_sequences].tolist()
        return [
            c_loc_one_seq.tolist()
            for c_loc_one_seq in torch.split(self.cache_loc[: sum(pages_per_seq)], pages_per_seq)
        ]

    @classmethod
    def _get_sanitized_seq_len(cls, input_ids: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Sanitize sequence lengths.

        We want to cover the following scenarios with this function:

        1. Pre-fill:
            input_ids: [1, s_total, ...]
            seq_len: [s_0, s_1, ..., s_{b-1}, 0, 0, ..., 0]
            ---> returns [s_0, s_1, ..., s_{b-1}]
        2. Decode:
            input_ids: [b, 1, ...]
            seq_len: [1, 1, ..., 1, 0, 0, ..., ..., ..., ..., 0]
                     |---- b ----|--- (max_batch_size - b) ---|
            --> returns [1,] * b
        3. Decode in Cudagraph:
            input_ids: [b_cudagraph, 1, ...]
            seq_len: [1, 1, ..., 1, 0, 0, ..., ..., ..., ..., 0]
                     |---- b ----|--- (max_batch_size - b) ---|

            --> returns [1,] * b_cudagraph
            Here b <= b_cudagraph. We want to make sure that the seq_len is one-padded to
            b_cudagraph.

            # TODO: I could see one possible issue with this approach in the future.
            # If we have b < b_cudagraph we now one-pad. However, we don't pad the cache location
            # information. What could happen is that the for the padded sequences the cache location
            # tensors point to allocated pages. This could lead to a situation where we write into
            # allocated cache pages polluting the cache of other sequences. Now this is not an issue
            # if we write the dummy sequences into unallocated cache pages... One fix could be to
            # pad not only the seq len but also pad the cache locations by just repeating the last
            # valid cache location in the batch. This would ensure that the dummy sequences just
            # repeats valid computation...
        """
        _, s = input_ids.shape[:2]
        num_seq = cls._get_sanitized_num_sequences(input_ids, seq_len)
        if s > 1:
            return seq_len[:num_seq].detach().clone()
        else:
            return torch.ones(num_seq, dtype=seq_len.dtype, device=seq_len.device)

    @staticmethod
    def _get_sanitized_num_sequences(input_ids: torch.Tensor, seq_len: torch.Tensor) -> int:
        """Get number of sequences.

        We makes sure that this function is compatible with both torch graph capture and cudagraph.
        Both can be a bit temparamental when trying to extract the number of sequences from a tensor
        with max_batch_size or max_batch_size*max_seq_len.
        """
        b, s = input_ids.shape[:2]
        if s > 1:
            num_seq = torch.sum(seq_len > 0)
            assert seq_len[num_seq:].sum() == 0, "seq_len should be zero-padded"
        else:
            num_seq = b
        return num_seq

    def switch_to_cached_attn_inputs(self) -> List[str]:
        """Switch to inputs for cached+flattened attention operators.

        Returns:
            List[str]: List of new argument names that are now activated.

        This function will change the inputs provided by the interface from the arguments expected
        by regular attention in PyTorch (SDPA-style) to the arguments needed once we use attention
        operators with cache support and flattened sequences.

        NOTE: The graph inference optimizer is responsible for ensuring the the new inputs are
        correctly reflected in the graph after this function is called.
        """
        assert not self._is_cached_attn, "Cached+flattened attention already activated"
        self._is_cached_attn = True
        return self._cached_attn_arg_names

    def to(self, *args, **kwargs) -> None:
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(*args, **kwargs))

    def sync(self, other: "SequenceInfo") -> None:
        for f in fields(self):
            val = getattr(self, f.name)
            val_other = getattr(other, f.name)
            if f.name in ["input_ids", "position_ids"]:
                setattr(self, f.name, val_other.to(self.device))
            elif f.name == "_sequence_lengths":
                self._sequence_lengths = val_other
            elif isinstance(val, torch.Tensor):
                val[: len(val_other)] = val_other.to(self.device)
            else:
                assert val == val_other, f"Field {f.name} mismatch: {val} != {val_other}."

    def reset(self) -> None:
        """Reset the sequence information.

        After reset the sequence information should correspond to a "generate-only" batch of
        sequences (b, s==1) without cache history.
        """
        # reset input_pos
        self.input_pos.zero_()

        # set a dummy sequence corresponding to a generate-only batch (will also reset position_ids)
        self.nest_sequences(torch.zeros(self.max_batch_size, 1, dtype=torch.int))

        # reset cache information
        self.cache_loc[:] = torch.arange(self.num_pages, dtype=torch.int, device=self.device)
        self.pages_per_seq.fill_(1)

    def set_example_sequence(self) -> None:
        """Set an example sequence useful for testing and export purposes."""
        self.reset()
        bs, seq_len = min(2, self.max_batch_size), min(4, self.max_seq_len)
        input_ids = torch.ones(
            bs,
            seq_len,
            dtype=torch.int,
            device=self.device,
        )
        self.nest_sequences(input_ids)

        # unflatten if we are not yet using cached+flattened attention
        if not self._is_cached_attn:
            self.input_ids = self.input_ids.view(bs, seq_len)
            self.position_ids = self.position_ids.view(bs, seq_len)

    def _set_max_num_tokens_sample(self) -> None:
        """Set an example sequence with max_num_tokens."""
        self.reset()
        seq_len = self.max_num_tokens // self.max_batch_size
        input_ids = torch.ones(
            self.max_batch_size,
            seq_len,
            dtype=torch.int,
            device=self.device,
        )
        self.pages_per_seq.fill_(seq_len // self.page_size)
        self.nest_sequences(input_ids)

    def set_generate_only_batch(self) -> None:
        """Set an example sequence for generate-only batch.

        NOTE: this batch is already formatted as [b, 1] in both original and in the cached attention
        mode. So we don't need to do anything mode-specific here.
        """
        self.reset()
        self.nest_sequences([[1]] * self.max_batch_size)

    def _update_position_ids(self) -> None:
        # set new position_ids as new tensor from input_pos and seq_len via torch.arange
        position_ids_list = [
            num
            for in_pos, seq_len in zip(self.input_positions, self.sequence_lengths)
            for num in range(in_pos, in_pos + seq_len)
        ]
        self.position_ids = torch.tensor(position_ids_list, dtype=torch.long).to(self.device)

        # use [b,1] shape to indicate generate-only batch, otherwise use [1,total_len]
        if self.is_generate:
            self.position_ids = self.position_ids.view(-1, 1)
        else:
            self.position_ids = self.position_ids.view(1, -1)

    def nest_sequences(self, input_ids: Sequence[Sequence[int]]) -> None:
        """Create and store a flattened list of input_ids from the provided list of sequences.

        This i/f will also update any relevant sequence information.
        """
        # set new sequence lengths
        seq_lens = [len(ids) for ids in input_ids]
        self.seq_len.zero_()
        self.seq_len[: len(seq_lens)].copy_(torch.tensor(seq_lens), non_blocking=True)
        # We'll preserve the dtype of the input_ids tensor if it is a tensor, otherwise we'll use int
        dtype = input_ids.dtype if isinstance(input_ids, torch.Tensor) else torch.int
        # set new input_ids as new tensor from flattened input_ids
        ids_list = [
            val
            for lst in input_ids
            for val in (lst.detach().tolist() if isinstance(lst, torch.Tensor) else lst)
        ]
        self.input_ids = torch.tensor(ids_list, dtype=dtype).to(self.device)

        # set derivative properties
        self._sequence_lengths = seq_lens

        # use [b,1] shape to indicate generate-only batch, otherwise use [1,total_len]
        if self.is_generate:
            self.input_ids = self.input_ids.view(-1, 1, *self.input_ids.shape[1:])
        else:
            self.input_ids = self.input_ids.view(1, -1, *self.input_ids.shape[1:])

        # update position_ids
        self._update_position_ids()

    def unnest_sequences(self, t_nested: torch.Tensor) -> List[torch.Tensor]:
        t_squeezed = t_nested.squeeze(1) if self.is_generate else t_nested.squeeze(0)
        return list(torch.split(t_squeezed, self.sequence_lengths))

    def update_pos(self, seq_len: Union[torch.Tensor, List[int], int], reset: bool = False) -> None:
        """Update the starting position for each sequence in the cache.

        If ``reset=True`, ``input_pos`` will be reset to zero before updating.
        """
        if not isinstance(seq_len, torch.Tensor):
            seq_len = torch.tensor(seq_len, dtype=torch.int)
        bs = len(seq_len) if seq_len.dim() > 0 else self.max_batch_size

        if reset:
            self.input_pos[:bs] = seq_len.to(self.device)
        else:
            self.input_pos[:bs] += seq_len.to(self.device)

        # update position_ids
        self._update_position_ids()

    def assign_cache_loc(self, page_assignments: Sequence[Sequence[int]]) -> None:
        """Set the cache location and pages_per_seq tensors from page assignments."""
        cache_loc_flat = torch.tensor(
            [p_idx for pages in page_assignments for p_idx in pages], dtype=torch.int
        )
        self.cache_loc[: len(cache_loc_flat)].copy_(cache_loc_flat, non_blocking=True)

        pages_per_seq = torch.tensor([len(p) for p in page_assignments], dtype=torch.int)
        self.pages_per_seq[: len(pages_per_seq)].copy_(pages_per_seq, non_blocking=True)


Constant = Union[int, float, str, None]


class MHACallable(Protocol):
    def __call__(
        self,
        *qkv_metadata_and_caches: Union[torch.Tensor, Constant],
    ) -> torch.Tensor: ...


class PrepareMetadataCallable(Protocol):
    def __call__(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        seq_len: torch.Tensor,
        input_pos: torch.Tensor,
        cache_loc: torch.Tensor,
        pages_per_seq: torch.Tensor,
        page_size: int,
    ) -> List[torch.Tensor]: ...


class GetCacheCallable(Protocol):
    def __call__(self, sequence_info: SequenceInfo) -> torch.Tensor: ...


class GetBufferCallable(GetCacheCallable):
    pass


CacheInitializerDict = Dict[str, GetCacheCallable]
BufferInitializerDict = Dict[str, GetBufferCallable]
AttentionLayout = Literal["bsnd", "bnsd"]


class AttentionDescriptor(ABC):
    """An interface to define a functional attention operator.

    The main logic is contained with the actual attention op as well as the prepare_metadata op. The
    prepare_metadata op is responsible for converting the standardized sequence info into metadata
    specific to the attention op.
    """

    @classmethod
    @abstractmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""

    @classmethod
    @abstractmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the source op and the cached attention op."""

    @classmethod
    @abstractmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""

    @classmethod
    @abstractmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""

    @classmethod
    @abstractmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op .

        The attention_op should follow the below signature:

        ```
        def attention_op(
            *qkv,       # list of tensors corresponding to Q, K, V as in source attention op
            *metadata,  # global info about the sequences as returned by the prepare_metadata op
            *caches,    # contains layer-specific caches per provided cache initializers
            *buffers,   # global buffers used by the attention op as provided by buffer initializers
            *constants, # basic arguments (int, float, str, None) added as CONSTANTS in the graph
        ) -> torch.Tensor: ...
        ```

        **Note that the attention op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**

        **Note that the `qkv` tuple should be consistent across both the cached attention
        op and the source attention op that it is replacing.**

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        """Get the prepare_metadata op.

        The prepare_metadata op should follow the below signature:

        ```
        def prepare_metadata(
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            seq_len: torch.Tensor,
            input_pos: torch.Tensor,
            cache_loc: torch.Tensor,
        ) -> List[torch.Tensor]: ...
        ```
        The metadata should contain all necessary global information required for the underlying
        attention op to process the input sequence and the returned list of tensors will be passed
        on to each invocation of the attention op in the graph.

        prepare_metadata is called once at the beginning of the forward pass.

        **Note that the prepare_metadata op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**
        """

    @classmethod
    @abstractmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize the caches.

        The key corresponds to the argument name used in the attention op signature. The function
        key doesn't need to be unique across multiple attention nodes in the graph. The key used to
        describe the cache in the graph will be patched with the attention node index to ensure
        uniqueness.

        ``get_cache_initializers`` will be called *once* during cache initialization and before
        the initial forward pass for each attention op detected in the graph. The caches will be
        managed by the global CacheManager and passed back to the attention op during the forward
        pass.

        If the cache initializer requires information about the attention op, it can retrieve
        the necessary information from the source attention node and cache config.
        """

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize buffers.

        The key corresponds to the buffer name used in the graph module and will **not**
        be patched unlike a cache key. Hence, it is a **global** key that is shared across all
        attention ops in the model much like a regular buffer in an nn.Module. That means if this
        i/f is called for multiple attention ops, the same buffer will be shared across all of them
        if this function provides the same key multiple times.

        Buffers are initialize *once* after the model initialization and before the initial forward
        pass for each attention op detected in the graph. The buffer will be managed by the global
        CacheManager and passed back to the attention op during the forward pass.

        If the buffer initializer requires information about the attention op, it can retrieve
        the necessary information from the source attention node.
        """

    @classmethod
    @abstractmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Provide a list of constant arguments to be passed to the attention op.

        The constant arguments are passed to the attention op as additional arguments after the
        caches and buffers. The constants are expected to be of type int, float, str, or None.
        """


class AttentionRegistry:
    """A simple registry to look up different attention implementations."""

    _attention_registry: Dict[str, Type["AttentionDescriptor"]] = {}

    @classmethod
    def register(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        def decorator(attention_cls: Type["AttentionDescriptor"]):
            assert kernel_source not in cls._attention_registry, (
                f"Attention source {kernel_source} already registered."
            )
            cls._attention_registry[kernel_source] = attention_cls
            return attention_cls

        return decorator

    @classmethod
    def get(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        assert cls.has(kernel_source), f"Attention source {kernel_source} not registered."
        return cls._attention_registry[kernel_source]

    @classmethod
    def has(cls, kernel_source: str) -> bool:
        return kernel_source in cls._attention_registry
