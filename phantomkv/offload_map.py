"""Block <-> chunk mapping for offloaded KV, pure Python (unit-tested without vLLM).

vLLM's OffloadingConnector may offload in chunks of `blocks_per_chunk` GPU
blocks. A transfer names the GPU blocks, the CPU chunks, and the logical index
of the first GPU block within its request; the first chunk may be entered
part-way (vLLM's CPU worker skips `first_index % blocks_per_chunk` sub-blocks).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence


def chunk_pairs(gpu_blocks: Sequence[int], cpu_chunks: Sequence[int], first_index: int,
                blocks_per_chunk: int) -> Iterator[tuple[int, int, int]]:
    """Yield (gpu_block, cpu_chunk, sub_block) for every GPU block."""
    if blocks_per_chunk < 1:
        raise ValueError("blocks_per_chunk must be >= 1")
    base = first_index // blocks_per_chunk
    for j, gb in enumerate(gpu_blocks):
        logical = first_index + j
        k = logical // blocks_per_chunk - base
        if k >= len(cpu_chunks):
            raise ValueError(f"GPU block {gb} (logical {logical}) needs chunk #{k}, "
                             f"but only {len(cpu_chunks)} chunks were given")
        yield gb, cpu_chunks[k], logical % blocks_per_chunk
