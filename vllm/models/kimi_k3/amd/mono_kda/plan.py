# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch geometry, ticket order and workspace layout of the mono KDA launch.

The workspace is one device allocation: the control words, then the bf16
in_proj output and the bf16 gated-norm output of the step's rows. Every offset
is a compile-time constant.
"""

from dataclasses import dataclass

M_MAX = 16  # a step's rows: one MFMA tile of tokens
WAVE = 64
THREADS = 256
N_WAVES = THREADS // WAVE
TILE = 16  # GEMV output columns a ticket
HEAD_DIM = 128
CONV_WIDTH = 4
SPIN_SLEEP = 2

# Each control word sits on its own 128-byte line: words that different
# workgroups poll and bump never share a line.
LINE_WORDS = 32
W_TICKET = 0
W_SHARED = LINE_WORDS  # in_proj tiles of f_a and beta done
W_DECODE = 2 * LINE_WORDS  # decode tasks done
W_HEAD = 3 * LINE_WORDS  # in_proj tiles of head h done, one line a head


@dataclass(frozen=True)
class Dims:
    """TP-local shapes of one KDA layer."""

    heads: int  # local KDA heads
    hidden: int

    @property
    def proj(self) -> int:
        """P: the local q / k / v / g width."""
        return self.heads * HEAD_DIM

    @property
    def n_in(self) -> int:
        """in_proj_qkvgfab rows: q, k, v, g (P each), f_a (128), beta (heads),
        zero padding to a multiple of 16 (vLLM's layout)."""
        n = 4 * self.proj + HEAD_DIM + self.heads
        return n + -n % TILE

    @property
    def proj_row(self) -> int:
        """Row stride of the workspace's in_proj output: whole 128-byte lines, so
        no line holds two heads' columns. A decode task reads only lines its
        counters guard; a line read early would sit stale in its XCD's L2."""
        return -(-self.n_in // 64) * 64

    @property
    def fa_col(self) -> int:
        return 4 * self.proj

    @property
    def beta_col(self) -> int:
        return 4 * self.proj + HEAD_DIM

    @property
    def shared_tiles(self) -> int:
        """in_proj tiles every head needs: f_a, beta and the padding."""
        return (self.n_in - self.fa_col) // TILE

    @property
    def head_tiles(self) -> int:
        """in_proj tiles of one head: its q, k, v and g columns."""
        return 4 * HEAD_DIM // TILE

    @property
    def in_tiles(self) -> int:
        return self.n_in // TILE

    @property
    def out_tiles(self) -> int:
        return self.hidden // TILE

    def tickets(self, m: int) -> int:
        """in_proj tiles, then a decode task a (head, row), then o_proj tiles."""
        return self.in_tiles + self.heads * m + self.out_tiles


def ctrl_words(heads: int) -> int:
    return W_HEAD + heads * LINE_WORDS


def ws_layout(d: Dims) -> tuple[dict[str, int], int]:
    """Byte offsets of the workspace buffers and the total size."""
    sizes = dict(
        ctrl=ctrl_words(d.heads) * 4,
        proj=M_MAX * d.proj_row * 2,
        o=M_MAX * d.proj * 2,
    )
    offs, total = {}, 0
    for k, size in sizes.items():
        offs[k] = total
        total += (size + 255) // 256 * 256
    return offs, total
