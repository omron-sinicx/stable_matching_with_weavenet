"""Helpers for building `calc_residual` masks consumed by
`weavenet.model.WeaveNet` / `MatchingNet`.

`weavenet.MatchingNet` accepts `calc_residual: Optional[List[bool]]`
where ``True`` at index ``l`` means "add the most recently kept variable
to the output of layer ``l``". The library does not ship with helpers to
generate common patterns; this module collects the ones we use, so
configs do not have to inline long boolean lists.
"""
from __future__ import annotations

from typing import List


def every_2_residual_pattern(L: int) -> List[bool]:
    r"""Residual mask that matches the paper's "every two FW layers"
    rule (`arXiv:2310.12515`, Section 3).

    For a depth-``L`` WeaveNet, returns

    .. code-block::

        [F, F, T, F, T, F, ..., T, F]   # length L

    so the FIRST residual *save* happens after layer 0 (via
    ``keep_first_var_after=0``) and the FIRST residual *addition* happens
    after layer 2. Pair with ``keep_first_var_after=0`` on the WeaveNet
    constructor.

    Mirrors the legacy implementation::

        for i, FWLayer in enumerate(self.encoders):
            Z = FWLayer(Z)
            if self.use_resnet and i % 2 == 0:
                if Z_keep is not None:
                    Z = Z + Z_keep
                Z_keep = Z

    Args:
       L: number of WeaveNet layers (``len(output_channels_list)``).

    Returns:
       A list of ``bool`` of length ``L``.
    """
    return [(i > 0 and i % 2 == 0) for i in range(L)]
