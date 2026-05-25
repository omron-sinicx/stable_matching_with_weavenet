"""Legacy WeaveNet compatibility layer.

Translates the paper-time `MatcherWeaveNet` (from
`docs/papers/legacy_code/src/networks.py` of the project repo) onto the
current PyPI weavenet abstractions, so paper hyperparameters land on the
right architecture.

What this provides:
- :class:`LegacyMaxPoolEncoder` — re-implements the legacy
  ``EncoderMaxPool`` (Conv→MaxPool→Concat-with-x→Conv) in NHWC / Linear
  form.
- :class:`LegacyUnitListGenerator` — factory wiring
  :class:`LegacyMaxPoolEncoder` into :class:`~weavenet.model.Unit` so it
  can be dropped into :class:`~weavenet.model.MatchingNet`.
- :class:`LegacyWeaveNet` — like :class:`~weavenet.model.WeaveNet` but
  using the legacy-style units above.
- :class:`LegacyMeanAggregator` — replaces
  :class:`~weavenet.layers.DualSoftmaxSqrt`: averages the two streams in
  raw-logit form, exactly as ``MatcherWeaveNet.forward`` does at the
  network output.
- :func:`legacy_residual_pattern` — produces the
  ``[F, F, T, F, T, F, …]`` mask matching the legacy
  ``if use_resnet and i%2==0`` rule.

The remaining structural differences vs paper-time code (Linear vs
Conv2d-1x1, BatchNormXXC vs BatchNorm2d, parallel two-stream forward vs
batch-concatenated forward) are functionally equivalent and behave the
same on dense matching problems.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn

from .layers import BatchNormXXC, CrossConcat, Interactor
from .model import (
    ExclusiveElementsOfUnit,
    MatchingNet,
    Unit,
    WeaveNetUnitListGenerator,
)


class LegacyMaxPoolEncoder(nn.Module):
    """NHWC / Linear port of legacy ``EncoderMaxPool``.

    Legacy reference (``legacy_code/src/networks.py:259``)::

        z = self.conv_max(x)              # Conv2d 1x1, in -> mid
        z = max_pool_concat(x, z, dim)    # cat([x, maxpool(z, dim)], dim=channel)
        z = self.conv(z)                  # Conv2d 1x1, (in+mid) -> out, bias=False
        # BN + activation handled by the outer Unit wrapper.

    Our port keeps the exact arithmetic but operates in NHWC layout, where
    the channel axis is ``dim=-1``, matching the rest of weavenet.
    """

    def __init__(self, in_channels: int, mid_channels: int, output_channels: int) -> None:
        super().__init__()
        # legacy: conv_max = Conv2d(in_channels, mid_channels, 1x1, bias=True)
        self.first_process = nn.Linear(in_channels, mid_channels)
        # legacy: conv = Conv2d(in_channels + mid_channels, output_channels, 1x1, bias=False)
        self.conv = nn.Linear(in_channels + mid_channels, output_channels, bias=False)

    def forward(self, x: torch.Tensor, dim_target: int) -> torch.Tensor:
        z = self.first_process(x)  # (..., N, M, mid)
        z_max = z.amax(dim=dim_target, keepdim=True).expand_as(z)
        z = torch.cat([x, z_max], dim=-1)  # (..., N, M, in + mid)
        return self.conv(z)  # (..., N, M, out)


class LegacyUnitListGenerator(WeaveNetUnitListGenerator):
    """Generates :class:`Unit`s wrapped around :class:`LegacyMaxPoolEncoder`.

    The Unit order is ``'ena'`` (encoder → normalizer → activator), with
    :class:`BatchNormXXC` and :class:`~torch.nn.PReLU` — the same wrapping
    used by :class:`~weavenet.model.WeaveNetUnitListGenerator`.
    """

    def _build(self, in_channels_list: List[int]) -> List[Unit]:
        return [
            Unit(
                LegacyMaxPoolEncoder(in_ch, mid_ch, out_ch),
                "ena",
                BatchNormXXC(out_ch),
                nn.PReLU(),
            )
            for in_ch, mid_ch, out_ch in zip(
                in_channels_list, self.mid_channels_list, self.output_channels_list
            )
        ]


class LegacyWeaveNet(MatchingNet):
    """:class:`~weavenet.model.WeaveNet` variant using
    :class:`LegacyUnitListGenerator` — i.e., the legacy paper's encoder
    structure rather than the modern :class:`SetEncoderPointNet`.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels_list: List[int],
        mid_channels_list: List[int],
        calc_residual: Optional[List[bool]] = None,
        keep_first_var_after: int = 0,
        exclusive_elements_of_unit: ExclusiveElementsOfUnit = "none",
        is_single_stream: bool = False,
    ) -> None:
        interactor: Optional[Interactor] = None if is_single_stream else CrossConcat()
        super().__init__(
            LegacyUnitListGenerator(input_channels, mid_channels_list, output_channels_list),
            interactor=interactor,
            calc_residual=calc_residual,
            keep_first_var_after=keep_first_var_after,
            exclusive_elements_of_unit=exclusive_elements_of_unit,
        )


class LegacyMeanAggregator(nn.Module):
    """Mean of two stream outputs in raw-logit form.

    Replaces :class:`~weavenet.layers.DualSoftmaxSqrt`. Reflects the legacy
    aggregation::

        m = (Z[:batch_size] + Z[batch_size:]) / 2  # MatcherWeaveNet.forward L386

    No softmax is applied — outputs are passed downstream as raw scores,
    matching the paper's training recipe where any probabilistic
    normalization is the criterion's responsibility.

    Returns the triplet ``(m, xab, xba_t)`` for interface compatibility
    with the rest of weavenet's StreamAggregator family.
    """

    def forward(
        self,
        xab: torch.Tensor,
        xba_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if xba_t is None:
            return xab, xab, xab
        m = (xab + xba_t) / 2
        return m, xab, xba_t


def legacy_residual_pattern(L: int) -> List[bool]:
    """Residual mask matching legacy ``MatcherWeaveNet.forward``.

    Legacy reference (``networks.py:372–378``)::

        for i, FWLayer in enumerate(self.encoders):
            Z = FWLayer(Z)
            if self.use_resnet and i % 2 == 0:
                if Z_keep is not None:
                    Z = Z + Z_keep
                Z_keep = Z

    At ``i==0`` legacy only initializes ``Z_keep`` (no add); at every
    later even index it both adds and re-saves. The PyPI semantics
    achieve the same with ``keep_first_var_after=0`` plus this mask:
    ``calc_residual[i] = (i > 0 and i % 2 == 0)``.
    """
    return [(i > 0 and i % 2 == 0) for i in range(L)]
