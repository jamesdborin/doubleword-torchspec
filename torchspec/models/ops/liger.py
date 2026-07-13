# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Optional Liger-Kernel integration helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_liger_fused_linear_ce_cls() -> type | None:
    """Return Liger's fused linear CE module class when importable."""
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except (ImportError, RuntimeError):
        return None
    return LigerFusedLinearCrossEntropyLoss


def is_liger_available() -> bool:
    return get_liger_fused_linear_ce_cls() is not None


def make_liger_fused_linear_ce(**kwargs: Any):
    cls = get_liger_fused_linear_ce_cls()
    if cls is None:
        raise ImportError(
            "liger_kernel is not importable. Install torchspec with the liger-kernel "
            "dependency to use DFlash fused linear cross entropy."
        )
    return cls(**kwargs)
