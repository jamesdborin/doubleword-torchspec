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

"""Transfer layer: remote tensor storage and data transfer."""

from torchspec.transfer.base import (
    TensorBundle,
    TensorSpec,
    TransferBackend,
    TransferCapacityError,
    TransferClosedError,
    TransferError,
    TransferNotFoundError,
    TransferProtocolError,
    TransferRef,
    TransferRole,
    TransferSetupError,
    TransferState,
    TransferStateError,
    TransferTimeoutError,
    TransferUnavailableError,
)
from torchspec.transfer.factory import create_transfer_backend, create_transfer_backend_from_env
from torchspec.transfer.runtime import TransferRuntime

__all__ = [
    "TensorBundle",
    "TensorSpec",
    "TransferBackend",
    "TransferCapacityError",
    "TransferClosedError",
    "TransferError",
    "TransferNotFoundError",
    "TransferProtocolError",
    "TransferRef",
    "TransferRole",
    "TransferRuntime",
    "TransferSetupError",
    "TransferState",
    "TransferStateError",
    "TransferTimeoutError",
    "TransferUnavailableError",
    "create_transfer_backend",
    "create_transfer_backend_from_env",
]
