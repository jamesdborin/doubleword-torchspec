"""Backend-neutral tensor format constants for transfer engines."""

import torch

# Canonical wire/storage dtype for speculative-training hidden states.
HIDDEN_STATES_STORAGE_DTYPE = torch.bfloat16
