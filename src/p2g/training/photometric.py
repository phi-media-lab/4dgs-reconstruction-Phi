from __future__ import annotations

import torch
from torch import Tensor, nn

from p2g.errors import ContractError


class CameraColorCorrectors(nn.Module):
    """White-box per-camera affine RGB correction used only by the train loss.

    The physical transform is ``rgb @ (I + matrix[c]) + offset[c]``.  Keeping
    the identity separate makes the zero initialization exact and keeps every
    learned nuisance parameter inspectable.
    """

    matrices: nn.Parameter
    offsets: nn.Parameter
    identity: Tensor

    def __init__(self, camera_ids: tuple[str, ...]) -> None:
        super().__init__()
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ContractError("color-correction camera IDs must be non-empty and unique")
        self.camera_ids = camera_ids
        self._camera_to_index = {camera_id: index for index, camera_id in enumerate(camera_ids)}
        self.matrices = nn.Parameter(torch.zeros((len(camera_ids), 3, 3), dtype=torch.float32))
        self.offsets = nn.Parameter(torch.zeros((len(camera_ids), 3), dtype=torch.float32))
        self.register_buffer("identity", torch.eye(3, dtype=torch.float32), persistent=False)

    def camera_index(self, camera_id: str) -> int:
        try:
            return self._camera_to_index[camera_id]
        except KeyError as exc:
            raise ContractError(f"color correction has no camera {camera_id!r}") from exc

    def forward(self, camera_id: str, rgb: Tensor) -> Tensor:
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ContractError("color correction expects one HWC RGB image")
        index = self.camera_index(camera_id)
        transform = self.identity + self.matrices[index]
        return rgb @ transform + self.offsets[index]

    def regularization(self) -> Tensor:
        return self.matrices.square().sum() + self.offsets.square().sum()

    def finite(self) -> bool:
        return bool(torch.isfinite(self.matrices).all() and torch.isfinite(self.offsets).all())
