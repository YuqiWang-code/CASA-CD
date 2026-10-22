"""Learn spatial change assignments for aligned context compression."""
from torch import nn


class ChangeAssignmentHead(nn.Sequential):
    def __init__(self, channels, hidden=32):
        super().__init__(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
