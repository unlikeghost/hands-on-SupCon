import torch
import torch.nn as nn

class LeNet5_(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
    ) -> None:
        super(LeNet5_, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)  # Extrae características espaciales (features)
        return x


class LeNet5Classifier(LeNet5_):
    def __init__(
        self,
        input_channels: int = 3,
        output_classes: int = 10
    ) -> None:
        super(LeNet5Classifier, self).__init__(
            input_channels=input_channels
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_features(x)
        x = x.view(x.size(0), -1) # Aplana (N, C, H, W) → (N, features)
        x = self.mlp(x)
        return x


class LeNet5Encoder(LeNet5_):
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 128
    ) -> None:
        super(LeNet5Encoder, self).__init__(
            input_channels=input_channels
        )
        self.projector = nn.Sequential(
            nn.Linear(in_features=16 * 5 * 5, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x