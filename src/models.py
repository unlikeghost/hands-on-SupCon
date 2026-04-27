import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniConvNet_(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
    ) -> None:
        super(MiniConvNet_, self).__init__()
        self.feature_extractor = nn.Sequential(
            # Bloque 1
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class MiniConvNetClassifier(MiniConvNet_):
    def __init__(
        self,
        input_channels: int = 3,
        output_classes: int = 10
    ) -> None:
        super(MiniConvNetClassifier, self).__init__(
            input_channels=input_channels
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=254),
            nn.ReLU(),
            nn.Linear(in_features=254, out_features=output_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.get_features(x)
        x = x.view(x.size(0), -1) # Aplana (N, C, H, W) → (N, features)
        x = self.mlp(x)
        return x


class MiniConvNetEncoder(MiniConvNet_):
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 256
    ) -> None:
        super(MiniConvNetEncoder, self).__init__(
            input_channels=input_channels
        )

        self.projector = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 8, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return F.normalize(x, p=2, dim=1)


class MiniConvNetLinearProbe(nn.Module):
    def __init__(
        self,
        encoder: MiniConvNetEncoder,
        embedding_dim: int = 256,
        output_classes: int = 10,
    ) -> None:
        super(MiniConvNetLinearProbe, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(
            in_features=embedding_dim,
            out_features=output_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not next(self.encoder.parameters()).requires_grad:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)

        x = self.classifier(x)
        return x