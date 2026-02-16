"""
divNet: Diverging 3D CNN for Alzheimer's Disease Classification
Based on: "3D CNN Design for the Classification of Alzheimer's Disease
           Using Brain MRI and PET" (Khagi & Kwon, 2020)

Architecture: Diverging kernel sizes (3 -> 5 -> 7 -> 9)
Input: [B, 1, 192, 192, 192] preprocessed MRI volumes
Output: [B, 3] logits for CN/MCI/AD classification
"""

import torch
import torch.nn as nn


class DivNet(nn.Module):
    """
    Diverging 3D CNN (divNet / L4) adapted for 192x192x192 input.

    Spatial flow:
        Block 1: Conv3d(k=3, s=1, p=1) -> BN -> ReLU -> MaxPool(2) : 192 -> 96
        Block 2: Conv3d(k=5, s=2, p=2) -> BN -> ReLU -> MaxPool(2) : 96  -> 24
        Block 3: Conv3d(k=7, s=2, p=3) -> BN -> ReLU -> MaxPool(2) : 24  -> 6
        Block 4: Conv3d(k=9, s=1, p=4) -> BN -> ReLU -> MaxPool(2) : 6   -> 3

    Classifier:
        Flatten(3*3*3*64=1728) -> FC(512) -> FC(100) -> FC(3)
    """

    def __init__(self, num_filters=64, num_classes=3, dropout1=0.5, dropout2=0.3):
        super(DivNet, self).__init__()

        # Block 1: k=3, diverging kernel start
        self.block1 = nn.Sequential(
            nn.Conv3d(1, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 2: k=5
        self.block2 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 3: k=7
        self.block3 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 4: k=9
        self.block4 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Classifier
        flatten_size = num_filters * 3 * 3 * 3  # 64 * 27 = 1728

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(100, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = DivNet()
    dummy = torch.randn(1, 1, 192, 192, 192)
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
