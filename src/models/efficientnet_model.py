import torch
import torch.nn as nn
import timm

class EfficientNetAudio(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # modify first layer to accept single channel input
        original_layer = self.model.conv_stem
        self.model.conv_stem = nn.Conv2d(
            1, original_layer.out_channels,
            kernel_size = original_layer.kernel_size,
            stride = original_layer.stride,
            padding = original_layer.padding,
            bias = False
        )

        # modify classifier for multi-label classification
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)