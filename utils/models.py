import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes, pretrained=False):

    weights = (
        models.ResNet50_Weights.IMAGENET1K_V2
        if pretrained else None
    )

    model = models.resnet50(weights=weights)

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model


def build_vgg16(num_classes, pretrained=False):

    weights = (
        models.VGG16_Weights.IMAGENET1K_V1
        if pretrained else None
    )

    model = models.vgg16(weights=weights)

    if pretrained:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )

    return model


def build_efficientnet_b0(num_classes, pretrained=False):

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        if pretrained else None
    )

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def build_densenet121(num_classes, pretrained=False):

    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
        if pretrained else None
    )

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes
    )

    return model


def get_model(model_name, num_classes, pretrained=False):

    model_name = model_name.lower()

    if model_name == "resnet50":
        return build_resnet50(num_classes, pretrained=pretrained)

    elif model_name == "vgg16":
        return build_vgg16(num_classes, pretrained=pretrained)

    elif model_name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes, pretrained=pretrained)

    elif model_name == "densenet121":
        return build_densenet121(num_classes, pretrained=pretrained)

    raise ValueError(
        f"Unsupported model: {model_name}"
    )
