import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


GRADCAM_LAYERS = {

    "resnet50":
        lambda m: [m.layer4[-1]],

    "vgg16":
        lambda m: [m.features[28]],

    "densenet121":
        lambda m: [m.features.denseblock4],

    "efficientnet_b0":
        lambda m: [m.features[-2]],
}


def get_target_layers(model, model_name):

    model_name = model_name.lower()

    if model_name not in GRADCAM_LAYERS:

        raise ValueError(
            f"GradCAM not configured for {model_name}"
        )

    return GRADCAM_LAYERS[model_name](model)


def generate_gradcam(
    model,
    model_name,
    input_tensor,
    original_image
):

    target_layers = get_target_layers(
        model,
        model_name
    )

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor
        )

        grayscale_cam = grayscale_cam[0]

    rgb_image = (
        np.array(
            original_image.resize(
                (224, 224)
            )
        ).astype(np.float32)
        / 255.0
    )

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization
