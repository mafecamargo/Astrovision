import torch

from pathlib import Path

from PIL import Image

from torchvision import transforms

from utils.models import get_model
from utils.explainability import generate_gradcam


class Predictor:

    def __init__(
        self,
        checkpoint_path,
        class_names,
        device=None
    ):

        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.model_name = (
            self.checkpoint_path.stem
            .replace("_best", "")
        )

        self.class_names = class_names

        self.model = get_model(
            self.model_name,
            len(class_names)
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(
            state_dict
        )

        #
        # IMPORTANT:
        # GradCAM requires gradients.
        #
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.to(
            self.device
        )

        self.model.eval()

        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    def preprocess(self, image):

        if isinstance(image, str):

            image = Image.open(
                image
            )

        image = image.convert(
            "RGB"
        )

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        return image, tensor

    @torch.no_grad()
    def predict(self, image):

        image, tensor = self.preprocess(
            image
        )

        tensor = tensor.to(
            self.device
        )

        outputs = self.model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        prediction_idx = prediction.item()

        return {

            "class":
                self.class_names[
                    prediction_idx
                ],

            "confidence":
                confidence.item(),

            "probabilities": {

                self.class_names[i]:
                    probabilities[0][i].item()

                for i in range(
                    len(self.class_names)
                )
            }
        }

    def gradcam(self, image):

        image, tensor = self.preprocess(
            image
        )

        tensor = tensor.to(
            self.device
        )

        try:

            gradcam_image = generate_gradcam(

                model=self.model,

                model_name=self.model_name,

                input_tensor=tensor,

                original_image=image
            )

            return gradcam_image

        except Exception as e:

            print(
                f"GradCAM failed for "
                f"{self.model_name}: {e}"
            )

            return None
