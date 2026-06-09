import streamlit as st
import pandas as pd

from pathlib import Path
from PIL import Image

from utils.predict import Predictor
from utils.dataset import build_dataset


DATASET_DIR = "data"

CHECKPOINT_DIR = Path(
    "checkpoints"
)


#
# Automatically infer classes
#
_, CLASS_NAMES = build_dataset(
    DATASET_DIR
)


#
# Discover checkpoints
#
AVAILABLE_MODELS = {}

for checkpoint in CHECKPOINT_DIR.glob(
    "*.pth"
):

    model_name = (
        checkpoint.stem
        .replace("_best", "")
    )

    AVAILABLE_MODELS[
        model_name
    ] = checkpoint


if len(AVAILABLE_MODELS) == 0:

    st.error(
        "No checkpoints found."
    )

    st.stop()


st.set_page_config(
    page_title="Disaster Image Classifier",
    page_icon="🌎",
    layout="wide"
)


st.title(
    "🌎 Disaster Image Classification"
)

st.write(
    """
    Upload an image and compare
    predictions across models.
    """
)


selected_model = st.sidebar.selectbox(

    "Select Model",

    sorted(
        AVAILABLE_MODELS.keys()
    )
)


@st.cache_resource
def load_predictor(
    model_name
):

    return Predictor(

        checkpoint_path=
            AVAILABLE_MODELS[
                model_name
            ],

        class_names=
            CLASS_NAMES
    )


predictor = load_predictor(
    selected_model
)


st.sidebar.markdown("---")

st.sidebar.write(
    f"Model: **{selected_model}**"
)

st.sidebar.write(
    f"Classes: **{len(CLASS_NAMES)}**"
)


uploaded_file = st.file_uploader(

    "Upload Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    result = predictor.predict(
        image
    )

    gradcam = predictor.gradcam(
        image
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.subheader(
            "Image"
        )

        st.image(
            image,
            width='content'
        )

    with col2:

        st.subheader(
            "Prediction"
        )

        st.success(
            result["class"]
        )

        st.metric(
            "Confidence",
            f"{result['confidence']:.2%}"
        )

    with col3:

        st.subheader(
            "Grad-CAM"
        )

        if gradcam is not None:

            st.image(
                gradcam,
                width='content'
            )

        else:

            st.warning(
                "GradCAM unavailable "
                "for this model."
            )

    st.markdown("---")

    st.subheader(
        "Class Probabilities"
    )

    df = pd.DataFrame({

        "Class":
            list(
                result[
                    "probabilities"
                ].keys()
            ),

        "Probability":
            list(
                result[
                    "probabilities"
                ].values()
            )
    })

    st.bar_chart(
        df.set_index(
            "Class"
        )
    )

    st.subheader(
        "Detailed Scores"
    )

    st.dataframe(

        df.sort_values(
            "Probability",
            ascending=False
        ),

        width='content'
    )
