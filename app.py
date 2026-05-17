from __future__ import annotations

import gradio as gr

from inference import load_model, predict_image

learner = None
load_error = None
try:
    learner = load_model()
except Exception as exc:  # pragma: no cover - guarded for startup failures
    load_error = str(exc)


def predict(image):
    if image is None:
        return "", ""
    if load_error:
        return "Error loading model", load_error
    label, confidence = predict_image(learner, image)
    return label, f"{confidence:.2%}"


with gr.Blocks(
    css=(
        ".fixed-image img { width: 100%; height: 100%; object-fit: contain; }"
    )
) as demo:
    gr.Markdown("# Cats vs Dogs Image Classifier")
    gr.Markdown(
        "Upload a photo and the model will predict whether it is a cat or a dog, "
        "with a confidence score."
    )
    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Upload an image",
            width=320,
            height=320,
            elem_classes=["fixed-image"],
        )
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction")
            confidence_output = gr.Textbox(label="Confidence")
    gr.Button("Run prediction").click(
        predict,
        inputs=image_input,
        outputs=[prediction_output, confidence_output],
    )


if __name__ == "__main__":
    demo.launch()
