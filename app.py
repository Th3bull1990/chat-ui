import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

def predict(input_img):
    predictions = pipeline(input_img)
    return input_img, {p["label"]: p["score"] for p in predictions} 

gradio_app = gr.Interface(
    predict,
    inputs=gr.Image(label="Select hot dog candidate", sources=['upload', 'webcam'], type="pil"),
    outputs=[gr.Image(label="Processed Image"), gr.Label(label="Result", num_top_classes=2)],
    title="Hot Dog? Or Not?",
)

if __name__ == "__main__":
    gradio_app.launch()
