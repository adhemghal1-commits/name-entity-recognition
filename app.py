import sys
import gradio as gr
import torch
from transformers import pipeline

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "ner-distilbert-final"
COLORS = {"PER": "#ffd54f", "ORG": "#81d4fa", "LOC": "#a5d6a7", "MISC": "#f48fb1"}

ner_pipe = pipeline(
    "token-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1,
)


def ner_predict(text):
    if not text or not text.strip():
        return {"text": "", "entities": []}
    ents = ner_pipe(text)
    return {
        "text": text,
        "entities": [
            {"entity": e["entity_group"], "start": int(e["start"]), "end": int(e["end"])}
            for e in ents
        ],
    }


with gr.Blocks(title="NER Highlighter") as app:
    gr.Markdown("# Named Entity Recognition\nType any text. Entities are highlighted live: **PER**, **ORG**, **LOC**, **MISC**.")
    inp = gr.Textbox(lines=5, label="Input text", placeholder="Type or paste text here...")
    out = gr.HighlightedText(label="Detected entities", color_map=COLORS, show_legend=True)
    inp.change(ner_predict, inputs=inp, outputs=out)
    gr.Examples(
        examples=[
            "Barack Obama met Angela Merkel in Berlin to discuss NATO and the European Union.",
            "Mohamed Salah scored twice for Liverpool at Anfield on Saturday.",
            "Google and Microsoft opened new offices in Cairo and Dubai.",
        ],
        inputs=inp,
    )

if __name__ == "__main__":
    app.launch()
