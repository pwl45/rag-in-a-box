import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import gradio as gr
from pathlib import Path
from rag_prs import answer_question

CUSTOM_PATH = "/gradio"

app = FastAPI()

@app.get("/demo")
def read_main():
    return {"message": "This is your main app"}

def greet(q,h):
    return f"Hello, {q}!"

# io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox", allow_flagging="never")
io = gr.ChatInterface(fn=answer_question)
app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)
# static_dir = Path("./output")
# print(static_dir)
app.mount("/static", StaticFiles(directory="output", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
