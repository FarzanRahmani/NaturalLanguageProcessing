from rag import RAG
import gradio as gr


def multimodal_rag_interface(question, image):
    return RAG(question, image)

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Multimodal RAG Demo")
    with gr.Row():
        question = gr.Textbox(label="Please Enter your Question")
        image_url = gr.Textbox(label="Please Enter your Image url (Optional)")
    with gr.Row():
        output = gr.Textbox(label="Answer")
    submit = gr.Button("Run RAG")
    submit.click(multimodal_rag_interface, inputs=[question, image_url], outputs=output)

demo.launch()