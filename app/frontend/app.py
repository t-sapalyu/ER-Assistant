import gradio as gr
import requests

BACKEND_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{BACKEND_URL}/query"


def answer_question(message, history):
    try:
        resp = requests.post(
            QUERY_ENDPOINT,
            json={"question": message, "history": history},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "(no answer returned)")
        sources = data.get("sources", [])
        if sources:
            answer += "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
        return answer
    except requests.RequestException as e:
        return f"❌ Request failed: {e}"


demo = gr.ChatInterface(
    fn=answer_question,
    title="ER-Assistant",
    chatbot=gr.Chatbot(
        height=600,
        show_label=False,
        container=False,
        placeholder="<div style='font-size:28px;font-weight:500;'>What can I help with?</div>",
    ),
    textbox=gr.Textbox(
        placeholder="Ask anything",
        show_label=False,
        submit_btn=True,
    ),
    submit_btn="Send",
    examples=[
        "What should I do for a minor burn?",
        "How do I recognize signs of a stroke?",
        "How do I perform CPR on an adult?",
    ],
)


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate").set(
            body_background_fill="#212121",
            body_background_fill_dark="#212121",
            background_fill_primary="#212121",
            background_fill_primary_dark="#212121",
            background_fill_secondary="#2f2f2f",
            background_fill_secondary_dark="#2f2f2f",
            block_background_fill="#212121",
            block_background_fill_dark="#212121",
            block_border_color="#212121",
            block_border_color_dark="#212121",
            border_color_primary="#212121",
            border_color_primary_dark="#212121",
        ),
        css="""
        /* Textbox wrapper — kill the blue tint, use ChatGPT-style dark gray */
        label.container.show_textbox_border {
            background-color: #2f2f2f !important;
            border: 1px solid #3f3f3f !important;
            border-radius: 24px !important;
        }
        label.container.show_textbox_border textarea {
            background-color: transparent !important;
            padding-right: 16px !important;
        }
        /* Space the send button from the textarea */
        label.container.show_textbox_border .input-container {
            gap: 12px !important;
            padding-right: 8px !important;
        }
        """,
    )
