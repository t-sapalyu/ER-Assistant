import gradio as gr
import requests

BACKEND_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{BACKEND_URL}/query"


def answer_question(message, history):
    """Send the question + chat history to the backend."""
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


custom_css = """
/* Full-height dark background */
.gradio-container {
    background-color: #212121 !important;
    max-width: 100% !important;
    padding: 0 !important;
}
footer { display: none !important; }

/* Top-left brand */
#brand {
    position: fixed;
    top: 16px;
    left: 24px;
    font-size: 18px;
    font-weight: 600;
    color: #ececec;
    z-index: 100;
}

/* Center the chat column */
.main-col {
    max-width: 780px;
    margin: 0 auto !important;
}

/* Kill ALL backgrounds and borders on the chatbot and its wrappers */
.chatbot,
.chatbot > div,
.chatbot .bubble-wrap,
.chatbot .message-wrap,
.chatbot .placeholder-content {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Rounded input bar like ChatGPT */
.input-row textarea {
    background-color: #2f2f2f !important;
    border: none !important;
    border-radius: 24px !important;
    color: #ececec !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
}
.input-row textarea::placeholder { color: #8e8e8e !important; }

/* Wrap the whole input row in the rounded pill */
.input-row {
    background-color: #2f2f2f !important;
    border-radius: 24px !important;
    border: 1px solid #3f3f3f !important;
}

/* Send button — circular, white, on the right */
.input-row button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 50% !important;
    min-width: 36px !important;
    width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    margin-right: 6px !important;
}
"""

with gr.Blocks(css=custom_css, title="ER-Assistant") as demo:
    gr.HTML('<div id="brand">ER-Assistant</div>')

    with gr.Column(elem_classes="main-col"):
        gr.ChatInterface(
            fn=answer_question,
            chatbot=gr.Chatbot(
                elem_classes="chatbot",
                height=600,
                show_label=False,
                placeholder="<div style='text-align:center;font-size:28px;color:#ececec;margin-top:240px;'>What can I help with?</div>",
            ),
            textbox=gr.Textbox(
                placeholder="Ask anything",
                show_label=False,
                elem_classes="input-row",
                container=False,
            ),
            examples=[
                "What should I do for a minor burn?",
                "How do I recognize signs of a stroke?",
                "How do I perform CPR on an adult?",
            ],
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate"))
