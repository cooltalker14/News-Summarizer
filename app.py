import gradio as gr
import requests  # <-- Added

def generate_report(company):
    """Call FastAPI endpoint to generate report."""
    try:
        response = requests.get(f"http://127.0.0.1:8000/report?company={company}")
        if response.status_code == 200:
            data = response.json()
            return data["report"], data["audio"]
        return {"error": "API call failed"}, None
    except Exception as e:
        return {"error": str(e)}, None

iface = gr.Interface(
    fn=generate_report,
    inputs=gr.Textbox(label="कंपनी का नाम दर्ज करें (Enter Company Name)"),
    outputs=[
        gr.JSON(label="रिपोर्ट (Report)"),
        gr.Audio(label="हिंदी ऑडियो सारांश (Hindi Audio Summary)", type="filepath")
    ],
    title="समाचार सारांश और भावना विश्लेषण (News Summarization & Sentiment Analysis)",
    description="कंपनी का नाम दर्ज करें और स्वचालित रिपोर्ट व ऑडियो सारांश प्राप्त करें।"
)

if __name__ == "__main__":
    iface.launch()