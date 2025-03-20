import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Initialize environment variables
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

def fetch_news(company):
    """Fetch news articles using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    articles = response.json().get('articles', [])[:10]  # Limit to 10 articles
    return articles

def scrape_article(url):
    """Scrape article title and content using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.text.strip() if soup.title else "No Title"
        paragraphs = soup.find_all('p')
        content = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
        return title, content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    """Analyze sentiment of the text."""
    truncated_text = text[:512]  # Truncate to model's max length
    result = sentiment_analyzer(truncated_text)[0]
    return result['label']

def generate_report(company):
    """Process company name to generate report."""
    articles = fetch_news(company)
    if not articles:
        return {"error": "No articles found"}, None
    
    report = {"Company": company, "Articles": [], "Comparative Sentiment": {"Positive": 0, "Negative": 0, "Neutral": 0}}
    
    for article in articles:
        url = article.get('url')
        if not url:
            continue
        title, content = scrape_article(url)
        if not content:
            continue
        
        # Analyze sentiment
        sentiment = analyze_sentiment(content)
        sentiment_key = "Positive" if sentiment == "POSITIVE" else "Negative" if sentiment == "NEGATIVE" else "Neutral"
        report["Comparative Sentiment"][sentiment_key] += 1
        
        # Extract summary and topics (simplified)
        summary = content[:100] + '...' if len(content) > 100 else content
        words = [word.lower() for word in content.split() if len(word) > 3]
        topics = list(set(words))[:3]  # Simple keyword extraction
        
        report["Articles"].append({
            "Title": title,
            "Summary": summary,
            "Sentiment": sentiment_key,
            "Topics": topics
        })
    
    # Generate Hindi TTS
    tts_text = f"{company} के लिए समाचार सारांश। सकारात्मक लेख: {report['Comparative Sentiment']['Positive']}, नकारात्मक: {report['Comparative Sentiment']['Negative']}, तटस्थ: {report['Comparative Sentiment']['Neutral']}."
    tts = gTTS(tts_text, lang='hi')
    tts_file = "summary_hi.mp3"
    tts.save(tts_file)
    
    return report, tts_file

# Gradio Interface
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