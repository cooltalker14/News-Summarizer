import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
from dotenv import load_dotenv
import torch
from collections import defaultdict
import spacy

load_dotenv()
nlp = spacy.load("en_core_web_sm")  # For topic extraction

# Initialize environment variables
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

def fetch_news(company):
    """Fetch news articles using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    articles = response.json().get('articles', [])
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

# Load models
sentiment_analyzer = pipeline("sentiment-analysis", 
                            model="distilbert-base-uncased-finetuned-sst-2-english", 
                            framework="pt")

summarizer = pipeline("summarization", 
                    model="facebook/bart-large-cnn",
                    framework="pt")

def analyze_sentiment(text):
    """Analyze sentiment of the text."""
    truncated_text = text[:512]
    result = sentiment_analyzer(truncated_text)[0]
    return result['label']

def generate_report(company):
    """Process company name to generate report with 10 unique articles."""
    articles = fetch_news(company)
    if not articles:
        return {"error": "No articles found"}, None

    report = {
        "Company": company,
        "Articles": [],
        "Comparative Analysis": {
            "Sentiment Distribution": {"Positive": 0, "Negative": 0, "Neutral": 0},
            "Coverage Differences": [],
            "Topic Overlap": {}
        }
    }
    
    unique_articles = []
    seen_urls = set()
    all_topics = []
    
    # Collect up to 10 unique articles
    for article in articles:
        url = article.get('url')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
        if len(unique_articles) >= 10:
            break

    # Process articles
    for article in unique_articles:
        url = article.get('url')
        title, content = scrape_article(url)
        if not content:
            continue

        # Generate summary
        try:
            summary = summarizer(content, max_length=130, min_length=30)[0]['summary_text']
        except:
            summary = content[:100] + '...' if len(content) > 100 else content

        # Analyze sentiment
        sentiment = analyze_sentiment(content)
        sentiment_key = "Positive" if sentiment == "POSITIVE" else "Negative" if sentiment == "NEGATIVE" else "Neutral"
        report["Comparative Analysis"]["Sentiment Distribution"][sentiment_key] += 1

        # Extract topics with spaCy
        doc = nlp(content)
        topics = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "LAW")][:3]
        all_topics.extend(topics)

        report["Articles"].append({
            "Title": title,
            "Summary": summary,
            "Sentiment": sentiment_key,
            "Topics": topics
        })

    # Comparative Analysis
    topic_counts = defaultdict(int)
    for topic in all_topics:
        topic_counts[topic] += 1

    common_topics = [topic for topic, count in topic_counts.items() if count > 1]
    unique_topics = list(set(all_topics))
    
    # Add coverage differences
    if len(report["Articles"]) >= 2:
        report["Comparative Analysis"]["Coverage Differences"].append({
            "Comparison": f"{report['Articles'][0]['Title']} vs {report['Articles'][1]['Title']}",
            "Impact": "Different aspects of the company covered"
        })

    report["Comparative Analysis"]["Topic Overlap"] = {
        "Common Topics": common_topics,
        "Unique Topics": unique_topics
    }

    # Generate Hindi TTS with gTTS
    tts_text = f"{company} के लिए समाचार सारांश। सकारात्मक लेख: {report['Comparative Analysis']['Sentiment Distribution']['Positive']}, नकारात्मक: {report['Comparative Analysis']['Sentiment Distribution']['Negative']}, तटस्थ: {report['Comparative Analysis']['Sentiment Distribution']['Neutral']}."
    tts = gTTS(tts_text, lang='hi')
    tts_file = "summary_hi.mp3"
    tts.save(tts_file)

    return report, tts_file