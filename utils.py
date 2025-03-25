import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newspaper import Article
from transformers import pipeline
from gtts import gTTS
import os
from collections import Counter
from typing import List, Dict, Any
import numpy as np
import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

def search_news_articles(company_name: str, num_articles: int = 10) -> List[Dict[str, str]]:
    """
    Search for news articles related to a company using SerpAPI.
    
    Args:
        company_name: Name of the company to search for
        num_articles: Number of articles to return
        
    Returns:
        List of dictionaries containing article URLs
    """
    from serpapi import GoogleSearch
    
    try:
        # Get API key from environment variables
        api_key = os.getenv("SERPAPI_API_KEY")
        
        params = {
            "q": f"{company_name} news",
            "num": num_articles + 5,  # Request extra to filter JS-heavy pages later
            "tbm": "nws",
            "api_key": api_key
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        articles = []
        if "news_results" in results:
            for article in results["news_results"][:num_articles + 5]:
                articles.append({
                    "title": article.get("title", ""),
                    "url": article.get("link", ""),
                    "source": article.get("source", ""),
                    "date": article.get("date", "")
                })
                
        return articles[:num_articles]
    except Exception as e:
        print(f"Error searching for news articles: {e}")
        # If API key not available or other issues, return sample data for testing
        return [
            {
                "title": f"{company_name} Reports Strong Q3 Earnings",
                "url": "https://example.com/news1",
                "source": "Finance News",
                "date": "1 day ago"
            },
            {
                "title": f"{company_name} Announces New Product Line",
                "url": "https://example.com/news2",
                "source": "Tech Today",
                "date": "2 days ago"
            }
        ]

def extract_article_content(url: str) -> Dict[str, Any]:
    """
    Extract content from a news article URL using newspaper3k.
    
    Args:
        url: URL of the news article
        
    Returns:
        Dictionary containing article title, text, and metadata
    """
    try:
        # Skip extraction for example.com URLs (our fallback data)
        if "example.com" in url:
            return {
                "title": "Sample Article Title",
                "text": "This is sample article text content used for testing when actual news articles cannot be fetched.",
                "summary": "Sample article for testing purposes.",
                "published_date": None,
                "authors": ["Sample Author"],
                "url": url
            }
            
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        
        return {
            "title": article.title,
            "text": article.text,
            "summary": article.summary,
            "published_date": article.publish_date,
            "authors": article.authors,
            "url": url
        }
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {
            "title": "Could not extract",
            "text": "Content extraction failed. This could be due to JavaScript-rendered content or access restrictions.",
            "summary": "Content extraction failed.",
            "published_date": None,
            "authors": [],
            "url": url
        }

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of the given text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary containing sentiment labels and scores
    """
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        
        # Determine sentiment label
        if sentiment_scores['compound'] >= 0.05:
            sentiment = "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "sentiment": sentiment,
            "compound_score": sentiment_scores['compound'],
            "positive_score": sentiment_scores['pos'],
            "negative_score": sentiment_scores['neg'],
            "neutral_score": sentiment_scores['neu']
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {
            "sentiment": "Neutral",
            "compound_score": 0,
            "positive_score": 0,
            "negative_score": 0,
            "neutral_score": 1
        }

def extract_topics(text: str, num_topics: int = 5) -> List[str]:
    """
    Extract key topics from the article using NLTK.
    
    Args:
        text: Text to analyze
        num_topics: Number of topics to extract
        
    Returns:
        List of topic strings
    """
    try:
        # Tokenize and normalize text
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 3]
        
        # Find frequent terms
        fdist = FreqDist(tokens)
        most_common = [word for word, _ in fdist.most_common(num_topics*2)]
        
        # Use transformers for keyword extraction if available
        try:
            from keybert import KeyBERT
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_topics)
            return [keyword for keyword, _ in keywords]
        except Exception as ke:
            print(f"KeyBERT extraction failed, falling back to NLTK: {ke}")
            # Fallback to n-gram extraction if KeyBERT not available
            from nltk.util import ngrams
            bigrams = list(ngrams(tokens, 2))
            bigram_freq = FreqDist(bigrams)
            common_bigrams = [' '.join(bg) for bg, _ in bigram_freq.most_common(num_topics)]
            
            # Combine single words and bigrams
            topics = most_common[:num_topics//2] + common_bigrams[:num_topics//2]
            return topics[:num_topics]
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return ["General", "News", "Business", "Technology", "Finance"]  # Default topics

def perform_comparative_analysis(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform comparative sentiment analysis across multiple articles.
    
    Args:
        articles: List of article dictionaries with sentiment data
        
    Returns:
        Dictionary containing comparative analysis results
    """
    try:
        # Handle empty article list
        if not articles:
            return {
                "sentiment_distribution": {"Neutral": 0, "Positive": 0, "Negative": 0},
                "average_compound_score": 0.0,
                "common_topics": [],
                "unique_topics_by_article": {},
                "comparisons": [],
                "final_sentiment_analysis": "No articles were available for analysis."
            }
            
        # Count sentiment distribution
        sentiment_counts = Counter([article['sentiment_analysis']['sentiment'] for article in articles])
        
        # Calculate average sentiment scores
        avg_compound = np.mean([article['sentiment_analysis']['compound_score'] for article in articles])
        
        # Find common and unique topics
        all_topics = [topic for article in articles for topic in article.get('topics', [])]
        topic_counter = Counter(all_topics)
        common_topics = [topic for topic, count in topic_counter.items() if count > 1]
        
        # Prepare comparison insights
        comparisons = []
        
        # Compare articles with different sentiments
        positive_articles = [a for a in articles if a['sentiment_analysis']['sentiment'] == 'Positive']
        negative_articles = [a for a in articles if a['sentiment_analysis']['sentiment'] == 'Negative']
        neutral_articles = [a for a in articles if a['sentiment_analysis']['sentiment'] == 'Neutral']
        
        if positive_articles and negative_articles:
            comparison = {
                "comparison": "Some articles present positive news while others highlight negative aspects.",
                "impact": "This mixed coverage suggests complex factors affecting the company's reputation."
            }
            comparisons.append(comparison)
        
        # Generate topic-based comparisons
        unique_topics_by_article = {}
        for i, article in enumerate(articles):
            topics = set(article.get('topics', []))
            unique_topics = topics - set(common_topics)
            if unique_topics:
                unique_topics_by_article[i] = list(unique_topics)
        
        # Get company name safely
        company_name = articles[0].get('company', 'The company') if articles else "The company"
        
        # Generate final sentiment summary
        if avg_compound >= 0.25:
            final_sentiment = f"{company_name}'s latest news coverage is mostly positive. Potential stock growth expected."
        elif avg_compound <= -0.25:
            final_sentiment = f"{company_name}'s latest news coverage is mostly negative. Caution is advised."
        else:
            final_sentiment = f"{company_name}'s latest news coverage is mixed or neutral. Market response may be subdued."
            
        return {
            "sentiment_distribution": dict(sentiment_counts),
            "average_compound_score": float(avg_compound),
            "common_topics": common_topics[:5] if common_topics else [],
            "unique_topics_by_article": unique_topics_by_article,
            "comparisons": comparisons,
            "final_sentiment_analysis": final_sentiment
        }
    except Exception as e:
        print(f"Error performing comparative analysis: {e}")
        return {
            "sentiment_distribution": {"Neutral": 0, "Positive": 0, "Negative": 0},
            "average_compound_score": 0,
            "common_topics": [],
            "unique_topics_by_article": {},
            "comparisons": [],
            "final_sentiment_analysis": "Could not analyze sentiment trends."
        }

def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi.
    
    Args:
        text: English text to translate
        
    Returns:
        Translated Hindi text
    """
    try:
        # First try to make sure sentencepiece is properly installed
        try:
            import sentencepiece
            print("SentencePiece is installed")
        except ImportError:
            print("SentencePiece not found, translation will not work")
            return f"[Translation failed: SentencePiece not available] {text}"
            
        # Using Hugging Face's translation pipeline with explicit model loading
        from transformers import MarianMTModel, MarianTokenizer
        
        try:
            model_name = "Helsinki-NLP/opus-mt-en-hi"
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            # Break text into chunks if it's too long
            max_length = 128  # Shorter chunks to avoid tokenizer issues
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            translated_chunks = []
            for chunk in chunks:
                # Translate chunk directly with model and tokenizer
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_chunks.append(translated_text)
                
            return " ".join(translated_chunks)
        except Exception as model_error:
            print(f"Error with translation model: {model_error}")
            
            # Try the pipeline approach as a fallback
            try:
                translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
                
                # Break text into shorter chunks
                max_length = 100
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                
                translated_chunks = []
                for chunk in chunks:
                    translation = translator(chunk)[0]['translation_text']
                    translated_chunks.append(translation)
                    
                return " ".join(translated_chunks)
            except Exception as pipeline_error:
                print(f"Pipeline translation also failed: {pipeline_error}")
                return f"[Translation failed] {text}"
                
    except Exception as e:
        print(f"Error translating to Hindi: {e}")
        # Provide a simple fallback Hindi translation for the most common phrases
        if "News report for" in text:
            return "समाचार रिपोर्ट: " + text
        # Fallback to original text with note
        return f"[अनुवाद विफल] {text}"

def generate_hindi_tts(text: str, output_file: str = "output.mp3") -> str:
    """
    Convert text to Hindi speech and save to a file.
    
    Args:
        text: Text to convert to speech
        output_file: Path to save the output audio file
        
    Returns:
        Path to the generated audio file
    """
    try:
        # If text is empty or translation failed, use a default message
        if not text or text.startswith("[Translation failed]"):
            text = "नमस्ते, अनुवाद उपलब्ध नहीं है। कृपया बाद में फिर से प्रयास करें।"  # Hello, translation is not available. Please try again later.
            
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(output_file)
        return output_file
    except Exception as e:
        print(f"Error generating Hindi TTS: {e}")
        # Create a simple audio file with a default message if TTS fails
        try:
            fallback_text = "नमस्ते, ऑडियो उपलब्ध नहीं है।"  # Hello, audio is not available.
            fallback_tts = gTTS(text=fallback_text, lang='hi', slow=False)
            fallback_tts.save(output_file)
            return output_file
        except:
            return ""

def generate_report(company_name: str) -> Dict[str, Any]:
    """
    Generate a complete news analysis report for a company.
    
    Args:
        company_name: Name of the company
        
    Returns:
        Dictionary containing the full report data
    """
    try:
        print(f"Generating report for {company_name}...")
        
        # Search for news articles
        article_urls = search_news_articles(company_name)
        print(f"Found {len(article_urls)} news articles")
        
        # Process each article
        articles = []
        for article_info in article_urls:
            url = article_info['url']
            print(f"Processing article: {url}")
            
            # Extract content
            content = extract_article_content(url)
            
            # Skip if no content was extracted
            if not content['text']:
                print(f"Skipping article with no content: {url}")
                continue
                
            # Analyze sentiment
            sentiment = analyze_sentiment(content['text'])
            
            # Extract topics
            topics = extract_topics(content['text'])
            
            # Create article entry
            article_data = {
                "company": company_name,
                "title": content['title'],
                "summary": content['summary'],
                "sentiment_analysis": sentiment,
                "topics": topics,
                "url": url,
                "source": article_info.get('source', ''),
                "date": article_info.get('date', '')
            }
            
            articles.append(article_data)
            print(f"Added article: {content['title']} (Sentiment: {sentiment['sentiment']})")
            
            # Stop after collecting 10 valid articles
            if len(articles) >= 10:
                break
                
        print(f"Processed {len(articles)} valid articles")
        
        # Perform comparative analysis
        comparison = perform_comparative_analysis(articles)
        
        # Generate report summary for TTS
        summary_text = f"News report for {company_name}. "
        
        if articles:
            summary_text += f"Based on {len(articles)} news articles analyzed, the overall sentiment is {comparison['final_sentiment_analysis']}. "
            summary_text += f"There are {comparison['sentiment_distribution'].get('Positive', 0)} positive articles, "
            summary_text += f"{comparison['sentiment_distribution'].get('Negative', 0)} negative articles, and "
            summary_text += f"{comparison['sentiment_distribution'].get('Neutral', 0)} neutral articles. "
            
            if comparison['common_topics']:
                summary_text += f"Common topics in the news include: {', '.join(comparison['common_topics'][:3])}. "
        else:
            summary_text += "No articles were found for analysis."
        
        print("Translating summary to Hindi...")
        # Translate summary to Hindi
        hindi_summary = translate_to_hindi(summary_text)
        
        print("Generating TTS file...")
        # Generate TTS file
        audio_file = generate_hindi_tts(hindi_summary)
        
        # Create final report
        report = {
            "company": company_name,
            "articles": articles,
            "comparative_analysis": comparison,
            "final_sentiment_analysis": comparison['final_sentiment_analysis'],
            "audio_summary": audio_file,
            "hindi_text_summary": hindi_summary
        }
        
        print("Report generation complete")
        return report
    except Exception as e:
        print(f"Error generating report: {e}")
        # Return a minimal report with error information
        return {
            "company": company_name,
            "articles": [],
            "comparative_analysis": {
                "sentiment_distribution": {"Neutral": 0, "Positive": 0, "Negative": 0},
                "average_compound_score": 0,
                "common_topics": [],
                "unique_topics_by_article": {},
                "comparisons": [],
                "final_sentiment_analysis": f"Could not generate analysis due to an error: {str(e)}"
            },
            "final_sentiment_analysis": f"Could not generate analysis due to an error: {str(e)}",
            "audio_summary": "",
            "hindi_text_summary": ""
        } 