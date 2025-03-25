from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import time
from utils import generate_report, search_news_articles, extract_article_content, analyze_sentiment, extract_topics
import traceback

app = FastAPI(title="News Analysis API", 
              description="API for news extraction, sentiment analysis, and TTS generation",
              version="1.0.0")

class ArticleResponse(BaseModel):
    title: str
    summary: str
    sentiment: str
    topics: List[str]
    url: str
    source: Optional[str] = None
    date: Optional[str] = None

class ComparisonResponse(BaseModel):
    sentiment_distribution: Dict[str, int]
    average_compound_score: float
    common_topics: List[str]
    comparisons: List[Dict[str, str]]
    final_sentiment_analysis: str

class ReportResponse(BaseModel):
    company: str
    articles: List[Dict[str, Any]]
    comparative_analysis: Dict[str, Any]
    final_sentiment_analysis: str
    audio_file_path: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the News Analysis API"}

@app.get("/search/{company_name}", response_model=List[Dict[str, str]])
def search_news(company_name: str, limit: int = Query(10, ge=1, le=20)):
    """
    Search for news articles related to a company
    """
    results = search_news_articles(company_name, limit)
    if not results:
        raise HTTPException(status_code=404, detail="No news articles found")
    return results

@app.get("/article")
def get_article_analysis(url: str = Query(...)):
    """
    Extract and analyze a single article from a URL
    """
    # Extract content
    content = extract_article_content(url)
    
    # Skip if no content was extracted
    if not content['text']:
        raise HTTPException(status_code=404, detail="Could not extract article content")
    
    # Analyze sentiment
    sentiment = analyze_sentiment(content['text'])
    
    # Extract topics
    topics = extract_topics(content['text'])
    
    return {
        "title": content['title'],
        "summary": content['summary'],
        "sentiment": sentiment['sentiment'],
        "sentiment_details": sentiment,
        "topics": topics,
        "url": url
    }

@app.get("/report/{company_name}")
async def get_company_report(company_name: str, request: Request):
    """
    Generate a complete news analysis report for a company
    """
    try:
        # Print request details for debugging
        print(f"Received request for company: {company_name}")
        
        # Generate the report
        start_time = time.time()
        report = generate_report(company_name)
        end_time = time.time()
        
        print(f"Report generation completed in {end_time - start_time:.2f} seconds")
        
        # Check if report was generated
        if not report:
            print("Report generation failed - empty report returned")
            raise HTTPException(status_code=500, detail="Failed to generate report")
            
        return report
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        print(traceback.format_exc())  # Print full stack trace
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/audio/{company_name}")
def get_audio_summary(company_name: str):
    """
    Generate and return the audio summary file for a company
    """
    report = generate_report(company_name)
    if not report or not report.get('audio_summary'):
        raise HTTPException(status_code=404, detail="Could not generate audio summary")
    
    # In a real implementation, you'd return the audio file here
    return {"audio_file_path": report['audio_summary']}

if __name__ == "__main__":
    # Run the API server when the script is executed directly
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 