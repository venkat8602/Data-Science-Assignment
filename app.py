import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import base64
import os
from io import BytesIO
import time

# Set the API URL
API_URL = "http://localhost:8000"  # Update this when deploying to Hugging Face Spaces

def get_company_report(company_name):
    """Get the full report for a company from the API"""
    try:
        response = requests.get(f"{API_URL}/report/{company_name}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"API connection error: {e}")
        return None

def display_sentiment_chart(sentiment_data):
    """Display a chart of sentiment distribution"""
    df = pd.DataFrame({
        'Sentiment': list(sentiment_data.keys()),
        'Count': list(sentiment_data.values())
    })
    
    fig = px.pie(df, values='Count', names='Sentiment', 
                 color='Sentiment',
                 color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'},
                 title='Sentiment Distribution')
    
    return fig

def display_topics_chart(articles):
    """Display a chart of most common topics"""
    # Extract all topics and count frequency
    all_topics = []
    for article in articles:
        all_topics.extend(article.get('topics', []))
    
    # Count topic occurrences
    topic_counts = {}
    for topic in all_topics:
        if topic in topic_counts:
            topic_counts[topic] += 1
        else:
            topic_counts[topic] = 1
    
    # Create DataFrame and sort
    if topic_counts:
        df = pd.DataFrame({
            'Topic': list(topic_counts.keys()),
            'Frequency': list(topic_counts.values())
        })
        df = df.sort_values('Frequency', ascending=False).head(10)
        
        fig = px.bar(df, x='Topic', y='Frequency', 
                     title='Most Common Topics',
                     color='Frequency',
                     color_continuous_scale='Viridis')
        
        return fig
    return None

def get_audio_player_html(audio_path):
    """Generate HTML for an audio player"""
    audio_file = open(audio_path, 'rb')
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    html = f'''
    <audio controls>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    '''
    return html

def main():
    # Set page configuration
    st.set_page_config(
        page_title="News Analysis & TTS",
        page_icon="📰",
        layout="wide"
    )
    
    # Title and description
    st.title("📰 Company News Analysis & Text-to-Speech")
    st.markdown("""
    Enter a company name to analyze recent news articles, perform sentiment analysis, 
    and generate a Hindi audio summary.
    """)
    
    # Company input
    company_name = st.text_input("Enter a company name:", "Tesla")
    
    # Button to trigger analysis
    analyze_button = st.button("Analyze News")
    
    if analyze_button and company_name:
        with st.spinner(f"Analyzing news for {company_name}..."):
            # Start API server if not already running (for local development)
            # In production (Hugging Face Spaces), the API should be running separately
            try:
                # Check if the API is running
                requests.get(f"{API_URL}/")
            except:
                st.warning("API server not running. Starting API server...")
                import subprocess
                import threading
                
                def run_api():
                    subprocess.run(["python", "api.py"])
                
                thread = threading.Thread(target=run_api)
                thread.daemon = True
                thread.start()
                time.sleep(5)  # Wait for API to start
            
            # Get report from API
            report = get_company_report(company_name)
            
            if report:
                # Create tabs for different sections of the report
                tabs = st.tabs(["Summary", "Articles", "Sentiment Analysis", "Audio Summary"])
                
                with tabs[0]:  # Summary tab
                    st.header("News Analysis Summary")
                    st.markdown(f"### {report['final_sentiment_analysis']}")
                    
                    # Display sentiment distribution chart
                    sentiment_data = report['comparative_analysis']['sentiment_distribution']
                    fig = display_sentiment_chart(sentiment_data)
                    st.plotly_chart(fig)
                    
                    # Display common topics
                    st.subheader("Common Topics")
                    st.write(", ".join(report['comparative_analysis']['common_topics']))
                
                with tabs[1]:  # Articles tab
                    st.header("Analyzed Articles")
                    
                    for i, article in enumerate(report['articles']):
                        with st.expander(f"{i+1}. {article['title']}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
                                st.markdown(f"**Date:** {article.get('date', 'Unknown')}")
                                st.markdown("### Summary")
                                st.write(article['summary'])
                                st.markdown(f"[Read Full Article]({article['url']})")
                            
                            with col2:
                                sentiment = article['sentiment_analysis']['sentiment']
                                sentiment_color = {
                                    'Positive': 'green',
                                    'Neutral': 'gray',
                                    'Negative': 'red'
                                }.get(sentiment, 'gray')
                                
                                st.markdown(f"### Sentiment")
                                st.markdown(f"<h4 style='color:{sentiment_color}'>{sentiment}</h4>", unsafe_allow_html=True)
                                st.markdown("### Topics")
                                st.write(", ".join(article.get('topics', [])))
                
                with tabs[2]:  # Sentiment Analysis tab
                    st.header("Comparative Sentiment Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display sentiment distribution
                        st.subheader("Sentiment Distribution")
                        fig = display_sentiment_chart(sentiment_data)
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Display topics chart
                        st.subheader("Top Topics")
                        topics_fig = display_topics_chart(report['articles'])
                        if topics_fig:
                            st.plotly_chart(topics_fig)
                    
                    # Display comparisons
                    st.subheader("Key Insights")
                    for comparison in report['comparative_analysis'].get('comparisons', []):
                        st.markdown(f"**Comparison:** {comparison.get('comparison', '')}")
                        st.markdown(f"**Impact:** {comparison.get('impact', '')}")
                        st.markdown("---")
                
                with tabs[3]:  # Audio Summary tab
                    st.header("Hindi Audio Summary")
                    st.markdown("Listen to a summarized version of the report in Hindi:")
                    
                    if report.get('audio_summary'):
                        # Display audio player
                        audio_html = get_audio_player_html(report['audio_summary'])
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
                        # Display Hindi text
                        st.subheader("Hindi Text Summary")
                        st.write(report.get('hindi_text_summary', ''))
                    else:
                        st.error("Audio summary not available.")

if __name__ == "__main__":
    main() 