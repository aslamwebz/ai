import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import os
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, WebsiteSearchTool
import requests
import json

# Disable CrewAI telemetry to fix the SSL certificate error
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

load_dotenv()

search_tool = SerperDevTool()
website_search_tool = WebsiteSearchTool()


os.environ["serper_api_key"] = os.getenv("SERPER_API_KEY")

def set_ollama_llm(llm_model="ollama/deepseek-r1:7b"):
    ollama_llm = LLM(
        model=llm_model,
        base_url='http://localhost:11434',
        api_key="",
    )
    return ollama_llm

def check_ollama_server():
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ Ollama server is running")
            models_response = requests.get("http://localhost:11434/api/tags").json()
            global available_models
            available_models = [f"ollama/{model['name']}" for model in models_response['models']]
            return True
        else:
            st.sidebar.error("❌ Ollama server is not running")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ Error checking Ollama server: {e}")
        return False

def create_video_search_agent(ollama_llm):
    video_search_agent = Agent(
        role="Video Search Specialist",
        goal="Find relevant videos across major video platforms based on user queries",
        backstory="""You are an expert at finding videos across the web. You know all the 
        major video platforms like YouTube, Vimeo, Dailymotion, and others. You're skilled at 
        crafting search queries that return the most relevant results.""",
        verbose=True,
        llm=ollama_llm,
        tools=[website_search_tool]
    )
    
    return video_search_agent

def search_videos(query, ollama_llm):
    agent = create_video_search_agent(ollama_llm)
    
    video_search_task = Task(
        description=f"""
        Search for videos related to: "{query}"
        
        Follow these steps:
        1. First search YouTube for relevant videos.
        2. Then check other major platforms like Vimeo and Dailymotion if needed.
        3. For EACH video you find, you MUST include the direct video URL that can be embedded.
        4. Compile a list of the top 5 most relevant videos, including:
           - Video title
           - Platform (YouTube, Vimeo, etc.)
           - Creator/channel
           - Brief description (1-2 sentences)
           - The FULL video URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
        
        IMPORTANT: Return your response as valid JSON that can be parsed. Use this structure:
        {{
            "videos": [
                {{
                    "title": "Video Title",
                    "platform": "YouTube",
                    "creator": "Creator Name",
                    "description": "Brief description",
                    "url": "https://www.youtube.com/watch?v=VIDEO_ID"
                }},
                ...
            ]
        }}
        """,
        expected_output="A JSON object containing a list of 5 relevant videos with titles, platforms, creators, descriptions, and URLs.",
        agent=agent,
        tools=[website_search_tool]
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[video_search_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result

def extract_video_data(result_text):
    """Extract JSON data from the result text, or parse URLs if JSON extraction fails."""
    try:
        # First try to parse directly if it's already JSON
        try:
            data = json.loads(result_text)
            if "videos" in data:
                return data
        except:
            pass
            
        # Next try to find JSON in the text
        json_pattern = r'({[\s\S]*})'
        json_matches = re.findall(json_pattern, result_text)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if "videos" in data:
                    return data
            except:
                continue
                
        # Look for JSON with single quotes instead of double quotes
        for json_str in json_matches:
            try:
                # Replace single quotes with double quotes for JSON parsing
                fixed_json = json_str.replace("'", '"')
                data = json.loads(fixed_json)
                if "videos" in data:
                    return data
            except:
                continue
    except Exception as e:
        st.warning(f"Error parsing JSON: {str(e)}")
    
    # Fallback: Extract URLs and other info manually
    videos = []
    url_pattern = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|vimeo\.com/|dailymotion\.com/video/)[^\s)"\']+'
    urls = re.findall(url_pattern, result_text)
    
    # Get titles and descriptions around the URLs
    lines = result_text.split('\n')
    
    for url in urls[:5]:  # Limit to 5 videos
        # Find the closest title to this URL
        closest_title = ""
        closest_description = ""
        
        for i, line in enumerate(lines):
            if url in line:
                # Look back for title and forward for description
                start_idx = max(0, i-5)
                end_idx = min(len(lines), i+5)
                
                for j in range(start_idx, i):
                    if len(lines[j].strip()) > 0 and len(lines[j]) < 100:
                        closest_title = lines[j].strip()
                        break
                
                for j in range(i+1, end_idx):
                    if len(lines[j].strip()) > 10:
                        closest_description = lines[j].strip()
                        break
        
        videos.append({
            "title": closest_title or f"Video from {url.split('/')[2]}",
            "platform": "YouTube" if "youtube" in url or "youtu.be" in url else 
                      ("Vimeo" if "vimeo" in url else 
                       ("Dailymotion" if "dailymotion" in url else "Other")),
            "creator": "Unknown",
            "description": closest_description or "No description available",
            "url": url
        })
    
    if not videos:
        # As a last resort, just look for any URLs in the text
        simple_url_pattern = r'https?://[^\s)"\']+'
        all_urls = re.findall(simple_url_pattern, result_text)
        
        for url in all_urls[:5]:
            if any(domain in url for domain in ["youtube", "vimeo", "dailymotion", "video"]):
                videos.append({
                    "title": f"Video from {url.split('/')[2]}",
                    "platform": "Unknown",
                    "creator": "Unknown",
                    "description": "No description available",
                    "url": url
                })
    
    return {"videos": videos}

def display_video(video_url):
    """Display a video in Streamlit based on its URL."""
    try:
        if "youtube.com" in video_url or "youtu.be" in video_url:
            # Extract YouTube video ID
            if "youtube.com/watch?v=" in video_url:
                video_id = video_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in video_url:
                video_id = video_url.split("youtu.be/")[1].split("?")[0]
            else:
                video_id = video_url
                
            # For YouTube, Streamlit accepts both the full URL and just the ID
            st.video(video_id)
            
        elif "vimeo.com" in video_url:
            # Extract Vimeo video ID
            if "/video/" in video_url:
                video_id = video_url.split("/video/")[1].split("?")[0]
            else:
                video_id = video_url.split("vimeo.com/")[1].split("?")[0]
            
            # Vimeo needs the full embed URL
            st.video(f"https://player.vimeo.com/video/{video_id}")
            
        elif "dailymotion.com" in video_url:
            # Extract Dailymotion video ID
            if "/video/" in video_url:
                video_id = video_url.split("/video/")[1].split("?")[0]
            else:
                video_id = video_url.split("dailymotion.com/")[1].split("?")[0]
            
            # Dailymotion embed URL
            st.video(f"https://www.dailymotion.com/embed/video/{video_id}")
            
        else:
            # For other URLs, try direct embedding
            st.video(video_url)
            
        # Add a direct link as backup
        st.markdown(f"[Open video in new tab]({video_url})")
        
    except Exception as e:
        st.error(f"Could not play this video: {str(e)}")
        st.markdown(f"### [Click here to watch the video]({video_url})")
        # Display a thumbnail if possible
        st.image("https://img.icons8.com/color/96/000000/video.png", width=100)

# Add this function to get video thumbnails
def get_video_thumbnail(video_url):
    """Get thumbnail URL for a video based on its platform."""
    if "youtube.com" in video_url or "youtu.be" in video_url:
        # Extract YouTube video ID
        if "youtube.com/watch?v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = video_url
        
        return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    
    elif "vimeo.com" in video_url:
        # For Vimeo we use a placeholder
        return "https://i.vimeocdn.com/filter/overlay?src=https://i.vimeocdn.com/video/default.jpg"
    
    elif "dailymotion.com" in video_url:
        # Extract Dailymotion video ID
        if "/video/" in video_url:
            video_id = video_url.split("/video/")[1].split("?")[0]
        else:
            video_id = video_url.split("dailymotion.com/")[1].split("?")[0]
        
        return f"https://www.dailymotion.com/thumbnail/video/{video_id}"
    
    else:
        # Generic video icon
        return "https://img.icons8.com/color/96/000000/video.png"

# Add this function to get embed URLs for hover-play
def get_embed_url(video_url):
    """Get the embed URL for a video with autoplay parameters."""
    if "youtube.com" in video_url or "youtu.be" in video_url:
        # Extract YouTube video ID
        if "youtube.com/watch?v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = video_url
        
        return f"https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1"
    
    elif "vimeo.com" in video_url:
        # Extract Vimeo video ID
        if "/video/" in video_url:
            video_id = video_url.split("/video/")[1].split("?")[0]
        else:
            video_id = video_url.split("vimeo.com/")[1].split("?")[0]
        
        return f"https://player.vimeo.com/video/{video_id}?autoplay=1&muted=1"
    
    elif "dailymotion.com" in video_url:
        # Extract Dailymotion video ID
        if "/video/" in video_url:
            video_id = video_url.split("/video/")[1].split("?")[0]
        else:
            video_id = video_url.split("dailymotion.com/")[1].split("?")[0]
        
        return f"https://www.dailymotion.com/embed/video/{video_id}?autoplay=1&mute=1"
    
    else:
        return video_url

# Replace your current display section with this function
def render_youtube_style_results(videos):
    """Render videos in a YouTube-like grid with hover-to-play."""
    
    # Add CSS for YouTube-style layout
    st.markdown("""
    <style>
    .video-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .video-card {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        background: white;
        height: 100%;
    }
    
    .video-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .thumbnail-container {
        position: relative;
        width: 100%;
        padding-top: 56.25%; /* 16:9 aspect ratio */
        overflow: hidden;
    }
    
    .thumbnail {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .video-info {
        padding: 12px;
    }
    
    .video-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .video-creator {
        font-size: 14px;
        color: #606060;
        margin-bottom: 4px;
    }
    
    .video-platform {
        font-size: 12px;
        padding: 3px 6px;
        background: #f0f0f0;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .video-description {
        font-size: 13px;
        color: #606060;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: 8px;
    }
    
    .thumbnail-container:hover .thumbnail {
        opacity: 0;
    }
    
    .embed-container {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        transition: opacity 0.2s;
    }
    
    .thumbnail-container:hover .embed-container {
        opacity: 1;
    }
    
    iframe {
        width: 100%;
        height: 100%;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Start video grid
    video_grid_html = '<div class="video-grid">'
    
    # Generate HTML for each video card
    for video in videos:
        title = video.get("title", "Untitled Video")
        creator = video.get("creator", "Unknown Creator")
        platform = video.get("platform", "Unknown Platform")
        description = video.get("description", "No description available")
        url = video.get("url", "#")
        
        # Get thumbnail and embed URLs
        thumbnail_url = get_video_thumbnail(url)
        embed_url = get_embed_url(url)
        
        video_grid_html += f'''
        <a href="{url}" target="_blank" style="text-decoration: none; color: inherit;">
            <div class="video-card">
                <div class="thumbnail-container">
                    <img class="thumbnail" src="{thumbnail_url}" alt="{title}">
                    <div class="embed-container">
                        <iframe src="{embed_url}" allowfullscreen></iframe>
                    </div>
                </div>
                <div class="video-info">
                    <div class="video-title">{title}</div>
                    <div class="video-creator">{creator}</div>
                    <div class="video-platform">{platform}</div>
                    <div class="video-description">{description}</div>
                </div>
            </div>
        </a>
        '''
    
    # Close video grid
    video_grid_html += '</div>'
    
    # Render HTML
    st.components.v1.html(video_grid_html, height=(len(videos) // 3 + 1) * 360)

# Main Streamlit App
st.title("🎬 Video Search Engine using Ollama")

# Check if Ollama server is running
ollama_running = check_ollama_server()

if ollama_running:
    # Model selection
    
    selected_model = st.sidebar.selectbox("Select Ollama Model", available_models)

    ollama_llm = set_ollama_llm(selected_model)
    
    # Search interface
    st.markdown("### Search for videos across major platforms")
    with st.form("search_form"):
        query = st.text_input("Enter your search query:")
        search_button = st.form_submit_button("🔍 Search")
    
    if search_button and query:
        with st.spinner(f"Searching for videos related to '{query}'..."):
            try:
                crew_output = search_videos(query, ollama_llm)
                # Extract the string from CrewOutput object
                if hasattr(crew_output, 'raw_output'):
                    results_text = crew_output.raw_output
                elif hasattr(crew_output, 'result'):
                    results_text = crew_output.result
                else:
                    # If we can't find the expected attributes, convert to string
                    results_text = str(crew_output)
                
                st.markdown("### Search Results")
                
                # For debugging - show the raw text if needed
                with st.expander("Show raw search results"):
                    st.code(results_text)
                
                # Extract video data
                video_data = extract_video_data(results_text)
                
                # Display videos
                if "videos" in video_data and video_data["videos"]:
                    st.markdown(f"### Found {len(video_data['videos'])} videos for '{query}'")
                    render_youtube_style_results(video_data["videos"])
                else:
                    st.warning("No videos found. Try a different search term or platform.")
                    with st.expander("Show raw results"):
                        st.markdown(results_text)
            except Exception as e:
                st.error(f"An error occurred during the search: {str(e)}")
                st.error("Please try again with a different query or model.")
else:
    st.error("""
    Ollama server is not running. Please start the Ollama server first.
    
    Instructions:
    1. If you haven't installed Ollama yet, visit https://ollama.com/ to download and install it.
    2. Start Ollama on your machine.
    3. Make sure the model is pulled by running: `ollama pull deepseek-r1:7b`
    4. Refresh this page once Ollama is running.
    """)

