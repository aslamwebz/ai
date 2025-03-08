## Installation

1. Install Ollama:
   - Visit [ollama.com](https://ollama.com/) to download and install
   - Pull a model: `ollama pull deepseek-r1:7b`

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   SERPER_API_KEY=your_serper_api_key_here
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Features

- Video search across YouTube, Vimeo, and Dailymotion
- AI-powered relevance ranking
- YouTube-style interface with hover-to-play
- Runs completely locally using Ollama