# Equity Research News Tool

An AI-powered web app that fetches recent financial news and summarizes it for equity research. Enter a company, ticker or topic, and the app retrieves real articles from NewsAPI, then uses LangChain with Google Gemini to write a structured summary with citations.

## Features

- Search by company name, ticker or financial topic
- Real news from NewsAPI (last 14 days, sorted by relevance)
- AI summary built **only** from the retrieved article text, with `[n]` citations
- Article list with headline, source, publication date and link to the original
- Clear error messages: empty input, no results, invalid keys, rate limits, network and Gemini errors
- Automatic fallback to a second Gemini model when the first is busy
- Results cached for 15 minutes to save API quota

## How it works

```
User query -> NewsAPI -> clean articles -> LangChain prompt -> Gemini -> Streamlit
```

## Tech stack

Python, LangChain, Google Gemini (`langchain-google-genai`), NewsAPI, Streamlit

## Project structure

```
equity-research-news-tool/
├── app.py                  # Streamlit interface
├── langchain_config.py     # NewsAPI client, Gemini + LangChain, prompt, summary workflow
├── utils/
│   └── news_processor.py   # Cleans articles and builds the digest sent to the AI
├── tests/
│   └── test_error_handling.py   # Offline unit tests (no API keys needed)
├── validate_queries.py     # Live check of 4 example queries (uses real API quota)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup (Windows)

Requires Python 3.10 or higher.

1. Clone the repository and open the folder:

   ```powershell
   git clone https://github.com/YOUR-USERNAME/equity-research-news-tool.git
   cd equity-research-news-tool
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Get your API keys:
   - **Gemini:** https://aistudio.google.com/apikey
   - **NewsAPI:** https://newsapi.org (click Get API Key)

5. Create your `.env` file from the template and fill in your keys:

   ```powershell
   Copy-Item .env.example .env
   ```

   Open `.env` and replace the placeholder values. Never commit this file.

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key |
| `NEWSAPI_KEY` | NewsAPI key |
| `GEMINI_MODEL` | Main Gemini model name |
| `GEMINI_FALLBACK_MODEL` | Used if the main model is busy |

Gemini model names change over time. If you get a "model not found" error, check Google AI Studio for a current model name and update `.env`.

## Usage

```powershell
streamlit run app.py
```

Open http://localhost:8501, type a query such as `Tesla stock news`, and click **Get News**.

Example searches: `Tesla stock news`, `Apple financial news`, `NVIDIA AI market news`, `Reliance Industries stock news`.

Tips for better results: use specific words like `earnings`, `shares` or `Jio`, and put exact names in quotes, for example `"Reliance Industries" shares`.

## Testing

Offline unit tests (no internet, no API quota used):

```powershell
python -m unittest tests.test_error_handling -v
```

Live validation of the four example queries (uses about 4 NewsAPI requests and 4 Gemini calls):

```powershell
python validate_queries.py
```

Manual checks in the app: empty search, a nonsense query, clicking article links, and comparing three summary claims with the original articles.

## Limitations

- NewsAPI's free plan returns short description snippets rather than full articles, so summaries reflect those snippets only.
- The free plan delays articles by about 24 hours and allows about 100 requests per day.
- AI summaries can contain mistakes. Always verify important figures in the original articles.
- This project is for research and learning. It is **not** financial advice.

## Deployment notes

- Never commit `.env`. On a hosting platform, set the four variables in its secrets or environment settings (on Streamlit Community Cloud, add them under App settings -> Secrets as `KEY = "value"` lines).
- NewsAPI's free plan is intended for development and local testing. A public deployment needs a paid NewsAPI plan or a different news source. Check https://newsapi.org/pricing for current terms.
- The free Gemini tier has rate limits and its terms may allow Google to use submitted prompts. Review Google's current terms before any public launch.

## Possible improvements

User login, saved searches, summary export, date filters, and sentiment tracking over time.