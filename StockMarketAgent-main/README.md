# AI Investment Strategist

AI-powered Stock Market Analysis Agent is a tool designed to instantly analyze multiple stocks using the AGNO Agentic AI library and the Google Gemini Flash language model.

Streamlit app that compares selected stocks, gathers company context from Yahoo Finance, and generates an AI-assisted investment report using Gemini.

# Components and Functionality:

Market Analyst Agent:
This agent fetches six months of stock history, calculates the overall percentage change in stock prices daily, and ranks stocks based on their relative performance.

Company Researcher Agent:
This agent gathers company-specific data, including the name, sector, market capitalization, a company summary, and the latest five news articles.

Stock Strategist Agent:
This agent integrates the data from both the Market Analyst and Company Researcher agents to formulate investment strategies, determining if a stock should be bought, sold, or held.

Team Lead Agent:
This agent manages the specialized agents, compiles their analysis into a summarized report for each company, and generates a final ranked list of stocks.

## Features

- Compare recent stock performance (6 months)
- Generate company-level summaries with latest news
- Produce AI investment recommendations and ranked report
- Visualize stock price movement with Plotly

## Requirements

- Python 3.10+
- A Google AI Studio API key

## Setup

1. Clone the repo and move into the project:
   - `cd AI_Project`
2. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate` (macOS/Linux)
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Configure environment variables:
   - `cp .env.example .env`
   - Add your real `GOOGLE_API_KEY` in `.env`

## Run

- `streamlit run investment.py`

Open the local URL shown by Streamlit in your browser.

## Notes

- Do not commit `.env` or any API keys.
- If a Gemini model is unavailable for your key, update `DEFAULT_GEMINI_MODEL` in `investment.py`.
