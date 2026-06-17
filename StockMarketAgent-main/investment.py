import os
import yfinance as yf
import streamlit as st
from agno.agent import Agent
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DEFAULT_SYMBOLS = "AAPL, TSLA, GOOG"

# Function to fetch and compare stock data
def compare_stocks(symbols):
    data = {}
    for symbol in symbols:
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")  # Fetch last 6 months' data
            
            if hist.empty:
                print(f"No data found for {symbol}, skipping it.")
                continue  # Skip this ticker if no data found
            
            # Calculate overall % change
            data[symbol] = hist['Close'].pct_change().sum()
        
        except Exception as e:
            print(f"Could not retrieve data for {symbol}. Reason: {str(e)}")
            continue  # Skip this ticker if an error occurs

    return data

# Function to fetch key live financial metrics
def fetch_live_metrics(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Safely extract metrics
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        prev_close = info.get("regularMarketPreviousClose") or 0.0
        
        # Calculate daily change percent
        change_pct = 0.0
        if prev_close > 0:
            change_pct = ((current_price - prev_close) / prev_close) * 100.0
            
        pe_ratio = info.get("trailingPE", "N/A")
        div_yield = info.get("dividendYield", 0.0)
        if isinstance(div_yield, (int, float)):
            div_yield = f"{div_yield * 100:.2f}%" if div_yield > 0 else "0.0%"
        else:
            div_yield = "N/A"
            
        market_cap = info.get("marketCap", 0)
        if market_cap > 1e12:
            market_cap_str = f"${market_cap / 1e12:.2f}T"
        elif market_cap > 1e9:
            market_cap_str = f"${market_cap / 1e9:.2f}B"
        elif market_cap > 1e6:
            market_cap_str = f"${market_cap / 1e6:.2f}M"
        else:
            market_cap_str = f"${market_cap:,}" if market_cap else "N/A"
            
        high_52w = info.get("fiftyTwoWeekHigh", 0.0)
        low_52w = info.get("fiftyTwoWeekLow", 0.0)
        
        return {
            "name": info.get("longName", symbol),
            "price": current_price,
            "change_pct": change_pct,
            "pe_ratio": pe_ratio,
            "dividend_yield": div_yield,
            "market_cap": market_cap_str,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "currency": info.get("currency", "USD")
        }
    except Exception as e:
        print(f"Error fetching live metrics for {symbol}: {e}")
        return None

def get_company_info(symbol):
    stock = yf.Ticker(symbol)
    return {
        "name": stock.info.get("longName", "N/A"),
        "sector": stock.info.get("sector", "N/A"),
        "market_cap": stock.info.get("marketCap", "N/A"),
        "summary": stock.info.get("longBusinessSummary", "N/A"),
    }

def get_company_news(symbol):
    stock = yf.Ticker(symbol)
    news_items = stock.news or []
    news = news_items[:5]  # Get latest 5 news articles
    return news

# Helper to get the model instance dynamically
def get_model_instance(provider, model_id, api_key):
    if provider == "Google Gemini":
        from agno.models.google import Gemini
        os.environ["GOOGLE_API_KEY"] = api_key
        return Gemini(id=model_id, api_key=api_key)
    elif provider == "Groq":
        from agno.models.groq import Groq
        os.environ["GROQ_API_KEY"] = api_key
        return Groq(id=model_id, api_key=api_key)
    elif provider == "OpenAI":
        from agno.models.openai import OpenAIChat
        os.environ["OPENAI_API_KEY"] = api_key
        return OpenAIChat(id=model_id, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Helper to define the agents dynamically with the chosen model
def get_investment_agents(model_instance):
    market_analyst = Agent(
        model=model_instance,
        description="Analyzes and compares stock performance over time.",
        instructions=[
            "Retrieve and compare stock performance from Yahoo Finance.",
            "Calculate percentage change over a 6-month period.",
            "Rank stocks based on their relative performance.",
            "Do not output any programming code (such as Python or code blocks). Write in plain markdown text."
        ],
        markdown=True
    )

    company_researcher = Agent(
        model=model_instance,
        description="Fetches company profiles, financials, and latest news.",
        instructions=[
            "Retrieve company information from Yahoo Finance.",
            "Summarize latest company news relevant to investors.",
            "Provide sector, market cap, and business overview.",
            "Do not output any programming code (such as Python or code blocks). Write in plain markdown text."
        ],
        markdown=True
    )

    stock_strategist = Agent(
        model=model_instance,
        description="Provides investment insights and recommends top stocks.",
        instructions=[
            "Analyze stock performance trends and company fundamentals.",
            "Evaluate risk-reward potential and industry trends.",
            "Provide top stock recommendations for investors.",
            "Do not output any programming code (such as Python or code blocks). Write in plain markdown text."
        ],
        markdown=True
    )

    team_lead = Agent(
        model=model_instance,
        description="Aggregates stock analysis, company research, and investment strategy.",
        instructions=[
            "Compile stock performance, company analysis, and recommendations.",
            "Ensure all insights are structured in an investor-friendly report.",
            "Rank the top stocks based on combined analysis.",
            "Clearly specify which stock is the single best stock to buy.",
            "Provide the absolute best time/entry strategy to buy that stock (e.g. key indicators to watch, buy on dips, etc.).",
            "Do not output any programming code (such as Python or code blocks). Write purely in formatted markdown."
        ],
        markdown=True
    )
    
    return market_analyst, company_researcher, stock_strategist, team_lead

# News Sentiment Analysis agent function
def analyze_news_sentiment(symbol, news_items, model_instance):
    if not news_items:
        return 0, "Neutral", "No recent news found."
    
    prompt = (
        f"Analyze the market sentiment for {symbol} based on these news titles:\n"
    )
    for idx, item in enumerate(news_items):
        prompt += f"{idx+1}. {item.get('title', 'No Title')}\n"
    
    prompt += (
        "\nProvide a response in the following format:\n"
        "SCORE: <a number from -100 to 100>\n"
        "SENTIMENT: <Positive, Neutral, or Negative>\n"
        "SUMMARY: <one-sentence summary of the sentiment reasoning>\n"
        "Make sure to follow the format exactly. Do not output anything else."
    )
    
    sentiment_agent = Agent(
        model=model_instance,
        description=f"Performs financial news sentiment analysis for {symbol}.",
        instructions=[
            "Determine an overall sentiment score from -100 (extremely bearish) to +100 (extremely bullish).",
            "Classify the sentiment as Positive, Neutral, or Negative.",
            "Write a concise, single-sentence summary explaining why."
        ]
    )
    
    try:
        response = sentiment_agent.run(prompt)
        content = response.content
        
        score = 0
        sentiment = "Neutral"
        summary = "No reasoning provided."
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("SCORE:"):
                score = int(float(line.replace("SCORE:", "").strip()))
            elif line.startswith("SENTIMENT:"):
                sentiment = line.replace("SENTIMENT:", "").strip()
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                
        score = max(-100, min(100, score))
        return score, sentiment, summary
    except Exception as e:
        print(f"Error analyzing news sentiment for {symbol}: {e}")
        return 0, "Neutral", "Failed to analyze news sentiment."

# Optimized function to get the final investment report in a single pass
def generate_optimized_investment_report(symbols, model_instance, budget, risk_tolerance):
    # Initialize agents dynamically with the selected model
    market_analyst, company_researcher, stock_strategist, team_lead = get_investment_agents(model_instance)
    
    # 1. Compare stocks performance
    performance_data = compare_stocks(symbols)
    if not performance_data:
        return "No valid stock data found for the given symbols.", {}, {}
    
    # Format performance metrics into a clean plain text summary
    performance_text = "\n".join([f"- {sym}: {pct:.2%}" for sym, pct in performance_data.items()])
    
    # 2. Get market analysis (1 call)
    market_analysis_res = market_analyst.run(f"Compare these stock performances:\n{performance_text}")
    market_analysis = market_analysis_res.content
    
    # 3. Get company analysis and news sentiment for each symbol
    company_analyses_dict = {}
    company_analyses_list = []
    sentiment_data = {}
    
    for symbol in symbols:
        info = get_company_info(symbol)
        news = get_company_news(symbol)
        
        # Call news sentiment agent (1 call per symbol)
        score, sentiment, summary = analyze_news_sentiment(symbol, news, model_instance)
        sentiment_data[symbol] = {"score": score, "sentiment": sentiment, "summary": summary}
        
        # Format news articles into a clean plain text block
        news_text = ""
        for idx, item in enumerate(news):
            news_text += f"{idx+1}. {item.get('title', 'No Title')} (Publisher: {item.get('publisher', 'N/A')})\n"
            
        # Call company researcher agent (1 call per symbol)
        response = company_researcher.run(
            f"Provide an analysis for {info['name']} in the {info['sector']} sector.\n"
            f"Market Cap: {info['market_cap']}\n"
            f"Summary: {info['summary']}\n"
            f"News Sentiment: {sentiment} ({score:+} Score) - Reason: {summary}\n"
            f"Latest News Articles:\n{news_text}"
        )
        company_analyses_dict[symbol] = response.content
        company_analyses_list.append(response.content)
        
    # Format company analysis into a clean text block
    company_analyses_formatted = ""
    for sym, analysis in company_analyses_dict.items():
        company_analyses_formatted += f"### {sym} Research and News Analysis:\n{analysis}\n\n"
        
    # 4. Get stock recommendations and portfolio allocations (1 call)
    recommendations_res = stock_strategist.run(
        f"Based on the market analysis:\n{market_analysis}\n\n"
        f"and company research:\n{company_analyses_formatted}\n\n"
        f"How would you allocate a budget of ${budget} across these stocks: {symbols} for a {risk_tolerance} risk appetite?\n"
        f"Provide detailed allocation reasoning, and at the very end write a single line summarizing the allocation in this exact format:\n"
        f"ALLOCATION: Symbol1:Pct1%, Symbol2:Pct2%...\n"
        f"where the percentages add up to exactly 100%."
    )
    stock_recommendations = recommendations_res.content
    
    # Extract the ALLOCATION line if possible, and remove it from display text
    allocations = {sym: round(100 / len(symbols), 2) for sym in symbols}
    clean_stock_recommendations_lines = []
    
    for line in stock_recommendations.split("\n"):
        if "ALLOCATION:" in line:
            try:
                alloc_str = line.replace("ALLOCATION:", "").strip()
                parts = alloc_str.split(",")
                parsed_alloc = {}
                for part in parts:
                    sym, pct = part.split(":")
                    sym = sym.strip().upper()
                    pct = float(pct.replace("%", "").strip())
                    parsed_alloc[sym] = pct
                if abs(sum(parsed_alloc.values()) - 100.0) < 5.0:
                    allocations = parsed_alloc
            except Exception:
                pass
        else:
            clean_stock_recommendations_lines.append(line)
            
    clean_stock_recommendations = "\n".join(clean_stock_recommendations_lines).strip()
    
    # Format company analyses list as a clean readable text block
    company_analyses_list_text = "\n\n".join(company_analyses_list)
    
    # 5. Get final report (1 call)
    final_report_res = team_lead.run(
        f"Market Analysis:\n{market_analysis}\n\n"
        f"Company Analyses & Sentiments:\n{company_analyses_list_text}\n\n"
        f"Stock Recommendations & Recommended Allocation:\n{clean_stock_recommendations}\n\n"
        f"Provide the full analysis of each stock with Fundamentals and market news.\n"
        f"At the very beginning of the report, add a clear, prominent section titled:\n"
        f"'## 🎯 AI Buy Decision & Entry Strategy'\n"
        f"In this section, explicitly state which single stock we should buy, why, and the best time or entry trigger to buy that stock.\n"
        f"Following this section, present the rest of the ranked analysis report."
    )
    return final_report_res.content, allocations, sentiment_data


# Streamlit page configuration
st.set_page_config(page_title="AI Investment Strategist", page_icon="📈", layout="wide")

# Title and header
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">📈 AI Investment Strategist</h1>
    <h3 style="text-align: center; color: #6c757d;">Generate personalized investment reports with the latest market insights.</h3>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("""
    <h2 style="color: #343a40;">Configuration</h2>
    <p style="color: #6c757d;">Select your preferred LLM provider, model, and configure stock symbols for analysis.</p>
""", unsafe_allow_html=True)

# Provider selection
provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Google Gemini", "Groq", "OpenAI"]
)

# Model & API Key options based on Provider Selection
if provider == "Google Gemini":
    model_options = [
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    default_model = "gemini-2.0-flash-001"
    env_key = os.getenv("GOOGLE_API_KEY", "")
    key_label = "Enter Google Gemini API Key"
elif provider == "Groq":
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    default_model = "llama-3.3-70b-versatile"
    env_key = os.getenv("GROQ_API_KEY", "")
    key_label = "Enter Groq API Key"
else:  # OpenAI
    model_options = [
        "gpt-4o-mini",
        "gpt-4o",
    ]
    default_model = "gpt-4o-mini"
    env_key = os.getenv("OPENAI_API_KEY", "")
    key_label = "Enter OpenAI API Key"

model_id = st.sidebar.selectbox(
    "Select Model",
    model_options,
    index=model_options.index(default_model) if default_model in model_options else 0
)

api_key = st.sidebar.text_input(key_label, value=env_key, type="password")

# Stock Selection Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Stock Selection")
input_symbols = st.sidebar.text_input("Enter Stock Symbols (separated by commas)", DEFAULT_SYMBOLS)
stocks_symbols = [symbol.strip().upper() for symbol in input_symbols.split(",") if symbol.strip()]

# Chart Configuration Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Chart Configuration")
chart_period = st.sidebar.selectbox("Chart Timeframe", ["1mo", "3mo", "6mo", "1y", "5y"], index=2)
show_sma20 = st.sidebar.checkbox("Show 20-Day SMA", value=False)
show_sma50 = st.sidebar.checkbox("Show 50-Day SMA", value=False)

# Portfolio Simulator Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 💼 Portfolio Simulator")
budget = st.sidebar.number_input("Total Budget to Invest ($)", min_value=100, value=10000, step=500)
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)

# Generate Investment Report button
if st.sidebar.button("Generate Investment Report"):
    if not stocks_symbols:
        st.sidebar.warning("Please enter at least one stock symbol.")
    elif not api_key:
        st.sidebar.warning(f"Please enter your {provider} API Key.")
    else:
        try:
            with st.spinner(f"Generating report using {provider} ({model_id})..."):
                # Get the dynamic model instance
                model_instance = get_model_instance(provider, model_id, api_key)

                # 1. Display Live KPI Dashboard
                st.markdown("## 💎 Live Financial Metrics Dashboard")
                metric_cols = st.columns(len(stocks_symbols))
                
                live_metrics = {}
                for idx, symbol in enumerate(stocks_symbols):
                    metrics = fetch_live_metrics(symbol)
                    if metrics:
                        live_metrics[symbol] = metrics
                        with metric_cols[idx]:
                            color_prefix = "+" if metrics['change_pct'] >= 0 else ""
                            st.metric(
                                label=metrics['name'],
                                value=f"${metrics['price']:.2f} {metrics['currency']}",
                                delta=f"{color_prefix}{metrics['change_pct']:.2f}%"
                            )
                            st.caption(f"PE: {metrics['pe_ratio']} | Yield: {metrics['dividend_yield']} | Cap: {metrics['market_cap']}")

                # 2. Run the dynamic, optimized agent flow
                report, allocations, sentiment_data = generate_optimized_investment_report(
                    stocks_symbols, model_instance, budget, risk_tolerance
                )

                # 3. Render News Sentiment Indicators
                st.markdown("## 📰 News Sentiment Indicators")
                sentiment_cols = st.columns(len(stocks_symbols))
                for idx, symbol in enumerate(stocks_symbols):
                    with sentiment_cols[idx]:
                        if symbol in sentiment_data:
                            data = sentiment_data[symbol]
                            score = data['score']
                            sentiment = data['sentiment']
                            summary = data['summary']
                            
                            # Card coloring based on sentiment
                            card_color = "#28a745" if score > 15 else ("#dc3545" if score < -15 else "#ffc107")
                            st.markdown(
                                f"""
                                <div style="background-color: #1e2530; padding: 15px; border-radius: 8px; border-top: 5px solid {card_color}; margin-bottom: 15px; min-height: 120px;">
                                    <h4 style="margin: 0; color: #fff;">{symbol} Sentiment</h4>
                                    <span style="font-weight: bold; color: {card_color}; font-size: 1.1em;">{sentiment} ({score:+} Score)</span><br/>
                                    <p style="font-size: 0.85em; color: #b0b3b8; margin: 5px 0 0 0;">{summary}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # 4. Render Portfolio Allocation Visualization
                st.markdown("## 💼 Portfolio Simulator Allocation")
                sim_col1, sim_col2 = st.columns([1, 1])
                
                with sim_col1:
                    fig_pie = px.pie(
                        names=list(allocations.keys()),
                        values=list(allocations.values()),
                        title=f"Optimal Allocation for ${budget:,} ({risk_tolerance} Risk)",
                        color_discrete_sequence=px.colors.sequential.Viridis
                    )
                    fig_pie.update_layout(template="plotly_dark", title_x=0.2)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with sim_col2:
                    st.markdown("### Recommended Asset Breakdown")
                    alloc_data = []
                    for sym, pct in allocations.items():
                        amount = (pct / 100.0) * budget
                        name = live_metrics.get(sym, {}).get("name", sym)
                        alloc_data.append({
                            "Symbol": sym,
                            "Company": name,
                            "Percentage": f"{pct:.1f}%",
                            "Suggested Investment": f"${amount:,.2f}"
                        })
                    st.table(alloc_data)

                # 5. Display the Detailed Markdown Investor Report
                st.markdown("---")
                st.subheader("📋 Detailed Investor Report")
                st.markdown(report)

                # 6. Technical Charts Section
                st.markdown(f"## 📊 Stock Performance & Technical Indicators ({chart_period})")
                stock_data = yf.download(stocks_symbols, period=chart_period)
                
                fig = go.Figure()
                close_data = stock_data["Close"]
                
                for symbol in stocks_symbols:
                    if len(stocks_symbols) == 1:
                        prices = close_data
                        name = symbol
                    else:
                        prices = close_data[symbol]
                        name = symbol
                        
                    fig.add_trace(go.Scatter(x=prices.index, y=prices, mode="lines", name=f"{name} Close"))
                    
                    if show_sma20:
                        sma20 = prices.rolling(window=20).mean()
                        fig.add_trace(go.Scatter(x=sma20.index, y=sma20, mode="lines", line=dict(dash="dash"), name=f"{name} SMA-20"))
                        
                    if show_sma50:
                        sma50 = prices.rolling(window=50).mean()
                        fig.add_trace(go.Scatter(x=sma50.index, y=sma50, mode="lines", line=dict(dash="dot"), name=f"{name} SMA-50"))

                fig.update_layout(
                    title=f"Stock Performance Over the Last {chart_period}",
                    xaxis_title="Date",
                    yaxis_title="Price (in USD)",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # 7. Generate Downloadable Report Content
                st.markdown("---")
                st.markdown("### 📥 Download Results")
                
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
                full_md_report = f"""# AI Investment Strategist Report
                
**Date:** {current_date}
**Symbols Analyzed:** {', '.join(stocks_symbols)}
**LLM Model Used:** {provider} ({model_id})
**Portfolio Budget:** ${budget:,} ({risk_tolerance} Risk Tolerance)

---

## 1. Live Market Summary
"""
                for sym, metrics in live_metrics.items():
                    full_md_report += f"- **{metrics['name']} ({sym})**: ${metrics['price']:.2f} | PE: {metrics['pe_ratio']} | Div Yield: {metrics['dividend_yield']} | Market Cap: {metrics['market_cap']}\n"
                
                full_md_report += "\n## 2. Sentiment Analysis Summary\n"
                for sym, data in sentiment_data.items():
                    full_md_report += f"- **{sym}**: {data['sentiment']} (Score: {data['score']:+}) - {data['summary']}\n"
                
                full_md_report += f"\n## 3. Recommended Allocation (${budget:,} - {risk_tolerance} Risk)\n"
                for sym, pct in allocations.items():
                    amount = (pct / 100.0) * budget
                    full_md_report += f"- **{sym}**: {pct:.1f}% (${amount:,.2f})\n"
                
                full_md_report += f"\n---\n\n## 4. Full Report Details\n\n{report}"
                
                st.download_button(
                    label="📥 Download Full Report (Markdown)",
                    data=full_md_report,
                    file_name=f"Investment_Report_{'_'.join(stocks_symbols)}.md",
                    mime="text/markdown"
                )
        except Exception as exc:
            st.error(f"Failed to generate report: {exc}")
