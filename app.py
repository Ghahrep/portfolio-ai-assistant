import streamlit as st
import asyncio
import json
import re
import sqlite3
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Claude integration
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    st.warning("⚠️ Anthropic library not installed. Advanced AI features disabled.")

# Page configuration
st.set_page_config(
    page_title="Hybrid Portfolio AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Material Design CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .main-container {
        font-family: 'Roboto', sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .chat-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        padding: 16px;
    }
    
    .message {
        margin: 12px 0;
        animation: fadeIn 0.3s ease-in;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 8px;
    }
    
    .user-bubble {
        background: #1976D2;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 8px;
    }
    
    .bot-bubble {
        background: #F5F5F5;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        white-space: pre-wrap;
    }
    
    .routing-indicator {
        font-size: 10px;
        color: #666;
        margin-top: 4px;
        font-style: italic;
    }
    
    .claude-indicator {
        color: #9C27B0;
    }
    
    .rules-indicator {
        color: #4CAF50;
    }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1px solid #E0E0E0;
        padding: 12px 16px;
        font-size: 14px;
        font-family: 'Roboto', sans-serif;
    }
    
    .stButton > button {
        background: #1976D2;
        color: white;
        border: none;
        border-radius: 24px;
        padding: 12px 24px;
        font-size: 14px;
        font-weight: 500;
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #1565C0;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .chat-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1976D2, #42A5F5);
        color: white;
        border-radius: 12px 12px 0 0;
        margin-bottom: 0;
    }
    
    .chat-header h1 {
        margin: 0;
        font-size: 24px;
        font-weight: 500;
    }
    
    .chat-header p {
        margin: 8px 0 0 0;
        opacity: 0.9;
        font-size: 14px;
    }
    
    .quick-actions {
        display: flex;
        gap: 8px;
        margin: 16px 0;
        flex-wrap: wrap;
    }
    
    .quick-action {
        background: #E3F2FD;
        color: #1976D2;
        padding: 8px 12px;
        border-radius: 16px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
    }
    
    .quick-action:hover {
        background: #BBDEFB;
        transform: translateY(-1px);
    }
    
    .plot-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 20px;
        margin-top: 20px;
    }
    
    .stats-badge {
        background: #E8F5E8;
        color: #2E7D32;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-left: 8px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DATABASE_FILE = "streamlit_portfolio_agent.db"
BENCHMARK_TICKER = 'SPY'
SCREENING_UNIVERSE = [
    'VTI', 'VEA', 'VWO', 'QQQ', 'IWM', 'MTUM', 'QUAL', 'VLUE', 'USMV',
    'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLY', 'XLP', 'XLU', 'XLRE',
    'TLT', 'IEF', 'LQD', 'HYG', 'GLD', 'DBC', 'UUP'
]

@dataclass
class UserContext:
    user_id: str
    portfolio: Optional[Dict[str, float]] = None
    last_analysis: Optional[Dict] = None
    dialogue_state: str = "start"
    risk_tolerance: str = "moderate"
    investment_goals: str = "general growth"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

# ===================================================================
# DATABASE MANAGER
# ===================================================================
class DatabaseManager:
    def __init__(self, db_file):
        self.db_file = db_file
        try:
            with sqlite3.connect(db_file) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, risk_tolerance TEXT, investment_goals TEXT)")
                conn.execute("CREATE TABLE IF NOT EXISTS portfolios (user_id TEXT PRIMARY KEY, portfolio_json TEXT)")
                conn.execute("""CREATE TABLE IF NOT EXISTS claude_cache 
                               (user_id TEXT, query_hash TEXT, response TEXT, timestamp REAL, 
                                PRIMARY KEY (user_id, query_hash))""")
        except Exception as e:
            st.error(f"Database initialization failed: {e}")

    def get_user_context(self, user_id: str) -> UserContext:
        context = UserContext(user_id=user_id)
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT risk_tolerance, investment_goals FROM users WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    context.risk_tolerance, context.investment_goals = user_row
                cursor.execute("SELECT portfolio_json FROM portfolios WHERE user_id = ?", (user_id,))
                portfolio_row = cursor.fetchone()
                if portfolio_row and portfolio_row[0]:
                    try:
                        context.portfolio = json.loads(portfolio_row[0])
                    except json.JSONDecodeError:
                        context.portfolio = None
        except Exception as e:
            st.error(f"Failed to load user context: {e}")
        return context

    def save_user_context(self, context: UserContext):
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute("INSERT OR REPLACE INTO users (user_id, risk_tolerance, investment_goals) VALUES (?, ?, ?)",
                           (context.user_id, context.risk_tolerance, context.investment_goals))
                conn.execute("INSERT OR REPLACE INTO portfolios (user_id, portfolio_json) VALUES (?, ?)", 
                           (context.user_id, json.dumps(context.portfolio) if context.portfolio else None))
        except Exception as e:
            st.error(f"Failed to save context: {e}")

    def cache_claude_response(self, user_id: str, query: str, response: str):
        """Cache Claude responses to reduce API calls"""
        try:
            query_hash = str(hash(query.lower().strip()))
            with sqlite3.connect(self.db_file) as conn:
                conn.execute("INSERT OR REPLACE INTO claude_cache VALUES (?, ?, ?, ?)",
                            (user_id, query_hash, response, time.time()))
        except Exception as e:
            st.error(f"Failed to cache response: {e}")

    def get_cached_response(self, user_id: str, query: str, max_age_hours: int = 24) -> Optional[str]:
        """Retrieve cached Claude response if recent enough"""
        try:
            query_hash = str(hash(query.lower().strip()))
            max_age = time.time() - (max_age_hours * 3600)
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT response FROM claude_cache WHERE user_id = ? AND query_hash = ? AND timestamp > ?",
                              (user_id, query_hash, max_age))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception:
            return None

# ===================================================================
# CLAUDE AGENT LAYER
# ===================================================================
class ClaudeAgentLayer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self.enabled = False
        
        if CLAUDE_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.enabled = True
            except Exception as e:
                st.error(f"Claude initialization failed: {e}")
        
        if not self.enabled:
            st.info("💡 Set ANTHROPIC_API_KEY environment variable to enable advanced AI features")

    def build_context_prompt(self, context: UserContext) -> str:
        """Build context-aware system prompt"""
        base_prompt = """You are an expert portfolio analyst assistant. You help users analyze and optimize their investment portfolios.

Available functions that you can reference (but don't call directly):
- Risk analysis (volatility, Sharpe ratio, beta, alpha, max drawdown)
- Portfolio optimization (Max Sharpe, Minimize Volatility)
- Factor analysis (Fama-French factors)
- What-if scenario analysis
- Risk contribution analysis

Guidelines:
- Be conversational and educational
- Provide actionable investment advice
- Handle edge cases professionally
- Set realistic expectations about returns
- Always consider risk management
- Explain complex concepts in simple terms"""

        if context.portfolio:
            portfolio_str = ", ".join([f"{t}: {w:.1%}" for t, w in context.portfolio.items()])
            base_prompt += f"\n\nCurrent user portfolio: {portfolio_str}"
        
        base_prompt += f"\nUser risk tolerance: {context.risk_tolerance}"
        base_prompt += f"\nUser investment goals: {context.investment_goals}"
        
        return base_prompt

    def build_conversation_context(self, context: UserContext, max_history: int = 6) -> List[Dict[str, str]]:
        """Build conversation history for Claude"""
        recent_history = context.conversation_history[-max_history:] if context.conversation_history else []
        return recent_history

    async def process_complex_query(self, message: str, context: UserContext, db: DatabaseManager) -> Dict[str, Any]:
        """Process complex queries using Claude"""
        if not self.enabled:
            return {
                'text': "Advanced AI features not available. Please set ANTHROPIC_API_KEY environment variable.",
                'tokens_used': 0
            }

        # Check cache first
        cached = db.get_cached_response(context.user_id, message)
        if cached:
            return {'text': cached, 'tokens_used': 0, 'cached': True}

        try:
            system_prompt = self.build_context_prompt(context)
            conversation_history = self.build_conversation_context(context)
            
            messages = conversation_history + [{"role": "user", "content": message}]
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=system_prompt,
                messages=messages
            )
            
            result = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Cache the response
            db.cache_claude_response(context.user_id, message, result)
            
            # Update conversation history
            context.conversation_history.append({"role": "user", "content": message})
            context.conversation_history.append({"role": "assistant", "content": result})
            
            # Keep conversation history manageable
            if len(context.conversation_history) > 12:
                context.conversation_history = context.conversation_history[-12:]
            
            return {'text': result, 'tokens_used': tokens_used, 'cached': False}
            
        except Exception as e:
            return {
                'text': f"I encountered an issue processing your request: {str(e)}",
                'tokens_used': 0
            }

# ===================================================================
# SMART NLU WITH HYBRID ROUTING
# ===================================================================
class SmartNLUProcessor:
    def should_use_claude(self, message: str, context: UserContext) -> bool:
        """Determine if query should be routed to Claude or handled by rules"""
        message_lower = message.lower().strip()
        
        # Always use rules for simple patterns
        simple_patterns = [
            r'\d+%\s*[A-Z]{2,5}',
            r'^(show|view|display)\s+(my\s+)?(portfolio|holdings)',
            r'^(delete|clear|reset)\s+(my\s+)?(portfolio|holdings)',
            r'^(hi|hello|hey|help)$'
        ]
        
        # Force rules for queries with specific handlers
        rule_keywords = [
            'how risky is my portfolio',
            'analyze my factor exposure',
            'show my factor exposures',
            'where is my risk coming from',
            'optimize my portfolio',
            'what if i sell'
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, message_lower):
                return False
                
        for keyword in rule_keywords:
            if keyword in message_lower:
                return False
        
        # Use Claude for complex scenarios
        complex_indicators = [
            'why', 'how can', 'what does', 'explain', 'difference between', 'should i',
            'and', 'also', 'but', 'however', 'though',
            'compare', 'versus', 'vs', 'better than', 'worse than', 
            'advice', 'recommend', 'suggest', 'think', 'opinion', 'strategy',
            'market', 'economy', 'inflation', 'best performing', 'sp 500',
            'approach', 'consider', 'evaluation', 'analysis', 'diversify',
            'maybe', 'perhaps', 'might', 'could', 'would'
        ]
        
        if any(indicator in message_lower for indicator in complex_indicators):
            return True
        
        # If message is long (suggests complexity)
        if len(message.split()) > 8:
            return True
        
        # If user has been in a conversation with Claude recently
        if context.conversation_history and len(context.conversation_history) > 0:
            last_message = context.conversation_history[-1]
            if last_message.get('role') == 'assistant' and len(last_message.get('content', '')) > 200:
                return True
        
        return False

    def parse(self, message: str, context: UserContext) -> Dict[str, Any]:
        """Parse message and determine routing"""
        use_claude = self.should_use_claude(message, context)
        
        if use_claude:
            return {"intent": "claude_query", "text": message, "routing": "claude"}
        
        # Rule-based intent detection
        message_lower = message.lower().strip()
        intent = "general_query"
        
        if all(k in message_lower for k in ["factor", "exposur"]): intent = "factor_analysis"
        elif any(k in message_lower for k in ["help", "example", "what can you do"]): intent = "help"
        elif any(k in message_lower for k in ["delete", "clear", "reset"]): intent = "delete_portfolio"
        elif "add" in message_lower and "%" in message: intent = "add_to_portfolio"
        elif "what if" in message_lower or "simulate" in message_lower: intent = "what_if_analysis"
        elif any(k in message_lower for k in ["set risk", "my risk is"]): intent = "set_profile"
        elif any(k in message_lower for k in ["tail risk", "hedge", "crisis"]): intent = "tail_risk_screening"
        elif any(k in message_lower for k in ["uncorrelated", "diversify", "correlation"]): intent = "correlation_screening"
        elif any(k in message_lower for k in ["optimize", "rebalance"]): intent = "optimization"
        elif any(k in message_lower for k in ["risk", "risky", "analyze"]): intent = "risk_analysis"
        elif any(k in message_lower for k in ["show", "view", "holding", "portfolio"]): intent = "show_portfolio"
        elif '%' in message and any(char.isalpha() for char in message): intent = "portfolio_input"
        elif any(k in message_lower for k in ['hello', 'hi', 'hey']): intent = "greeting"
        
        return {"intent": intent, "text": message, "routing": "rules"}

# ===================================================================
# ADVANCED PORTFOLIO ENGINE (Enhanced)
# ===================================================================
class AdvancedPortfolioEngine:
    def __init__(self):
        self.benchmark_ticker = BENCHMARK_TICKER
        
    @st.cache_data(ttl=3600)
    def fetch_data(_self, tickers: List[str], period: str = "2y"):
        """Fetch market data for given tickers"""
        try:
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if len(tickers) == 1:
                return pd.DataFrame({tickers[0]: data['Close']})
            elif 'Close' in data.columns:
                return data['Close']
            else:
                return data.xs('Close', axis=1, level=1)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def analyze_portfolio(self, portfolio: Dict[str, float]):
        """Run comprehensive portfolio analysis"""
        tickers = list(portfolio.keys()) + [self.benchmark_ticker]
        data = self.fetch_data(tickers)
        
        if data is None or data.empty:
            return None
            
        if self.benchmark_ticker not in data.columns:
            return None
            
        # Calculate returns
        returns = np.log(data / data.shift(1)).dropna()
        
        # Portfolio returns
        portfolio_tickers = [t for t in portfolio.keys() if t in returns.columns]
        if not portfolio_tickers:
            return None
            
        portfolio_weights = np.array([portfolio[t] for t in portfolio_tickers])
        portfolio_returns = returns[portfolio_tickers].dot(portfolio_weights)
        benchmark_returns = returns[self.benchmark_ticker]
        
        # Calculate comprehensive metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Beta and Alpha
        beta_model = LinearRegression().fit(
            benchmark_returns.values.reshape(-1, 1),
            portfolio_returns.values.reshape(-1, 1)
        )
        beta = beta_model.coef_[0][0]
        alpha = (portfolio_returns.mean() - beta * benchmark_returns.mean()) * 252
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'alpha': alpha,
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'returns': returns,
            'data': data,
            'portfolio_tickers': portfolio_tickers
        }
    
    def run_factor_analysis(self, portfolio_returns: pd.Series):
        """Run Fama-French factor analysis"""
        try:
            start_date = portfolio_returns.index.min()
            end_date = portfolio_returns.index.max()
            
            ff_factors = web.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start_date,
                end_date
            )[0] / 100
            
            data = pd.DataFrame(portfolio_returns).join(ff_factors, how='inner').dropna()
            
            if data.empty:
                return {"error": "Could not align portfolio data with factor data"}
            
            y = data.iloc[:, 0] - data['RF']
            X = data[['Mkt-RF', 'SMB', 'HML']]
            
            model = LinearRegression().fit(X, y)
            
            return {
                'market_beta': model.coef_[0],
                'size_factor': model.coef_[1],
                'value_factor': model.coef_[2],
                'alpha': model.intercept_ * 252,
                'r_squared': model.score(X, y)
            }
        except Exception as e:
            return {"error": f"Factor analysis failed: {str(e)}"}
    
    def optimize_portfolio(self, returns: pd.DataFrame, objective: str = "Max Sharpe"):
        """Optimize portfolio weights"""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(returns.columns)
        
        def portfolio_volatility(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_vol = portfolio_volatility(weights)
            return -(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 0.4) for _ in range(num_assets))
        
        if objective == "Max Sharpe":
            obj_func = negative_sharpe
        elif objective == "Min Volatility":
            obj_func = portfolio_volatility
        else:
            obj_func = negative_sharpe
        
        result = minimize(
            obj_func,
            num_assets * [1.0 / num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return dict(zip(returns.columns, result.x))
        else:
            return None

    def screen_tail_risk_hedges(self, portfolio_returns: pd.Series, n_results: int = 5):
        """Screen for tail risk hedges"""
        screening_data = self.fetch_data(SCREENING_UNIVERSE + [self.benchmark_ticker])
        
        if screening_data is None:
            return {"error": "Could not fetch screening data"}
        
        spy_returns = np.log(screening_data[self.benchmark_ticker] / screening_data[self.benchmark_ticker].shift(1))
        spy_cumulative = (1 + spy_returns).cumprod()
        spy_peak = spy_cumulative.expanding().max()
        spy_drawdown = (spy_cumulative - spy_peak) / spy_peak
        
        worst_drawdown_end = spy_drawdown.idxmin()
        worst_drawdown_start = spy_cumulative.loc[:worst_drawdown_end].idxmax()
        
        crisis_period = screening_data.loc[worst_drawdown_start:worst_drawdown_end]
        crisis_returns = np.log(crisis_period / crisis_period.shift(1)).dropna()
        
        aligned_data = pd.DataFrame(portfolio_returns).join(crisis_returns, how='inner')
        
        if aligned_data.empty:
            return {"error": "Could not align portfolio data with screening universe"}
        
        portfolio_col = aligned_data.columns[0]
        hedges = []
        
        for ticker in SCREENING_UNIVERSE:
            if ticker in aligned_data.columns:
                correlation = aligned_data[portfolio_col].corr(aligned_data[ticker])
                if not np.isnan(correlation):
                    hedges.append({
                        'ticker': ticker,
                        'crisis_correlation': round(correlation, 3),
                        'hedge_quality': 'Excellent' if correlation < -0.5 else 'Good' if correlation < 0 else 'Poor'
                    })
        
        hedges.sort(key=lambda x: x['crisis_correlation'])
        return hedges[:n_results]
    
    def screen_uncorrelated_assets(self, portfolio_returns: pd.Series, n_results: int = 5):
        """Screen for uncorrelated assets"""
        screening_data = self.fetch_data(SCREENING_UNIVERSE)
        
        if screening_data is None:
            return {"error": "Could not fetch screening data"}
        
        screening_returns = np.log(screening_data / screening_data.shift(1)).dropna()
        aligned_data = pd.DataFrame(portfolio_returns).join(screening_returns, how='inner')
        
        if aligned_data.empty:
            return {"error": "Could not align data"}
        
        portfolio_col = aligned_data.columns[0]
        correlations = []
        
        for ticker in SCREENING_UNIVERSE:
            if ticker in aligned_data.columns:
                corr = aligned_data[portfolio_col].corr(aligned_data[ticker])
                if not np.isnan(corr):
                    correlations.append({
                        'ticker': ticker,
                        'correlation': round(corr, 3),
                        'diversification_benefit': 'High' if abs(corr) < 0.3 else 'Medium' if abs(corr) < 0.6 else 'Low'
                    })
        
        correlations.sort(key=lambda x: abs(x['correlation']))
        return correlations[:n_results]

# ===================================================================
# HYBRID CHATBOT (Enhanced)
# ===================================================================
class HybridChatbot:
    def __init__(self, claude_api_key: Optional[str] = None):
        self.engine = AdvancedPortfolioEngine()
        self.nlu = SmartNLUProcessor()
        self.claude = ClaudeAgentLayer(claude_api_key)
        self.db = DatabaseManager(DATABASE_FILE)
        self.routing_stats = {"rules": 0, "claude": 0}
        
    def parse_portfolio(self, text: str) -> Optional[Dict[str, float]]:
        """Parse portfolio from text"""
        pattern = r'(\d+(?:\.\d*)?%)\s*([A-Z]{1,5})'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            return None
        
        portfolio = {}
        for percent, ticker in matches:
            portfolio[ticker.upper()] = float(percent.strip('%')) / 100
        
        total = sum(portfolio.values())
        if 0.9 <= total <= 1.1:
            portfolio = {k: v/total for k, v in portfolio.items()}
        
        return portfolio
    
    def parse_what_if_changes(self, text: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Parse what-if scenario changes"""
        sell_pattern = r'sell\s+(\d+(?:\.\d*)?%)\s+([A-Z]{1,5})'
        buy_pattern = r'buy\s+(\d+(?:\.\d*)?%)\s+([A-Z]{1,5})'
        
        sells = re.findall(sell_pattern, text, re.IGNORECASE)
        buys = re.findall(buy_pattern, text, re.IGNORECASE)
        
        if not sells and not buys:
            return None
        
        changes = {"sell": {}, "buy": {}}
        
        for percent, ticker in sells:
            changes["sell"][ticker.upper()] = float(percent.strip('%')) / 100
        
        for percent, ticker in buys:
            changes["buy"][ticker.upper()] = float(percent.strip('%')) / 100
        
        return changes

    async def generate_response(self, message: str, context: UserContext):
        """Generate response using hybrid routing"""
        nlu_result = self.nlu.parse(message, context)
        routing = nlu_result.get("routing", "rules")
        
        # Track routing stats
        self.routing_stats[routing] += 1
        
        if routing == "claude":
            claude_response = await self.claude.process_complex_query(message, context, self.db)
            response = {
                'text': claude_response['text'],
                'routing': 'claude',
                'tokens_used': claude_response.get('tokens_used', 0),
                'cached': claude_response.get('cached', False)
            }
        else:
            # Use rule-based handlers
            response = self._handle_rule_based(nlu_result, context)
            response['routing'] = 'rules'
            response['tokens_used'] = 0
        
        # Save context
        self.db.save_user_context(context)
        return response
    
    def _handle_rule_based(self, nlu_result: Dict[str, Any], context: UserContext) -> Dict[str, Any]:
        """Handle rule-based responses"""
        intent = nlu_result.get("intent")
        message = nlu_result.get("text")
        
        if intent == "greeting":
            return {
                'text': "👋 **Welcome to your Hybrid AI Portfolio Assistant!**\n\n"
                       "I combine fast rule-based analysis with advanced AI reasoning for the best experience.\n\n"
                       "🚀 **What makes me special:**\n"
                       "• **Smart Routing:** Simple queries get instant responses, complex questions get AI analysis\n"
                       "• **Cost Optimized:** Institutional-grade analytics without breaking the bank\n"
                       "• **Context Aware:** I remember our conversation and your portfolio\n\n"
                       "**Quick Start:** Tell me your portfolio like `'60% AAPL, 30% GOOGL, 10% BONDS'`\n"
                       "**Or ask:** `'what can you do?'` to see all features"
            }
        
        elif intent == "help":
            return {
                'text': "🤖 **Hybrid AI Portfolio Assistant - Full Feature Guide**\n\n"
                       "**🔧 SMART ROUTING SYSTEM:**\n"
                       "• Simple queries → Fast rule-based responses (0 cost)\n"
                       "• Complex questions → Advanced AI analysis (optimized cost)\n\n"
                       "**📊 PORTFOLIO SETUP & MANAGEMENT**\n"
                       "• `'60% AAPL, 30% GOOGL, 10% BONDS'` - Set portfolio (Rules)\n"
                       "• `'add 15% TSLA'` - Add positions (Rules)\n"
                       "• `'show my portfolio'` - View holdings (Rules)\n\n"
                       "**🔍 RISK ANALYSIS**\n"
                       "• `'how risky is my portfolio?'` - Risk dashboard (Rules)\n"
                       "• `'show my factor exposures'` - Fama-French analysis (Rules)\n\n"
                       "**🎯 OPTIMIZATION**\n"
                       "• `'optimize my portfolio'` - Smart allocation (Rules)\n"
                       "• `'what if I sell 10% AAPL and buy 10% QQQ?'` - Scenarios (Rules)\n\n"
                       "**🤖 AI-POWERED FEATURES** (Advanced reasoning)\n"
                       "• `'How can I diversify my portfolio better?'` - Strategy advice\n"
                       "• `'What's the current market regime?'` - Market analysis\n"
                       "• `'Should I be worried about inflation?'` - Economic insights\n"
                       "• `'Explain why my alpha is negative'` - Educational content\n\n"
                       "**💡 PRO TIP:** The system automatically chooses the best approach for each query!"
            }
        
        elif intent == "portfolio_input":
            portfolio = self.parse_portfolio(message)
            if portfolio:
                context.portfolio = portfolio
                analysis = self.engine.analyze_portfolio(portfolio)
                context.last_analysis = analysis
                
                if analysis:
                    return {
                        'text': f"✅ **Portfolio Set Successfully!** (Rules Engine)\n\n"
                               f"**Holdings:** {len(portfolio)} assets\n"
                               f"**Volatility:** {analysis['volatility']:.2%}\n"
                               f"**Sharpe Ratio:** {analysis['sharpe_ratio']:.2f}\n"
                               f"**Beta:** {analysis['beta']:.2f}\n"
                               f"**Max Drawdown:** {analysis['max_drawdown']:.2%}\n\n"
                               f"📊 Interactive charts are now available below!\n\n"
                               f"**🔥 Try these next:**\n"
                               f"• `'optimize my portfolio'` (Rules - Fast)\n"
                               f"• `'how can I improve my diversification?'` (AI - Deep analysis)",
                        'show_charts': True
                    }
                else:
                    return {'text': "❌ I couldn't analyze your portfolio. Please check the ticker symbols."}
            else:
                return {'text': "❌ I couldn't parse your portfolio. Try: '60% AAPL, 30% GOOGL, 10% BONDS'"}
        
        elif intent == "risk_analysis":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
            
            analysis = context.last_analysis
            risk_level = "Low" if analysis['volatility'] < 0.15 else "High" if analysis['volatility'] > 0.25 else "Moderate"
            
            return {
                'text': f"📊 **Risk Analysis Dashboard** (Rules Engine)\n\n"
                       f"**Risk Level:** {risk_level}\n"
                       f"**Volatility:** {analysis['volatility']:.2%}\n"
                       f"**Sharpe Ratio:** {analysis['sharpe_ratio']:.2f}\n"
                       f"**Beta:** {analysis['beta']:.2f}\n"
                       f"**Alpha:** {analysis['alpha']:.2%}\n"
                       f"**Max Drawdown:** {analysis['max_drawdown']:.2%}\n\n"
                       f"**Quick Insights:**\n"
                       f"• {'Higher' if analysis['beta'] > 1.1 else 'Lower' if analysis['beta'] < 0.9 else 'Similar'} volatility vs market\n"
                       f"• {'Outperforming' if analysis['alpha'] > 0 else 'Underperforming'} by {abs(analysis['alpha']):.2%}\n\n"
                       f"**🤖 Want deeper analysis?** Ask: 'Why is my portfolio performing this way?'",
                'show_charts': True
            }
        
        elif intent == "factor_analysis":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first."}
            
            factors = self.engine.run_factor_analysis(context.last_analysis['portfolio_returns'])
            
            if "error" in factors:
                return {'text': f"❌ Factor analysis failed: {factors['error']}"}
            
            return {
                'text': f"📈 **Fama-French Factor Analysis** (Rules Engine)\n\n"
                       f"**Market Beta:** {factors['market_beta']:.2f}\n"
                       f"**Size Factor (SMB):** {factors['size_factor']:.2f}\n"
                       f"**Value Factor (HML):** {factors['value_factor']:.2f}\n"
                       f"**Alpha:** {factors['alpha']:.2%}\n"
                       f"**R-squared:** {factors['r_squared']:.1%}\n\n"
                       f"**Style Classification:**\n"
                       f"• {'Small-cap' if factors['size_factor'] > 0.2 else 'Large-cap' if factors['size_factor'] < -0.2 else 'Size-neutral'} tilt\n"
                       f"• {'Value' if factors['value_factor'] > 0.2 else 'Growth' if factors['value_factor'] < -0.2 else 'Style-neutral'} orientation"
            }
        
        elif intent == "optimization":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first."}
            
            context.dialogue_state = 'awaiting_optimization_choice'
            return {
                'text': f"🎯 **Portfolio Optimization** (Rules Engine)\n\n"
                       f"Choose your optimization objective:\n\n"
                       f"**1. Max Sharpe Ratio** - Maximize risk-adjusted returns\n"
                       f"**2. Min Volatility** - Minimize portfolio risk\n\n"
                       f"Just reply with 'Max Sharpe' or 'Min Volatility'"
            }
        
        elif intent == "tail_risk_screening":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first."}
            
            hedges = self.engine.screen_tail_risk_hedges(context.last_analysis['portfolio_returns'])
            
            if "error" in hedges:
                return {'text': f"❌ Screening failed: {hedges['error']}"}
            
            hedge_text = "\n".join([f"• **{h['ticker']}** - Crisis correlation: {h['crisis_correlation']:.2f} ({h['hedge_quality']})" for h in hedges])
            
            return {
                'text': f"🛡️ **Tail Risk Hedge Analysis** (Rules Engine)\n\n"
                       f"Assets that performed well during market crashes:\n\n"
                       f"{hedge_text}\n\n"
                       f"**Note:** Negative correlations indicate better hedging properties."
            }
        
        elif intent == "correlation_screening":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first."}
            
            uncorrelated = self.engine.screen_uncorrelated_assets(context.last_analysis['portfolio_returns'])
            
            if "error" in uncorrelated:
                return {'text': f"❌ Screening failed: {uncorrelated['error']}"}
            
            assets_text = "\n".join([f"• **{a['ticker']}** - Correlation: {a['correlation']:.2f} ({a['diversification_benefit']})" for a in uncorrelated])
            
            return {
                'text': f"📊 **Diversification Opportunities** (Rules Engine)\n\n"
                       f"Low-correlation assets for your portfolio:\n\n"
                       f"{assets_text}\n\n"
                       f"**Lower correlations = better diversification benefits**"
            }
        
        elif intent == "show_portfolio":
            if not context.portfolio:
                return {'text': "You haven't set a portfolio yet. Try: '60% AAPL, 30% GOOGL, 10% BONDS'"}
            
            holdings = "\n".join([f"• **{ticker}:** {weight:.1%}" for ticker, weight in context.portfolio.items()])
            
            return {
                'text': f"📋 **Current Portfolio** (Rules Engine)\n\n{holdings}\n\n"
                       f"**Assets:** {len(context.portfolio)} | **Risk Profile:** {context.risk_tolerance.title()}",
                'show_charts': True
            }
        
        elif intent == "what_if_analysis":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first."}
            
            changes = self.parse_what_if_changes(message)
            if not changes:
                return {'text': "❌ Try: 'what if I sell 10% AAPL and buy 10% MSFT?'"}
            
            # Simulate changes
            new_portfolio = context.portfolio.copy()
            for ticker, amount in changes.get('sell', {}).items():
                if ticker in new_portfolio:
                    new_portfolio[ticker] -= amount
            for ticker, amount in changes.get('buy', {}).items():
                new_portfolio[ticker] = new_portfolio.get(ticker, 0) + amount
            
            new_analysis = self.engine.analyze_portfolio(new_portfolio)
            if not new_analysis:
                return {'text': "❌ Could not analyze simulated portfolio."}
            
            current = context.last_analysis
            
            return {
                'text': f"🔮 **What-If Analysis** (Rules Engine)\n\n"
                       f"**Volatility:** {current['volatility']:.2%} → {new_analysis['volatility']:.2%} "
                       f"({new_analysis['volatility']-current['volatility']:+.2%})\n"
                       f"**Sharpe:** {current['sharpe_ratio']:.2f} → {new_analysis['sharpe_ratio']:.2f} "
                       f"({new_analysis['sharpe_ratio']-current['sharpe_ratio']:+.2f})\n"
                       f"**Beta:** {current['beta']:.2f} → {new_analysis['beta']:.2f} "
                       f"({new_analysis['beta']-current['beta']:+.2f})"
            }
        
        # Handle optimization choice
        elif context.dialogue_state == 'awaiting_optimization_choice':
            context.dialogue_state = 'start'
            
            if "sharpe" in message.lower():
                objective = "Max Sharpe"
            elif "volatility" in message.lower():
                objective = "Min Volatility"
            else:
                return {'text': "❌ Please specify 'Max Sharpe' or 'Min Volatility'"}
            
            optimized = self.engine.optimize_portfolio(
                context.last_analysis['returns'][context.last_analysis['portfolio_tickers']], 
                objective
            )
            
            if not optimized:
                return {'text': "❌ Optimization failed."}
            
            comparison = "\n".join([
                f"• **{ticker}:** {context.portfolio.get(ticker, 0):.1%} → {optimized.get(ticker, 0):.1%}"
                for ticker in set(list(context.portfolio.keys()) + list(optimized.keys()))
            ])
            
            return {
                'text': f"🎯 **Optimization Results** ({objective}) - Rules Engine\n\n"
                       f"{comparison}\n\n"
                       f"**Constraints:** Max 40% per asset, fully invested"
            }
        
        else:
            return {
                'text': "🤔 **Not sure about that request** (Rules Engine)\n\n"
                       "**🔥 Popular Commands:**\n"
                       "• Set portfolio: `'60% AAPL, 30% GOOGL, 10% BONDS'`\n"
                       "• Risk analysis: `'how risky is my portfolio?'`\n"
                       "• Optimization: `'optimize my portfolio'`\n\n"
                       "**🤖 Try asking naturally:** 'How can I improve my portfolio diversification?'"
            }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing analytics"""
        total = sum(self.routing_stats.values())
        if total == 0:
            return {"rules": 0, "claude": 0, "rules_percentage": 0, "claude_percentage": 0}
        
        return {
            "rules": self.routing_stats["rules"],
            "claude": self.routing_stats["claude"],
            "rules_percentage": (self.routing_stats["rules"] / total) * 100,
            "claude_percentage": (self.routing_stats["claude"] / total) * 100,
            "total_queries": total
        }

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'context' not in st.session_state:
    st.session_state.context = UserContext(user_id="streamlit_user")
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = HybridChatbot(claude_api_key=os.getenv("ANTHROPIC_API_KEY"))
if 'total_tokens_used' not in st.session_state:
    st.session_state.total_tokens_used = 0

# Header with routing status
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div class="chat-header">
        <h1>🤖 Hybrid Portfolio AI Assistant</h1>
        <p>Smart routing: Fast rules + Advanced AI reasoning</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.session_state.chatbot.routing_stats["rules"] + st.session_state.chatbot.routing_stats["claude"] > 0:
        stats = st.session_state.chatbot.get_routing_stats()
        st.metric(
            "Routing Efficiency", 
            f"{stats['rules_percentage']:.0f}% Rules",
            f"{stats['claude_percentage']:.0f}% AI"
        )
        st.metric("Total Tokens", st.session_state.total_tokens_used)

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages with routing indicators
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            routing = message.get('routing', 'unknown')
            tokens = message.get('tokens_used', 0)
            cached = message.get('cached', False)
            
            routing_class = "claude-indicator" if routing == "claude" else "rules-indicator"
            routing_text = f"🤖 AI ({tokens} tokens)" if routing == "claude" else "⚡ Rules (0 tokens)"
            if cached:
                routing_text += " [Cached]"
            
            st.markdown(f"""
            <div class="bot-message">
                <div class="bot-bubble">
                    {message['content']}
                    <div class="routing-indicator {routing_class}">{routing_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Quick actions with routing hints
st.markdown("**🚀 Quick Actions:**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💼 Set Portfolio", key="quick_portfolio"):
        message = "60% AAPL, 30% GOOGL, 10% BONDS"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()
    
    if st.button("🤖 AI Strategy", key="quick_ai_strategy"):
        message = "How can I improve my portfolio diversification strategy?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()

with col2:
    if st.button("📊 Risk Analysis", key="quick_risk"):
        message = "how risky is my portfolio?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()
    
    if st.button("🤖 Market Analysis", key="quick_ai_market"):
        message = "What's the current market regime and how should I position my portfolio?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()

with col3:
    if st.button("🎯 Optimize", key="quick_optimize"):
        message = "optimize my portfolio"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()
    
    if st.button("🤖 AI Insights", key="quick_ai_insights"):
        message = "What are the key risks in my portfolio that I should be aware of?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()

with col4:
    if st.button("🔮 What-If", key="quick_whatif"):
        message = "what if I sell 10% AAPL and buy 10% QQQ?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()
    
    if st.button("❓ Help", key="quick_help"):
        message = "what can you do?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        st.session_state.messages.append({"role": "assistant", **response})
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()

# Input container
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Type your message...", key="user_input", label_visibility="collapsed")
with col2:
    send_button = st.button("Send", type="primary")

# Handle user input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate bot response
    response = asyncio.run(st.session_state.chatbot.generate_response(user_input, st.session_state.context))
    
    # Add bot response with routing info
    st.session_state.messages.append({"role": "assistant", **response})
    
    # Update token count
    st.session_state.total_tokens_used += response.get('tokens_used', 0)
    
    # Clear input and rerun
    st.rerun()

# Advanced Charts Section (same as before but with routing awareness)
if st.session_state.context.portfolio and st.session_state.context.last_analysis:
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Composition", "📈 Performance", "⚡ Risk Metrics", "🔄 Correlation"])
    
    with tab1:
        # Portfolio composition
        fig_pie = px.pie(
            values=list(st.session_state.context.portfolio.values()),
            names=list(st.session_state.context.portfolio.keys()),
            title="Portfolio Composition",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        # Performance chart
        analysis = st.session_state.context.last_analysis
        portfolio_returns = analysis['portfolio_returns']
        benchmark_returns = analysis['benchmark_returns']
        
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            mode='lines',
            name='Your Portfolio',
            line=dict(color='#1976D2', width=3)
        ))
        fig_perf.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            mode='lines',
            name='S&P 500 (SPY)',
            line=dict(color='#FF9800', width=2)
        ))
        
        fig_perf.update_layout(
            title="Cumulative Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
            hovermode='x unified'
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with tab3:
        # Risk metrics visualization
        metrics = {
            'Volatility': f"{analysis['volatility']:.2%}",
            'Sharpe Ratio': f"{analysis['sharpe_ratio']:.2f}",
            'Beta': f"{analysis['beta']:.2f}",
            'Max Drawdown': f"{analysis['max_drawdown']:.2%}",
            'Alpha': f"{analysis['alpha']:.2%}"
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Volatility", metrics['Volatility'])
            st.metric("Beta", metrics['Beta'])
        
        with col2:
            st.metric("Sharpe Ratio", metrics['Sharpe Ratio'])
            st.metric("Alpha", metrics['Alpha'])
        
        with col3:
            st.metric("Max Drawdown", metrics['Max Drawdown'])
    
    with tab4:
        # Correlation matrix
        if len(analysis['portfolio_tickers']) > 1:
            returns_data = analysis['returns'][analysis['portfolio_tickers']]
            correlation_matrix = returns_data.corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Asset Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            fig_corr.update_layout(
                title="Asset Correlation Matrix",
                xaxis_title="Assets",
                yaxis_title="Assets"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 2 assets to show correlation matrix")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with system status
stats = st.session_state.chatbot.get_routing_stats()
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: #666; font-size: 12px; border-top: 1px solid #eee; margin-top: 40px;">
    <p><strong>Hybrid Portfolio AI Assistant</strong> • 
    Routing: {stats['rules_percentage']:.0f}% Rules, {stats['claude_percentage']:.0f}% AI • 
    Total Tokens: {st.session_state.total_tokens_used} • 
    {'Claude Enabled' if st.session_state.chatbot.claude.enabled else 'Claude Disabled'}</p>
    <p>⚠️ This tool is for educational purposes only and does not constitute financial advice.</p>
</div>
""", unsafe_allow_html=True)
