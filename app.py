import streamlit as st
import asyncio
import json
import re
import setuptools 
import sqlite3
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

# Page configuration
st.set_page_config(
    page_title="Portfolio AI Assistant",
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
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Configuration
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

class AdvancedPortfolioEngine:
    def __init__(self):
        self.benchmark_ticker = BENCHMARK_TICKER
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
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
            
        # Ensure we have the benchmark
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
            
            # Fetch Fama-French factors
            ff_factors = web.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start_date,
                end_date
            )[0] / 100
            
            # Align data
            data = pd.DataFrame(portfolio_returns).join(ff_factors, how='inner').dropna()
            
            if data.empty:
                return {"error": "Could not align portfolio data with factor data"}
            
            # Run regression
            y = data.iloc[:, 0] - data['RF']  # Excess returns
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
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 0.4) for _ in range(num_assets))  # Max 40% per asset
        
        # Objective function
        if objective == "Max Sharpe":
            obj_func = negative_sharpe
        elif objective == "Min Volatility":
            obj_func = portfolio_volatility
        else:
            obj_func = negative_sharpe
        
        # Optimize
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
        # Fetch screening universe data
        screening_data = self.fetch_data(SCREENING_UNIVERSE + [self.benchmark_ticker])
        
        if screening_data is None:
            return {"error": "Could not fetch screening data"}
        
        # Calculate SPY drawdowns to find crisis periods
        spy_returns = np.log(screening_data[self.benchmark_ticker] / screening_data[self.benchmark_ticker].shift(1))
        spy_cumulative = (1 + spy_returns).cumprod()
        spy_peak = spy_cumulative.expanding().max()
        spy_drawdown = (spy_cumulative - spy_peak) / spy_peak
        
        # Find the worst drawdown period
        worst_drawdown_end = spy_drawdown.idxmin()
        worst_drawdown_start = spy_cumulative.loc[:worst_drawdown_end].idxmax()
        
        # Calculate crisis period performance
        crisis_period = screening_data.loc[worst_drawdown_start:worst_drawdown_end]
        crisis_returns = np.log(crisis_period / crisis_period.shift(1)).dropna()
        
        # Align with portfolio returns
        aligned_data = pd.DataFrame(portfolio_returns).join(crisis_returns, how='inner')
        
        if aligned_data.empty:
            return {"error": "Could not align portfolio data with screening universe"}
        
        # Calculate correlations during crisis
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
        
        # Sort by most negative correlation (best hedges)
        hedges.sort(key=lambda x: x['crisis_correlation'])
        return hedges[:n_results]
    
    def screen_uncorrelated_assets(self, portfolio_returns: pd.Series, n_results: int = 5):
        """Screen for uncorrelated assets"""
        screening_data = self.fetch_data(SCREENING_UNIVERSE)
        
        if screening_data is None:
            return {"error": "Could not fetch screening data"}
        
        screening_returns = np.log(screening_data / screening_data.shift(1)).dropna()
        
        # Align with portfolio returns
        aligned_data = pd.DataFrame(portfolio_returns).join(screening_returns, how='inner')
        
        if aligned_data.empty:
            return {"error": "Could not align data"}
        
        # Calculate correlations
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
        
        # Sort by lowest absolute correlation
        correlations.sort(key=lambda x: abs(x['correlation']))
        return correlations[:n_results]
    
    def simulate_what_if(self, current_portfolio: Dict[str, float], changes: Dict[str, Dict[str, float]]):
        """Simulate what-if scenario"""
        # Create new portfolio
        new_portfolio = current_portfolio.copy()
        
        # Apply sells
        for ticker, amount in changes.get('sell', {}).items():
            if ticker in new_portfolio:
                new_portfolio[ticker] -= amount
                if new_portfolio[ticker] <= 0:
                    del new_portfolio[ticker]
        
        # Apply buys
        for ticker, amount in changes.get('buy', {}).items():
            new_portfolio[ticker] = new_portfolio.get(ticker, 0) + amount
        
        # Analyze new portfolio
        new_analysis = self.analyze_portfolio(new_portfolio)
        
        return new_analysis, new_portfolio

class AdvancedChatbot:
    def __init__(self):
        self.engine = AdvancedPortfolioEngine()
        
    def parse_intent(self, message: str) -> str:
        """Parse user intent from message"""
        message_lower = message.lower().strip()
        
        # Priority order matters!
        if all(k in message_lower for k in ["factor", "exposur"]): return "factor_analysis"
        elif any(k in message_lower for k in ["help", "example", "what can you do", "what do you do"]): return "help"
        elif any(k in message_lower for k in ["delete", "clear", "reset"]): return "delete_portfolio"
        elif "add" in message_lower and "%" in message: return "add_to_portfolio"
        elif "what if" in message_lower or "simulate" in message_lower: return "what_if_analysis"
        elif any(k in message_lower for k in ["set risk", "my risk is", "risk tolerance"]): return "set_profile"
        elif any(k in message_lower for k in ["tail risk", "hedge", "crisis"]): return "tail_risk_screening"
        elif any(k in message_lower for k in ["uncorrelated", "diversify", "correlation"]): return "correlation_screening"
        elif any(k in message_lower for k in ["optimize", "rebalance"]): return "optimization"
        elif any(k in message_lower for k in ["risk", "risky", "analyze"]): return "risk_analysis"
        elif any(k in message_lower for k in ["show", "view", "holding", "portfolio"]): return "show_portfolio"
        elif '%' in message and any(char.isalpha() for char in message): return "portfolio_input"
        elif any(k in message_lower for k in ['hello', 'hi', 'hey']): return "greeting"
        
        return "general_query"
    
    def parse_portfolio(self, text: str) -> Optional[Dict[str, float]]:
        """Parse portfolio from text"""
        pattern = r'(\d+(?:\.\d*)?%)\s*([A-Z]{1,5})'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            return None
        
        portfolio = {}
        for percent, ticker in matches:
            portfolio[ticker.upper()] = float(percent.strip('%')) / 100
        
        # Normalize if close to 100%
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
    
    def generate_response(self, message: str, context: UserContext):
        """Generate comprehensive response"""
        intent = self.parse_intent(message)
        
        if intent == "greeting":
            return {
                'text': "👋 **Welcome to your AI Portfolio Assistant!**\n\n"
                       "I'm powered by institutional-grade analytics normally reserved for Wall Street. Here's what makes me special:\n\n"
                       "🎯 **Advanced Analytics:** Fama-French factors, optimization algorithms, crisis analysis\n"
                       "🔮 **Scenario Testing:** See how changes affect your portfolio before you make them\n"
                       "🛡️ **Smart Screening:** Find hedges and diversification opportunities\n"
                       "📊 **Professional Charts:** Visual analysis with interactive dashboards\n\n"
                       "**Quick Start:**\n"
                       "• Tell me your portfolio: `'60% AAPL, 30% GOOGL, 10% BONDS'`\n"
                       "• Or ask: `'what can you do?'` to see all features\n\n"
                       "**Pro Tip:** Try asking `'find tail risk hedges'` or `'optimize my portfolio'` after setting up your holdings!"
            }
        
        elif intent == "help":
            return {
                'text': "🚀 **Your Advanced AI Portfolio Assistant**\n\n"
                       "I'm powered by institutional-grade analytics. Here's what I can do:\n\n"
                       "**📊 PORTFOLIO SETUP & MANAGEMENT**\n"
                       "• `'60% AAPL, 30% GOOGL, 10% BONDS'` - Set your portfolio\n"
                       "• `'add 15% TSLA to my portfolio'` - Add new positions\n"
                       "• `'show my portfolio'` - View current holdings\n"
                       "• `'delete my portfolio'` - Start over\n\n"
                       "**🔍 ADVANCED RISK ANALYSIS**\n"
                       "• `'how risky is my portfolio?'` - Full risk dashboard\n"
                       "• `'show my factor exposures'` - Fama-French factor analysis\n"
                       "• `'analyze my portfolio'` - Comprehensive metrics\n\n"
                       "**🎯 PORTFOLIO OPTIMIZATION**\n"
                       "• `'optimize my portfolio'` - I'll ask for your goal\n"
                       "• `'optimize for max sharpe ratio'` - Maximize risk-adjusted returns\n"
                       "• `'optimize for min volatility'` - Minimize risk\n\n"
                       "**🔮 SCENARIO ANALYSIS**\n"
                       "• `'what if I sell 20% AAPL and buy 20% MSFT?'` - Simulate changes\n"
                       "• `'what if I add 10% Bitcoin?'` - Test new positions\n\n"
                       "**🛡️ ADVANCED SCREENING**\n"
                       "• `'find tail risk hedges'` - Crisis protection assets\n"
                       "• `'find uncorrelated assets'` - Diversification opportunities\n"
                       "• `'screen for hedges'` - Defensive positioning\n\n"
                       "**⚙️ PERSONALIZATION**\n"
                       "• `'set my risk tolerance to aggressive'` - Customize recommendations\n"
                       "• `'set my risk tolerance to conservative'` - Adjust strategy\n\n"
                       "**💡 PRO TIPS:**\n"
                       "✨ Try: 'optimize my portfolio for max Sharpe ratio'\n"
                       "✨ Try: 'what if I sell 10% AAPL and buy 10% QQQ?'\n"
                       "✨ Try: 'find assets that hedge against market crashes'\n\n"
                       "**Just type naturally - I understand complex financial questions!**"
            }
        
        elif intent == "portfolio_input":
            portfolio = self.parse_portfolio(message)
            if portfolio:
                context.portfolio = portfolio
                analysis = self.engine.analyze_portfolio(portfolio)
                context.last_analysis = analysis
                
                if analysis:
                    return {
                        'text': f"✅ **Portfolio Set Successfully!**\n\n"
                               f"**Holdings:** {len(portfolio)} assets\n"
                               f"**Volatility:** {analysis['volatility']:.2%}\n"
                               f"**Sharpe Ratio:** {analysis['sharpe_ratio']:.2f}\n"
                               f"**Beta:** {analysis['beta']:.2f}\n"
                               f"**Max Drawdown:** {analysis['max_drawdown']:.2%}\n\n"
                               f"📊 Charts updated below!\n\n"
                               f"**🚀 What's Next? Try These:**\n"
                               f"• `'show my factor exposures'` - See your factor tilts\n"
                               f"• `'optimize my portfolio'` - Get better allocation\n"
                               f"• `'find tail risk hedges'` - Protect against crashes\n"
                               f"• `'what if I sell 10% {list(portfolio.keys())[0]} and buy 10% QQQ?'` - Test scenarios",
                        'show_charts': True
                    }
                else:
                    return {'text': "❌ I couldn't analyze your portfolio. Please check the ticker symbols and try again."}
            else:
                return {'text': "❌ I couldn't parse your portfolio. Try: '60% AAPL, 30% GOOGL, 10% BONDS'"}
        
        elif intent == "risk_analysis":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
            
            analysis = context.last_analysis
            risk_level = "Low" if analysis['volatility'] < 0.15 else "High" if analysis['volatility'] > 0.25 else "Moderate"
            
            return {
                'text': f"📊 **Comprehensive Risk Analysis**\n\n"
                       f"**Risk Level:** {risk_level}\n"
                       f"**Volatility:** {analysis['volatility']:.2%} (annualized)\n"
                       f"**Sharpe Ratio:** {analysis['sharpe_ratio']:.2f}\n"
                       f"**Beta:** {analysis['beta']:.2f}\n"
                       f"**Alpha:** {analysis['alpha']:.2%}\n"
                       f"**Max Drawdown:** {analysis['max_drawdown']:.2%}\n\n"
                       f"**Interpretation:**\n"
                       f"• Your portfolio is {'more volatile' if analysis['beta'] > 1.1 else 'less volatile' if analysis['beta'] < 0.9 else 'similarly volatile'} than the market\n"
                       f"• {'Outperforming' if analysis['alpha'] > 0 else 'Underperforming'} the market by {abs(analysis['alpha']):.2%} annually\n"
                       f"• Risk-adjusted returns are {'excellent' if analysis['sharpe_ratio'] > 1.5 else 'good' if analysis['sharpe_ratio'] > 1.0 else 'fair' if analysis['sharpe_ratio'] > 0.5 else 'poor'}\n\n"
                       f"**🎯 Suggested Next Steps:**\n"
                       f"• `'show my factor exposures'` - Understand your style tilts\n"
                       f"• `'optimize my portfolio'` - Improve risk-adjusted returns\n"
                       f"• `'find tail risk hedges'` - Add crash protection\n"
                       f"• `'what if I...'` - Test portfolio changes",
                'show_charts': True
            }
        
        elif intent == "factor_analysis":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
            
            factors = self.engine.run_factor_analysis(context.last_analysis['portfolio_returns'])
            
            if "error" in factors:
                return {'text': f"❌ Factor analysis failed: {factors['error']}"}
            
            return {
                'text': f"📈 **Fama-French Factor Exposure**\n\n"
                       f"**Market Beta:** {factors['market_beta']:.2f}\n"
                       f"**Size Factor (SMB):** {factors['size_factor']:.2f}\n"
                       f"**Value Factor (HML):** {factors['value_factor']:.2f}\n"
                       f"**Alpha:** {factors['alpha']:.2%}\n"
                       f"**R-squared:** {factors['r_squared']:.2%}\n\n"
                       f"**Interpretation:**\n"
                       f"• {'Tilted toward small-cap' if factors['size_factor'] > 0.2 else 'Tilted toward large-cap' if factors['size_factor'] < -0.2 else 'Market-cap neutral'}\n"
                       f"• {'Value-oriented' if factors['value_factor'] > 0.2 else 'Growth-oriented' if factors['value_factor'] < -0.2 else 'Style-neutral'}\n"
                       f"• {factors['r_squared']:.0%} of returns explained by these factors"
            }
        
        elif intent == "optimization":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
            
            context.dialogue_state = 'awaiting_optimization_choice'
            suggestion = "Max Sharpe" if context.risk_tolerance == 'aggressive' else "Min Volatility"
            
            return {
                'text': f"🎯 **Portfolio Optimization**\n\n"
                       f"What's your optimization goal?\n\n"
                       f"**Available Options:**\n"
                       f"• **Max Sharpe Ratio** - Maximize risk-adjusted returns\n"
                       f"• **Min Volatility** - Minimize portfolio risk\n\n"
                       f"Based on your {context.risk_tolerance} risk profile, I'd suggest **{suggestion}**.\n\n"
                       f"Just reply with 'Max Sharpe' or 'Min Volatility'"
            }
        
        elif intent == "tail_risk_screening":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
            
            hedges = self.engine.screen_tail_risk_hedges(context.last_analysis['portfolio_returns'])
            
            if "error" in hedges:
                return {'text': f"❌ Screening failed: {hedges['error']}"}
            
            hedge_text = "\n".join([f"• **{h['ticker']}** - Correlation: {h['crisis_correlation']:.2f} ({h['hedge_quality']} hedge)" for h in hedges])
            
            return {
                'text': f"🛡️ **Tail Risk Hedge Recommendations**\n\n"
                       f"Based on performance during the last major market crisis:\n\n"
                       f"{hedge_text}\n\n"
                       f"**Note:** Negative correlations indicate better hedging properties during market stress."
            }
        
        elif intent == "correlation_screening":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
            
            uncorrelated = self.engine.screen_uncorrelated_assets(context.last_analysis['portfolio_returns'])
            
            if "error" in uncorrelated:
                return {'text': f"❌ Screening failed: {uncorrelated['error']}"}
            
            assets_text = "\n".join([f"• **{a['ticker']}** - Correlation: {a['correlation']:.2f} ({a['diversification_benefit']} diversification)" for a in uncorrelated])
            
            return {
                'text': f"📉 **Diversification Opportunities**\n\n"
                       f"Assets with low correlation to your portfolio:\n\n"
                       f"{assets_text}\n\n"
                       f"**Note:** Lower correlations provide better diversification benefits."
            }
        
        elif intent == "what_if_analysis":
            if not context.portfolio or not context.last_analysis:
                return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
            
            changes = self.parse_what_if_changes(message)
            if not changes:
                return {'text': "❌ I couldn't parse your what-if scenario. Try: 'what if I sell 10% AAPL and buy 10% MSFT?'"}
            
            # Validate changes
            for ticker, amount in changes.get('sell', {}).items():
                if ticker not in context.portfolio:
                    return {'text': f"❌ You don't hold {ticker} in your portfolio."}
                if context.portfolio[ticker] < amount:
                    return {'text': f"❌ You only hold {context.portfolio[ticker]:.1%} of {ticker}."}
            
            new_analysis, new_portfolio = self.engine.simulate_what_if(context.portfolio, changes)
            
            if not new_analysis:
                return {'text': "❌ Could not analyze the simulated portfolio."}
            
            current = context.last_analysis
            
            return {
                'text': f"🔮 **What-If Analysis Results**\n\n"
                       f"**Current vs Simulated Portfolio:**\n\n"
                       f"**Volatility:** {current['volatility']:.2%} → {new_analysis['volatility']:.2%} "
                       f"({new_analysis['volatility']-current['volatility']:+.2%})\n"
                       f"**Sharpe Ratio:** {current['sharpe_ratio']:.2f} → {new_analysis['sharpe_ratio']:.2f} "
                       f"({new_analysis['sharpe_ratio']-current['sharpe_ratio']:+.2f})\n"
                       f"**Beta:** {current['beta']:.2f} → {new_analysis['beta']:.2f} "
                       f"({new_analysis['beta']-current['beta']:+.2f})\n"
                       f"**Max Drawdown:** {current['max_drawdown']:.2%} → {new_analysis['max_drawdown']:.2%} "
                       f"({new_analysis['max_drawdown']-current['max_drawdown']:+.2%})\n\n"
                       f"**Changes:** {'; '.join([f'Sell {v:.1%} {k}' for k, v in changes.get('sell', {}).items()] + [f'Buy {v:.1%} {k}' for k, v in changes.get('buy', {}).items()])}"
            }
        
        elif intent == "show_portfolio":
            if not context.portfolio:
                return {'text': "You haven't set a portfolio yet. Try: '60% AAPL, 30% GOOGL, 10% BONDS'"}
            
            holdings = "\n".join([f"• **{ticker}:** {weight:.1%}" for ticker, weight in context.portfolio.items()])
            
            return {
                'text': f"📋 **Your Current Portfolio**\n\n{holdings}\n\n"
                       f"**Total Assets:** {len(context.portfolio)}\n"
                       f"**Risk Profile:** {context.risk_tolerance.title()}\n"
                       f"**Investment Goals:** {context.investment_goals}",
                'show_charts': True
            }
        
        elif intent == "set_profile":
            message_lower = message.lower()
            
            if "conservative" in message_lower:
                context.risk_tolerance = "conservative"
            elif "aggressive" in message_lower:
                context.risk_tolerance = "aggressive"
            elif "moderate" in message_lower:
                context.risk_tolerance = "moderate"
            else:
                return {'text': "❌ Please specify 'conservative', 'moderate', or 'aggressive' risk tolerance."}
            
            return {'text': f"✅ **Profile Updated**\n\nRisk tolerance set to **{context.risk_tolerance}**. This will influence my recommendations for optimization and asset allocation."}
        
        # Handle optimization choice
        elif context.dialogue_state == 'awaiting_optimization_choice':
            context.dialogue_state = 'start'
            
            if "sharpe" in message.lower():
                objective = "Max Sharpe"
            elif "volatility" in message.lower() or "vol" in message.lower():
                objective = "Min Volatility"
            else:
                return {'text': "❌ Please specify 'Max Sharpe' or 'Min Volatility'"}
            
            optimized_weights = self.engine.optimize_portfolio(
                context.last_analysis['returns'][context.last_analysis['portfolio_tickers']], 
                objective
            )
            
            if not optimized_weights:
                return {'text': "❌ Optimization failed. Please try again."}
            
            # Create comparison
            comparison_data = []
            for ticker in context.portfolio.keys():
                current_weight = context.portfolio.get(ticker, 0)
                optimized_weight = optimized_weights.get(ticker, 0)
                comparison_data.append({
                    'Asset': ticker,
                    'Current': f"{current_weight:.1%}",
                    'Optimized': f"{optimized_weight:.1%}",
                    'Change': f"{optimized_weight - current_weight:+.1%}"
                })
            
            comparison_text = "\n".join([f"• **{row['Asset']}:** {row['Current']} → {row['Optimized']} ({row['Change']})" for row in comparison_data])
            
            return {
                'text': f"🎯 **Optimization Results ({objective})**\n\n"
                       f"{comparison_text}\n\n"
                       f"**Constraints:** Max 40% per asset, fully invested\n"
                       f"**Objective:** {'Maximize risk-adjusted returns' if objective == 'Max Sharpe' else 'Minimize portfolio risk'}"
            }
        
        else:
            return {
                'text': "🤔 **I'm not sure how to help with that specific request.**\n\n"
                       "Here are some things you can try:\n\n"
                       "**🔥 Most Popular:**\n"
                       f"• `'60% AAPL, 30% GOOGL, 10% BONDS'` - Set portfolio\n"
                       f"• `'how risky is my portfolio?'` - Risk analysis\n"
                       f"• `'optimize my portfolio'` - Improve allocation\n\n"
                       "**🚀 Advanced Features:**\n"
                       f"• `'show my factor exposures'` - Factor analysis\n"
                       f"• `'find tail risk hedges'` - Crisis protection\n"
                       f"• `'what if I sell 10% AAPL and buy 10% QQQ?'` - Scenarios\n\n"
                       "**💡 Or simply ask:**\n"
                       f"• `'what can you do?'` - Full feature list\n"
                       f"• `'help'` - Detailed examples\n\n"
                       "**Remember:** I understand natural language, so just ask what you want to know!"
            }

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'context' not in st.session_state:
    st.session_state.context = UserContext(user_id="streamlit_user")
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = AdvancedChatbot()

# Header
st.markdown("""
<div class="chat-header">
    <h1>🤖 Advanced Portfolio AI Assistant</h1>
    <p>Comprehensive portfolio analysis, optimization, and strategic insights</p>
</div>
""", unsafe_allow_html=True)

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <div class="bot-bubble">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Quick actions buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("💼 Set Portfolio", key="quick_portfolio"):
        message = "60% AAPL, 30% GOOGL, 10% BONDS"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
        st.rerun()
    if st.button("🔮 What-If", key="quick_whatif"):
        message = "what if I sell 10% AAPL and buy 10% QQQ?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
        st.rerun()
with col2:
    if st.button("📊 Risk Analysis", key="quick_risk"):
        message = "how risky is my portfolio?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
        st.rerun()
    if st.button("🛡️ Find Hedges", key="quick_hedges"):
        message = "find tail risk hedges"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
        st.rerun()
with col3:
    if st.button("🎯 Optimize", key="quick_optimize"):
        message = "optimize my portfolio"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
        st.rerun()
    if st.button("📈 Factor Analysis", key="quick_factors"):
        message = "show my factor exposures"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
        st.rerun()
with col4:
    if st.button("🌟 Diversify", key="quick_diversify"):
        message = "find uncorrelated assets"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
        st.rerun()
    if st.button("❓ Full Help", key="quick_help"):
        message = "what can you do?"
        st.session_state.messages.append({"role": "user", "content": message})
        response = st.session_state.chatbot.generate_response(message, st.session_state.context)
        st.session_state.messages.append({"role": "assistant", "content": response['text']})
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
    response = st.session_state.chatbot.generate_response(user_input, st.session_state.context)
    
    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response['text']})
    
    # Clear input and rerun
    st.rerun()

# Advanced Charts Section
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

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666; font-size: 12px; border-top: 1px solid #eee; margin-top: 40px;">
    <p><strong>Portfolio AI Assistant</strong> • Advanced Financial Analysis • Built with Streamlit</p>
    <p>⚠️ This tool is for educational purposes only and does not constitute financial advice.</p>
</div>
""", unsafe_allow_html=True)
