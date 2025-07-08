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

# Import with error handling
try:
    import yfinance as yf
    import pandas_datareader.data as web
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.linear_model import LinearRegression
    from scipy.optimize import minimize
    FINANCIAL_LIBS_AVAILABLE = True
except ImportError as e:
    st.error(f"Missing financial libraries: {e}")
    FINANCIAL_LIBS_AVAILABLE = False

# Claude integration with error handling
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Hybrid Portfolio AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with error styling
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
    
    .claude-indicator { color: #9C27B0; }
    .rules-indicator { color: #4CAF50; }
    
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
    
    .error-message {
        background: #ffebee;
        border: 1px solid #f44336;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        color: #c62828;
    }
    
    .warning-message {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        color: #ef6c00;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DATABASE_FILE = "streamlit_portfolio_agent.db"
BENCHMARK_TICKER = 'SPY'

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
# SIMPLE DATABASE MANAGER
# ===================================================================
class SimpleDatabaseManager:
    def __init__(self):
        """Simple in-memory database using session state"""
        if 'user_contexts' not in st.session_state:
            st.session_state.user_contexts = {}
        if 'claude_cache' not in st.session_state:
            st.session_state.claude_cache = {}

    def get_user_context(self, user_id: str) -> UserContext:
        return st.session_state.user_contexts.get(user_id, UserContext(user_id=user_id))

    def save_user_context(self, context: UserContext):
        st.session_state.user_contexts[context.user_id] = context

    def cache_claude_response(self, user_id: str, query: str, response: str):
        cache_key = f"{user_id}:{hash(query.lower().strip())}"
        st.session_state.claude_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }

    def get_cached_response(self, user_id: str, query: str, max_age_hours: int = 24) -> Optional[str]:
        cache_key = f"{user_id}:{hash(query.lower().strip())}"
        cached = st.session_state.claude_cache.get(cache_key)
        if cached and (time.time() - cached['timestamp']) < (max_age_hours * 3600):
            return cached['response']
        return None

# ===================================================================
# CLAUDE AGENT LAYER
# ===================================================================
class ClaudeAgentLayer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self.enabled = False
        
        if CLAUDE_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.enabled = True
            except Exception as e:
                st.warning(f"Claude initialization failed: {e}")

    async def process_complex_query(self, message: str, context: UserContext, db: SimpleDatabaseManager) -> Dict[str, Any]:
        if not self.enabled:
            return {
                'text': "Advanced AI features not available. Please add ANTHROPIC_API_KEY to secrets.",
                'tokens_used': 0
            }

        # Check cache first
        cached = db.get_cached_response(context.user_id, message)
        if cached:
            return {'text': cached, 'tokens_used': 0, 'cached': True}

        try:
            system_prompt = f"""You are an expert portfolio analyst assistant. 
            Current user portfolio: {context.portfolio if context.portfolio else 'None set'}
            User risk tolerance: {context.risk_tolerance}
            Provide helpful, educational investment guidance."""
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": message}]
            )
            
            result = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Cache the response
            db.cache_claude_response(context.user_id, message, result)
            
            return {'text': result, 'tokens_used': tokens_used, 'cached': False}
            
        except Exception as e:
            return {
                'text': f"AI processing error: {str(e)}",
                'tokens_used': 0
            }

# ===================================================================
# SIMPLE PORTFOLIO ENGINE
# ===================================================================
class SimplePortfolioEngine:
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(_self, tickers: List[str], period: str = "1y"):
        if not FINANCIAL_LIBS_AVAILABLE:
            return None
        try:
            data = yf.download(tickers, period=period, progress=False)
            if len(tickers) == 1:
                return pd.DataFrame({tickers[0]: data['Close']})
            elif 'Close' in data.columns:
                return data['Close']
            else:
                return data.xs('Close', axis=1, level=1)
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            return None

    def analyze_portfolio(self, portfolio: Dict[str, float]):
        if not FINANCIAL_LIBS_AVAILABLE:
            return None
            
        tickers = list(portfolio.keys()) + [BENCHMARK_TICKER]
        data = self.fetch_data(tickers)
        
        if data is None or data.empty:
            return None
            
        # Simple returns calculation
        returns = data.pct_change().dropna()
        
        # Portfolio returns
        portfolio_tickers = [t for t in portfolio.keys() if t in returns.columns]
        if not portfolio_tickers:
            return None
            
        portfolio_weights = np.array([portfolio[t] for t in portfolio_tickers])
        portfolio_returns = returns[portfolio_tickers].dot(portfolio_weights)
        
        # Basic metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_returns': portfolio_returns,
            'portfolio_tickers': portfolio_tickers,
            'data': data
        }

# ===================================================================
# SMART NLU PROCESSOR
# ===================================================================
class SmartNLUProcessor:
    def should_use_claude(self, message: str, context: UserContext) -> bool:
        message_lower = message.lower().strip()
        
        # Simple patterns that should use rules
        simple_patterns = [
            r'\d+%\s*[A-Z]{2,5}',  # Portfolio input
            'show', 'view', 'display', 'portfolio',
            'help', 'hello', 'hi'
        ]
        
        if any(pattern in message_lower for pattern in simple_patterns):
            return False
        
        # Complex queries for Claude
        complex_indicators = [
            'why', 'how can', 'explain', 'strategy', 'advice',
            'market', 'economy', 'diversify', 'should i'
        ]
        
        return any(indicator in message_lower for indicator in complex_indicators)

    def parse(self, message: str, context: UserContext) -> Dict[str, Any]:
        use_claude = self.should_use_claude(message, context)
        
        if use_claude:
            return {"intent": "claude_query", "text": message, "routing": "claude"}
        
        # Simple rule-based routing
        message_lower = message.lower().strip()
        
        if '%' in message and any(char.isalpha() for char in message):
            intent = "portfolio_input"
        elif any(word in message_lower for word in ['help', 'what can you do']):
            intent = "help"
        elif any(word in message_lower for word in ['show', 'view', 'portfolio']):
            intent = "show_portfolio"
        elif any(word in message_lower for word in ['risk', 'risky']):
            intent = "risk_analysis"
        elif any(word in message_lower for word in ['hi', 'hello', 'hey']):
            intent = "greeting"
        else:
            intent = "general_query"
        
        return {"intent": intent, "text": message, "routing": "rules"}

# ===================================================================
# HYBRID CHATBOT
# ===================================================================
class HybridChatbot:
    def __init__(self, claude_api_key: Optional[str] = None):
        self.engine = SimplePortfolioEngine()
        self.nlu = SmartNLUProcessor()
        self.claude = ClaudeAgentLayer(claude_api_key)
        self.db = SimpleDatabaseManager()
        self.routing_stats = {"rules": 0, "claude": 0}

    def parse_portfolio(self, text: str) -> Optional[Dict[str, float]]:
        pattern = r'(\d+(?:\.\d*)?%)\s*([A-Z]{1,5})'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            return None
        
        portfolio = {}
        for percent, ticker in matches:
            portfolio[ticker.upper()] = float(percent.strip('%')) / 100
        
        total = sum(portfolio.values())
        if 0.8 <= total <= 1.2:  # Allow some flexibility
            portfolio = {k: v/total for k, v in portfolio.items()}
        
        return portfolio

    async def generate_response(self, message: str, context: UserContext):
        try:
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
                # Rule-based response
                response = self._handle_rule_based(nlu_result, context)
                response['routing'] = 'rules'
                response['tokens_used'] = 0
            
            # Save context
            self.db.save_user_context(context)
            return response
            
        except Exception as e:
            return {
                'text': f"Error processing message: {str(e)}",
                'routing': 'error',
                'tokens_used': 0
            }

    def _handle_rule_based(self, nlu_result: Dict[str, Any], context: UserContext) -> Dict[str, Any]:
        intent = nlu_result.get("intent")
        message = nlu_result.get("text")
        
        try:
            if intent == "greeting":
                return {
                    'text': "👋 **Welcome to your Hybrid Portfolio AI Assistant!**\n\n"
                           "I use smart routing: fast rules for simple tasks, AI for complex analysis.\n\n"
                           "**Quick Start:** `'60% AAPL, 30% GOOGL, 10% BONDS'`"
                }
            
            elif intent == "help":
                claude_status = "🟢 Enabled" if self.claude.enabled else "🔴 Disabled"
                return {
                    'text': f"🤖 **Hybrid Portfolio Assistant** (Claude: {claude_status})\n\n"
                           "**📊 Portfolio Setup:**\n"
                           "• `'60% AAPL, 30% GOOGL, 10% BONDS'` - Set portfolio\n"
                           "• `'show my portfolio'` - View holdings\n\n"
                           "**🔍 Analysis:**\n"
                           "• `'how risky is my portfolio?'` - Risk metrics\n\n"
                           "**🤖 AI Features** (if enabled):\n"
                           "• `'How can I diversify better?'` - Strategy advice\n"
                           "• `'What's the market outlook?'` - Market analysis"
                }
            
            elif intent == "portfolio_input":
                portfolio = self.parse_portfolio(message)
                if portfolio:
                    context.portfolio = portfolio
                    analysis = self.engine.analyze_portfolio(portfolio)
                    context.last_analysis = analysis
                    
                    if analysis:
                        return {
                            'text': f"✅ **Portfolio Set!** (Rules Engine)\n\n"
                                   f"**{len(portfolio)} assets:** {', '.join(portfolio.keys())}\n"
                                   f"**Volatility:** {analysis['volatility']:.2%}\n"
                                   f"**Sharpe Ratio:** {analysis['sharpe_ratio']:.2f}\n\n"
                                   f"Charts updated below! 📊",
                            'show_charts': True
                        }
                    else:
                        return {'text': "❌ Could not analyze portfolio. Check ticker symbols."}
                else:
                    return {'text': "❌ Could not parse portfolio. Try: '60% AAPL, 30% GOOGL, 10% BONDS'"}
            
            elif intent == "risk_analysis":
                if not context.portfolio or not context.last_analysis:
                    return {'text': "Please set your portfolio first using percentages like '60% AAPL, 40% GOOGL'"}
                
                analysis = context.last_analysis
                return {
                    'text': f"📊 **Risk Analysis** (Rules Engine)\n\n"
                           f"**Volatility:** {analysis['volatility']:.2%}\n"
                           f"**Sharpe Ratio:** {analysis['sharpe_ratio']:.2f}\n\n"
                           f"**🤖 Want deeper insights?** Ask: 'Why is my portfolio risky?'",
                    'show_charts': True
                }
            
            elif intent == "show_portfolio":
                if not context.portfolio:
                    return {'text': "You haven't set a portfolio yet. Try: '60% AAPL, 30% GOOGL, 10% BONDS'"}
                
                holdings = ", ".join([f"{t}: {w:.1%}" for t, w in context.portfolio.items()])
                return {
                    'text': f"📋 **Your Portfolio** (Rules Engine)\n\n{holdings}",
                    'show_charts': True
                }
            
            else:
                return {
                    'text': "🤔 **Not sure about that** (Rules Engine)\n\n"
                           "Try: `'help'` for commands or `'60% AAPL, 40% GOOGL'` to set portfolio"
                }
                
        except Exception as e:
            return {'text': f"Rule processing error: {str(e)}"}

    def get_routing_stats(self) -> Dict[str, Any]:
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

# ===================================================================
# MAIN APP
# ===================================================================

# Initialize session state with error handling
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'context' not in st.session_state:
    st.session_state.context = UserContext(user_id="streamlit_user")
if 'chatbot' not in st.session_state:
    # Try secrets first, then environment variable
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except:
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        except:
            pass
    st.session_state.chatbot = HybridChatbot(claude_api_key=api_key)
if 'total_tokens_used' not in st.session_state:
    st.session_state.total_tokens_used = 0

# Header with status
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

# Check for missing dependencies
if not FINANCIAL_LIBS_AVAILABLE:
    st.error("❌ **Missing Dependencies**: Some financial analysis features may not work properly.")

if not CLAUDE_AVAILABLE:
    st.warning("⚠️ **Claude Unavailable**: AI features disabled. Only rule-based responses available.")

# Chat container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages with safe access
    for i, message in enumerate(st.session_state.messages):
        try:
            if message.get('role') == 'user':
                content = message.get('content', f'Message {i}')
                st.markdown(f"""
                <div class="user-message">
                    <div class="user-bubble">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                routing = message.get('routing', 'unknown')
                tokens = message.get('tokens_used', 0)
                cached = message.get('cached', False)
                content = message.get('content', message.get('text', f'Response {i}'))
                
                routing_class = "claude-indicator" if routing == "claude" else "rules-indicator"
                routing_text = f"🤖 AI ({tokens} tokens)" if routing == "claude" else "⚡ Rules (0 tokens)"
                if cached:
                    routing_text += " [Cached]"
                
                st.markdown(f"""
                <div class="bot-message">
                    <div class="bot-bubble">
                        {content}
                        <div class="routing-indicator {routing_class}">{routing_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying message {i}: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Quick actions
st.markdown("**🚀 Quick Actions:**")
col1, col2, col3, col4 = st.columns(4)

def safe_button_handler(message: str, button_name: str):
    """Safe button handler with error handling"""
    try:
        st.session_state.messages.append({"role": "user", "content": message})
        response = asyncio.run(st.session_state.chatbot.generate_response(message, st.session_state.context))
        
        bot_message = {
            "role": "assistant", 
            "content": response.get('text', 'No response'),
            "routing": response.get('routing', 'unknown'),
            "tokens_used": response.get('tokens_used', 0),
            "cached": response.get('cached', False)
        }
        st.session_state.messages.append(bot_message)
        st.session_state.total_tokens_used += response.get('tokens_used', 0)
        st.rerun()
    except Exception as e:
        st.error(f"Error in {button_name}: {e}")

with col1:
    if st.button("💼 Set Portfolio", key="quick_portfolio"):
        safe_button_handler("60% AAPL, 30% GOOGL, 10% BONDS", "Set Portfolio")

with col2:
    if st.button("📊 Risk Analysis", key="quick_risk"):
        safe_button_handler("how risky is my portfolio?", "Risk Analysis")

with col3:
    if st.button("🤖 AI Strategy", key="quick_ai_strategy"):
        safe_button_handler("How can I improve my portfolio diversification strategy?", "AI Strategy")

with col4:
    if st.button("❓ Help", key="quick_help"):
        safe_button_handler("what can you do?", "Help")

# Input container
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Type your message...", key="user_input", label_visibility="collapsed")
with col2:
    send_button = st.button("Send", type="primary")

# Handle user input with error handling
if send_button and user_input:
    try:
        safe_button_handler(user_input, "User Input")
    except Exception as e:
        st.error(f"Error processing input: {e}")

# Simple charts section
if st.session_state.context.portfolio and st.session_state.context.last_analysis and FINANCIAL_LIBS_AVAILABLE:
    st.markdown("### 📊 Portfolio Visualization")
    
    try:
        # Simple pie chart
        fig_pie = px.pie(
            values=list(st.session_state.context.portfolio.values()),
            names=list(st.session_state.context.portfolio.keys()),
            title="Portfolio Composition"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Simple metrics
        analysis = st.session_state.context.last_analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volatility", f"{analysis['volatility']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{analysis['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Assets", len(st.session_state.context.portfolio))
            
    except Exception as e:
        st.error(f"Chart error: {e}")

# Footer with enhanced status
try:
    stats = st.session_state.chatbot.get_routing_stats()
    claude_status = "🟢 Enabled" if st.session_state.chatbot.claude.enabled else "🔴 Disabled"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; color: #666; font-size: 12px; border-top: 1px solid #eee; margin-top: 40px;">
        <p><strong>Hybrid Portfolio AI Assistant</strong> • 
        Routing: {stats['rules_percentage']:.0f}% Rules, {stats['claude_percentage']:.0f}% AI • 
        Claude: {claude_status} • 
        Tokens: {st.session_state.total_tokens_used}</p>
        <p>⚠️ Educational purposes only - not financial advice</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.markdown("**Hybrid Portfolio AI Assistant** - Educational tool")
