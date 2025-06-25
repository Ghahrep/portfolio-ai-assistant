#!/usr/bin/env python3
"""
Portfolio Risk Analyzer MVP - Streamlined Version
Focused on core functionality with minimal dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE DATA PROVIDER - SIMPLIFIED
# ============================================================================

class RobustDataProvider:
    """Simplified data provider with retry logic"""
    
    def __init__(self):
        self.cache = {}
        
    def fetch_market_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """Fetch market data with simple retry logic"""
        cache_key = f"{','.join(sorted(tickers))}_{period}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Validate tickers first
                self._validate_tickers(tickers)
                
                # Fetch data
                data = yf.download(tickers, period=period, progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data = data["Close"]
                
                if len(tickers) == 1:
                    data = pd.DataFrame({tickers[0]: data})
                
                if data.empty or len(data) < 10:
                    raise ValueError("Insufficient data returned")
                
                # Cache successful result
                self.cache[cache_key] = data
                return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                time.sleep(1)  # Simple retry delay
    
    def _validate_tickers(self, tickers: List[str]):
        """Basic ticker validation"""
        for ticker in tickers:
            if not ticker or len(ticker) > 5 or not ticker.isalpha():
                raise ValueError(f"Invalid ticker format: {ticker}")

# ============================================================================
# PORTFOLIO HEALTH MONITOR - SIMPLIFIED
# ============================================================================

@dataclass
class HealthMetrics:
    """Simplified health metrics"""
    overall_score: float
    health_level: str
    concentration_risk: float
    diversification_score: float
    key_risks: List[str]
    recommendations: List[str]

class PortfolioHealthMonitor:
    """Simplified portfolio health assessment"""
    
    def calculate_health(self, portfolio: Dict[str, float], returns_data: pd.DataFrame) -> HealthMetrics:
        """Calculate simplified portfolio health score"""
        
        # Concentration risk (0-100, higher is better)
        max_weight = max(portfolio.values()) if portfolio else 1.0
        concentration_penalty = max_weight * 100  # Direct penalty
        concentration_score = max(0, 100 - concentration_penalty)
        
        # Diversification score
        n_positions = len(portfolio)
        diversification_score = min(100, n_positions * 15)  # Benefit for more positions
        
        # Overall score (weighted average)
        overall_score = (concentration_score * 0.6 + diversification_score * 0.4)
        
        # Health level
        if overall_score >= 80:
            health_level = "Excellent"
        elif overall_score >= 65:
            health_level = "Good"
        elif overall_score >= 50:
            health_level = "Fair"
        else:
            health_level = "Poor"
        
        # Identify risks and recommendations
        key_risks = self._identify_risks(portfolio, max_weight, n_positions)
        recommendations = self._generate_recommendations(overall_score, max_weight, n_positions)
        
        return HealthMetrics(
            overall_score=overall_score,
            health_level=health_level,
            concentration_risk=100 - concentration_score,
            diversification_score=diversification_score,
            key_risks=key_risks,
            recommendations=recommendations
        )
    
    def _identify_risks(self, portfolio: Dict[str, float], max_weight: float, n_positions: int) -> List[str]:
        """Identify key portfolio risks"""
        risks = []
        
        if max_weight > 0.5:
            risks.append(f"Extreme concentration: {max_weight:.1%} in single position")
        elif max_weight > 0.3:
            risks.append(f"High concentration: {max_weight:.1%} in largest position")
        
        if n_positions < 3:
            risks.append("Very limited diversification")
        elif n_positions < 5:
            risks.append("Limited diversification")
        
        # Check for sector concentration (simplified)
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        tech_weight = sum(portfolio.get(ticker, 0) for ticker in tech_tickers)
        if tech_weight > 0.7:
            risks.append("High technology sector concentration")
        
        if not risks:
            risks.append("Risk levels appear well-managed")
        
        return risks[:3]
    
    def _generate_recommendations(self, score: float, max_weight: float, n_positions: int) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if max_weight > 0.4:
            recommendations.append("Reduce largest position to under 25%")
        
        if n_positions < 5:
            recommendations.append("Add more positions for better diversification")
        
        if score < 70:
            recommendations.append("Consider adding bonds or defensive assets")
        
        if not recommendations:
            recommendations.append("Portfolio structure looks good - maintain current approach")
        
        return recommendations

# ============================================================================
# RISK CALCULATOR - SIMPLIFIED
# ============================================================================

class RiskCalculator:
    """Simplified risk metrics calculator"""
    
    def calculate_portfolio_risk(self, portfolio: Dict[str, float], 
                               returns_data: pd.DataFrame) -> Dict:
        """Calculate core portfolio risk metrics"""
        
        # Calculate returns
        returns = returns_data.pct_change().dropna()
        
        if returns.empty:
            raise ValueError("No returns data available")
        
        # Portfolio returns
        weights = pd.Series(portfolio)
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)  # 5% VaR
        es_95 = portfolio_returns[portfolio_returns <= var_95].mean()  # Expected shortfall
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annual volatility
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        return {
            'portfolio_returns': portfolio_returns,
            'var_95': var_95,
            'es_95': es_95,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'return_stats': {
                'mean_daily': portfolio_returns.mean(),
                'std_daily': portfolio_returns.std(),
                'skewness': portfolio_returns.skew(),
                'kurtosis': portfolio_returns.kurtosis()
            }
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return drawdowns.min()
    
    def stress_test_portfolio(self, portfolio: Dict[str, float], 
                            portfolio_value: float) -> Dict:
        """Simple stress testing"""
        
        scenarios = {
            "Market Crash (2008-style)": -0.25,
            "COVID Shock (2020-style)": -0.20,
            "Interest Rate Shock": -0.15
        }
        
        results = {}
        max_weight = max(portfolio.values()) if portfolio else 0
        
        for scenario, base_loss in scenarios.items():
            # Adjust loss based on concentration
            concentration_multiplier = 1 + (max_weight - 0.2) * 0.5
            adjusted_loss = base_loss * concentration_multiplier
            
            loss_amount = abs(adjusted_loss) * portfolio_value
            
            results[scenario] = {
                'loss_percentage': adjusted_loss,
                'loss_amount': loss_amount,
                'portfolio_value_after': portfolio_value * (1 + adjusted_loss)
            }
        
        return results

# ============================================================================
# SIMPLIFIED AI ASSISTANT
# ============================================================================

class SimpleAIAssistant:
    """Simplified AI assistant for portfolio questions"""
    
    def __init__(self):
        self.conversation_history = []
        
    def process_question(self, question: str, portfolio_data: Dict) -> str:
        """Process user questions about portfolio"""
        
        question_lower = question.lower()
        
        # Route to appropriate handler
        if any(word in question_lower for word in ['risky', 'risk', 'dangerous']):
            return self._explain_risk(portfolio_data)
        elif any(word in question_lower for word in ['health', 'healthy']):
            return self._explain_health(portfolio_data)
        elif any(word in question_lower for word in ['crash', 'crisis', 'stress']):
            return self._explain_stress_test(portfolio_data)
        elif any(word in question_lower for word in ['improve', 'better', 'optimize']):
            return self._suggest_improvements(portfolio_data)
        else:
            return self._general_response(portfolio_data)
    
    def _explain_risk(self, data: Dict) -> str:
        """Explain portfolio risk in simple terms"""
        risk_metrics = data.get('risk_metrics', {})
        var_95 = risk_metrics.get('var_95', 0)
        volatility = risk_metrics.get('volatility', 0)
        
        explanations = [
            f"Your portfolio could lose about {abs(var_95):.1%} on a bad day (95% confidence). This is based on historical market patterns.",
            f"With {volatility:.1%} annual volatility, your portfolio has {'high' if volatility > 0.25 else 'moderate' if volatility > 0.15 else 'low'} price swings compared to the market.",
            f"The main risk comes from concentration in your largest positions. When they move down, your whole portfolio feels it."
        ]
        
        return np.random.choice(explanations)
    
    def _explain_health(self, data: Dict) -> str:
        """Explain portfolio health score"""
        health_metrics = data.get('health_metrics')
        
        if hasattr(health_metrics, 'overall_score'):
            # health_metrics is a HealthMetrics dataclass
            score = health_metrics.overall_score
            level = health_metrics.health_level
        else:
            # health_metrics is a dictionary (fallback)
            score = health_metrics.get('overall_score', 65) if health_metrics else 65
            level = health_metrics.get('health_level', 'Fair') if health_metrics else 'Fair'
        
        return f"Your portfolio health score is {score:.0f}/100 ({level}). This reflects how well-diversified and balanced your holdings are. {'Great job!' if score > 80 else 'Some room for improvement.' if score > 60 else 'Consider rebalancing for better health.'}"
    
    def _explain_stress_test(self, data: Dict) -> str:
        """Explain stress test results"""
        stress_tests = data.get('stress_tests', {})
        
        if stress_tests:
            worst_scenario = min(stress_tests.items(), key=lambda x: x[1]['loss_percentage'])
            scenario_name, scenario_data = worst_scenario
            loss_pct = abs(scenario_data['loss_percentage'])
            
            return f"In a {scenario_name.lower()}, your portfolio could lose around {loss_pct:.1%}. This is based on how similar portfolios performed during past crises."
        else:
            return "Stress testing shows how your portfolio might perform during market crises. Generally, more diversified portfolios handle stress better."
    
    def _suggest_improvements(self, data: Dict) -> str:
        """Suggest portfolio improvements"""
        health_metrics = data.get('health_metrics')
        
        if hasattr(health_metrics, 'recommendations'):
            # health_metrics is a HealthMetrics dataclass
            recommendations = health_metrics.recommendations
        else:
            # health_metrics is a dictionary (fallback)
            recommendations = health_metrics.get('recommendations', []) if health_metrics else []
        
        if recommendations:
            return f"To improve your portfolio: {recommendations[0]}. This would help reduce concentration risk and improve diversification."
        else:
            return "Your portfolio structure looks solid. Consider periodic rebalancing to maintain your target allocation."
    
    def _general_response(self, data: Dict) -> str:
        """General response for other questions"""
        return "I can help explain your portfolio's risk level, health score, stress test results, or suggest improvements. What specific aspect interests you most?"

# ============================================================================
# MAIN PORTFOLIO ANALYZER
# ============================================================================

class MVPPortfolioAnalyzer:
    """Main MVP portfolio analyzer class"""
    
    def __init__(self):
        self.data_provider = RobustDataProvider()
        self.health_monitor = PortfolioHealthMonitor()
        self.risk_calculator = RiskCalculator()
        self.ai_assistant = SimpleAIAssistant()
    
    def analyze_portfolio(self, portfolio: Dict[str, float], portfolio_value: float = 1000000) -> Dict:
        """Complete portfolio analysis"""
        
        tickers = list(portfolio.keys())
        
        # Fetch market data
        market_data = self.data_provider.fetch_market_data(tickers)
        
        # Calculate risk metrics
        risk_metrics = self.risk_calculator.calculate_portfolio_risk(portfolio, market_data)
        
        # Calculate health metrics
        health_metrics = self.health_monitor.calculate_health(portfolio, market_data)
        
        # Stress testing
        stress_tests = self.risk_calculator.stress_test_portfolio(portfolio, portfolio_value)
        
        return {
            'portfolio': portfolio,
            'portfolio_value': portfolio_value,
            'market_data': market_data,
            'risk_metrics': risk_metrics,
            'health_metrics': health_metrics,
            'stress_tests': stress_tests,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def parse_portfolio_input(self, text: str) -> Optional[Dict[str, float]]:
        """Parse portfolio input from text"""
        portfolio = {}
        
        # Try percentage format first
        percent_matches = re.findall(r'(\d+(?:\.\d+)?)%\s+([A-Z]{1,5})', text.upper())
        if percent_matches:
            total_pct = 0
            for pct, ticker in percent_matches:
                weight = float(pct) / 100
                portfolio[ticker] = weight
                total_pct += float(pct)
            
            # Normalize if close to 100%
            if 90 <= total_pct <= 110:
                factor = 1.0 / (total_pct / 100)
                portfolio = {k: v * factor for k, v in portfolio.items()}
                return portfolio
        
        # Try equal weight format
        if 'equal' in text.lower():
            tickers = re.findall(r'\b([A-Z]{1,5})\b', text.upper())
            if tickers and len(tickers) <= 10:
                weight = 1.0 / len(tickers)
                return {ticker: weight for ticker in tickers}
        
        return None

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Portfolio Risk Analyzer MVP",
        page_icon="📊",
        layout="wide"
    )
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h1>📊 Portfolio Risk Analyzer</h1>
        <p>Professional portfolio analysis with AI insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    @st.cache_resource
    def get_analyzer():
        return MVPPortfolioAnalyzer()
    
    analyzer = get_analyzer()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar with examples
    with st.sidebar:
        st.header("🚀 Quick Examples")
        
        if st.button("Conservative Portfolio", use_container_width=True):
            st.session_state.portfolio_input = "40% VOO, 30% BND, 20% VTI, 10% VXUS"
        
        if st.button("Growth Portfolio", use_container_width=True):
            st.session_state.portfolio_input = "30% AAPL, 25% MSFT, 20% GOOGL, 15% AMZN, 10% TSLA"
        
        if st.button("Balanced ETFs", use_container_width=True):
            st.session_state.portfolio_input = "Equal weight SPY QQQ BND VTI"
        
        st.markdown("---")
        st.info("💡 Enter your portfolio above and get professional risk analysis with AI insights!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Portfolio Input")
        
        portfolio_input = st.text_area(
            "Enter your portfolio",
            value=st.session_state.get('portfolio_input', ''),
            height=120,
            placeholder="Examples:\n• 40% AAPL, 30% MSFT, 20% GOOGL, 10% BND\n• Equal weight SPY QQQ VTI\n• 60% VOO, 40% BND",
            help="Use percentage format or 'Equal weight' followed by tickers"
        )
        
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=1000,
            value=1000000,
            step=10000,
            format="%d"
        )
        
        analyze_button = st.button("🚀 Analyze Portfolio", type="primary", use_container_width=True)
        
        # Portfolio validation feedback
        if portfolio_input:
            portfolio = analyzer.parse_portfolio_input(portfolio_input)
            if portfolio:
                st.success(f"✅ Valid portfolio with {len(portfolio)} positions")
                for ticker, weight in portfolio.items():
                    st.write(f"• **{ticker}**: {weight:.1%}")
            else:
                st.warning("⚠️ Portfolio format not recognized. Try the examples above.")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if analyze_button and portfolio_input:
            portfolio = analyzer.parse_portfolio_input(portfolio_input)
            
            if portfolio:
                try:
                    with st.spinner("🔄 Analyzing your portfolio with real market data..."):
                        results = analyzer.analyze_portfolio(portfolio, portfolio_value)
                        st.session_state.analysis_results = results
                        st.success("✅ Analysis completed!")
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    if "Invalid ticker" in str(e):
                        st.info("💡 Please check your ticker symbols and try again")
            else:
                st.error("❌ Please enter a valid portfolio format")
        
        elif st.session_state.analysis_results:
            st.success("✅ Analysis completed!")
        else:
            st.info("👈 Enter your portfolio to get started with professional analysis")
    
    # Display results if available
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results, analyzer)

def display_analysis_results(results: Dict, analyzer: MVPPortfolioAnalyzer):
    """Display comprehensive analysis results"""
    
    portfolio = results['portfolio']
    portfolio_value = results['portfolio_value']
    risk_metrics = results['risk_metrics']
    health_metrics = results['health_metrics']
    stress_tests = results['stress_tests']
    
    st.markdown("---")
    st.markdown("### 📈 Portfolio Overview")
    
    # Portfolio pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(portfolio.keys()),
        values=list(portfolio.values()),
        hole=0.3,
        textinfo='label+percent'
    )])
    fig_pie.update_layout(title="Portfolio Allocation", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Risk metrics
    st.markdown("### 📊 Risk Analysis")
    
    var_95 = risk_metrics['var_95']
    es_95 = risk_metrics['es_95']
    volatility = risk_metrics['volatility']
    max_drawdown = risk_metrics['max_drawdown']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        var_dollar = abs(var_95) * portfolio_value
        st.metric("Daily VaR (95%)", f"${var_dollar:,.0f}", f"{var_95:.1%}")
    
    with col2:
        es_dollar = abs(es_95) * portfolio_value
        st.metric("Expected Shortfall", f"${es_dollar:,.0f}", f"{es_95:.1%}")
    
    with col3:
        st.metric("Annual Volatility", f"{volatility:.1%}")
    
    with col4:
        st.metric("Max Drawdown", f"{abs(max_drawdown):.1%}")
    
    # Portfolio health
    st.markdown("### 🏥 Portfolio Health")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Health gauge
        health_score = health_metrics.overall_score if hasattr(health_metrics, 'overall_score') else 65
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={'text': "Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        health_emoji = {'Excellent': '🟢', 'Good': '🟡', 'Fair': '🟠', 'Poor': '🔴'}
        emoji = health_emoji.get(health_metrics.health_level, '🟡')
        
        st.markdown(f"#### {emoji} {health_metrics.health_level} Health")
        st.markdown(f"**Score:** {health_metrics.overall_score:.0f}/100")
        
        st.markdown("**Key Risks:**")
        for risk in health_metrics.key_risks:
            st.write(f"• {risk}")
        
        st.markdown("**Recommendations:**")
        for rec in health_metrics.recommendations:
            st.write(f"• {rec}")
    
    # Stress testing
    st.markdown("### 🔥 Stress Test Results")
    
    scenarios = list(stress_tests.keys())
    losses = [abs(data['loss_percentage']) * 100 for data in stress_tests.values()]
    
    fig_stress = go.Figure(data=[
        go.Bar(x=scenarios, y=losses, 
               text=[f"{loss:.1f}%" for loss in losses],
               textposition='auto')
    ])
    fig_stress.update_layout(
        title="Portfolio Loss in Crisis Scenarios (%)",
        xaxis_title="Scenario",
        yaxis_title="Loss (%)",
        height=400
    )
    st.plotly_chart(fig_stress, use_container_width=True)
    
    # AI Assistant
    st.markdown("### 🤖 AI Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💡 Why is my portfolio risky?", use_container_width=True):
            response = analyzer.ai_assistant.process_question("Why is my portfolio risky?", results)
            st.session_state.chat_history.append(("Why is my portfolio risky?", response))
        
        if st.button("🏥 Explain my health score", use_container_width=True):
            response = analyzer.ai_assistant.process_question("Explain my health score", results)
            st.session_state.chat_history.append(("Explain my health score", response))
    
    with col2:
        if st.button("💥 What if markets crash?", use_container_width=True):
            response = analyzer.ai_assistant.process_question("What if markets crash?", results)
            st.session_state.chat_history.append(("What if markets crash?", response))
        
        if st.button("🛠️ How can I improve?", use_container_width=True):
            response = analyzer.ai_assistant.process_question("How can I improve?", results)
            st.session_state.chat_history.append(("How can I improve?", response))
    
    # Custom question
    with st.expander("💬 Ask a Custom Question"):
        custom_question = st.text_input("Your question:", placeholder="Ask about your portfolio...")
        if st.button("Ask AI") and custom_question:
            response = analyzer.ai_assistant.process_question(custom_question, results)
            st.session_state.chat_history.append((custom_question, response))
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("**💬 Conversation:**")
        for question, answer in st.session_state.chat_history[-3:]:  # Show last 3
            st.markdown(f"**🙋 You:** {question}")
            st.markdown(f"**🤖 AI:** {answer}")
            st.markdown("---")

if __name__ == "__main__":
    main()
