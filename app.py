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
    correlation_score: float 
    regime_fitness_score: float
    factor_balance_score: float
    key_risks: List[str]
    recommendations: List[str]

class PortfolioHealthMonitor:
    """Simplified portfolio health assessment"""
    
    def calculate_health(self, portfolio: Dict[str, float], returns_data: pd.DataFrame) -> HealthMetrics:
        """Calculate comprehensive portfolio health score with all 5 components"""
        
        # Concentration risk (0-100, higher is better) - KEEP EXISTING
        max_weight = max(portfolio.values()) if portfolio else 1.0
        concentration_penalty = max_weight * 100  # Direct penalty
        concentration_score = max(0, 100 - concentration_penalty)
        
        # Diversification score - KEEP EXISTING  
        n_positions = len(portfolio)
        diversification_score = min(100, n_positions * 15)  # Benefit for more positions
        
        # Enhanced correlation assessment - KEEP EXISTING
        correlation_score = self.assess_correlation_health(portfolio, returns_data)
        
        # Regime fitness assessment - KEEP EXISTING
        regime_fitness_score = self.assess_regime_fitness(portfolio)
        
        # NEW: Factor balance assessment
        factor_balance_score = self.assess_factor_balance(portfolio)
        
        # UPDATED: Overall score with all 5 components (balanced weighting)
        overall_score = (
            concentration_score * 0.25 +     # Reduced slightly
            diversification_score * 0.20 +   # Reduced slightly  
            correlation_score * 0.20 +       # Reduced slightly
            regime_fitness_score * 0.20 +    # Keep existing
            factor_balance_score * 0.15      # NEW: 15% weight
        )
        
        # Health level - KEEP EXISTING
        if overall_score >= 80:
            health_level = "Excellent"
        elif overall_score >= 65:
            health_level = "Good"
        elif overall_score >= 50:
            health_level = "Fair"
        else:
            health_level = "Poor"
        
        # Enhanced: Include factor balance in risk assessment
        key_risks = self._identify_risks_comprehensive(portfolio, max_weight, n_positions, 
                                                     correlation_score, regime_fitness_score,
                                                     factor_balance_score)
        recommendations = self._generate_recommendations_comprehensive(overall_score, max_weight, 
                                                                     n_positions, correlation_score, 
                                                                     regime_fitness_score,
                                                                     factor_balance_score)
        
        return HealthMetrics(
            overall_score=overall_score,
            health_level=health_level,
            concentration_risk=100 - concentration_score,
            diversification_score=diversification_score,
            correlation_score=correlation_score,
            regime_fitness_score=regime_fitness_score,
            factor_balance_score=factor_balance_score,  # NEW: Add this
            key_risks=key_risks,
            recommendations=recommendations
        )
    
    def _identify_risks_comprehensive(self, portfolio: Dict[str, float], max_weight: float, 
                                    n_positions: int, correlation_score: float, 
                                    regime_fitness_score: float, factor_balance_score: float) -> List[str]:
        """Comprehensive risk identification including factor balance"""
        risks = []
        
        # Existing concentration risks
        if max_weight > 0.5:
            risks.append(f"Extreme concentration: {max_weight:.1%} in single position")
        elif max_weight > 0.3:
            risks.append(f"High concentration: {max_weight:.1%} in largest position")
        
        # Existing diversification risks
        if n_positions < 3:
            risks.append("Very limited diversification")
        elif n_positions < 5:
            risks.append("Limited diversification")
        
        # Correlation-based risks - KEEP EXISTING
        if correlation_score < 40:
            risks.append("High correlation between holdings reduces diversification benefits")
        elif correlation_score < 60:
            risks.append("Moderate correlation detected - consider uncorrelated assets")
        
        # Regime fitness risks - KEEP EXISTING
        if regime_fitness_score < 50:
            risks.append("Portfolio structure may not be optimal for current market conditions")
        elif regime_fitness_score < 70:
            risks.append("Portfolio could benefit from better market regime alignment")
        
        # NEW: Factor balance risks
        if factor_balance_score < 40:
            risks.append("Poor factor balance - concentrated investment style exposure")
        elif factor_balance_score < 60:
            risks.append("Factor balance could be improved with more diversification")
        
        # Existing sector concentration check
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        tech_weight = sum(portfolio.get(ticker, 0) for ticker in tech_tickers)
        if tech_weight > 0.7:
            risks.append("High technology sector concentration")
        
        if not risks:
            risks.append("Risk levels appear well-managed")
        
        return risks[:3]  # Keep max 3 risks
    
    def _generate_recommendations_comprehensive(self, score: float, max_weight: float, 
                                              n_positions: int, correlation_score: float,
                                              regime_fitness_score: float, 
                                              factor_balance_score: float) -> List[str]:
        """Comprehensive recommendations including factor balance insights"""
        recommendations = []
        
        # Existing concentration recommendations
        if max_weight > 0.4:
            recommendations.append("Reduce largest position to under 25%")
        
        # Existing diversification recommendations
        if n_positions < 5:
            recommendations.append("Add more positions for better diversification")
        
        # Correlation-based recommendations - KEEP EXISTING
        if correlation_score < 50:
            recommendations.append("Add assets from different sectors or asset classes to reduce correlation")
        elif correlation_score < 70:
            recommendations.append("Consider adding bonds or international assets for better correlation balance")
        
        # Regime fitness recommendations - KEEP EXISTING
        if regime_fitness_score < 60:
            if max_weight > 0.6:
                recommendations.append("High concentration reduces regime adaptability - consider rebalancing")
            elif n_positions < 4:
                recommendations.append("Increase diversification to improve market regime fitness")
            else:
                recommendations.append("Consider adjusting portfolio structure for current market conditions")
        
        # NEW: Factor balance recommendations
        if factor_balance_score < 50:
            if max_weight > 0.4:
                recommendations.append("Reduce concentration to improve factor balance and style diversification")
            elif n_positions < 4:
                recommendations.append("Add holdings across different investment styles (growth/value, small/large cap)")
            else:
                recommendations.append("Consider adding assets with different factor exposures")
        elif factor_balance_score < 70:
            recommendations.append("Good factor balance - consider minor adjustments for style diversification")
        
        # Existing general recommendations
        if score < 70:
            recommendations.append("Consider adding defensive assets for stability")
        
        if not recommendations:
            recommendations.append("Portfolio structure looks excellent - maintain current approach")
        
        return recommendations[:3]  # Keep max 3 recommendations
    
    def assess_correlation_health(self, portfolio: Dict[str, float], 
                                 market_data: pd.DataFrame = None) -> float:
        """Assess correlation health of portfolio - Enhanced version"""
        try:
            # If we have market data, calculate correlation matrix
            if market_data is not None and len(market_data.columns) > 1:
                # Calculate returns for correlation analysis
                returns = market_data.pct_change().dropna()
                
                if len(returns) > 10:  # Need sufficient data
                    correlation_matrix = returns.corr().values
                else:
                    return 60  # Neutral score if insufficient data
            else:
                return 60  # Neutral score if no market data
            
            # Calculate average correlation (excluding diagonal)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            avg_correlation = np.mean(correlation_matrix[mask])
            
            # Optimal correlation is around 0.3-0.5 (your original logic)
            if 0.3 <= avg_correlation <= 0.5:
                score = 100
            elif avg_correlation < 0.3:
                score = 70 + (avg_correlation * 100)  # Low correlation is still good
            else:
                score = max(0, 100 - (avg_correlation - 0.5) * 200)  # Penalize high correlation
            
            return min(100, max(0, score))
            
        except Exception:
            return 60  # Neutral score on error (your original fallback)
        
    def _identify_risks_enhanced(self, portfolio: Dict[str, float], max_weight: float, 
                               n_positions: int, correlation_score: float) -> List[str]:
        """Enhanced risk identification including correlation analysis"""
        risks = []
        
        # Existing concentration risks
        if max_weight > 0.5:
            risks.append(f"Extreme concentration: {max_weight:.1%} in single position")
        elif max_weight > 0.3:
            risks.append(f"High concentration: {max_weight:.1%} in largest position")
        
        # Existing diversification risks
        if n_positions < 3:
            risks.append("Very limited diversification")
        elif n_positions < 5:
            risks.append("Limited diversification")
        
        # NEW: Correlation-based risks
        if correlation_score < 40:
            risks.append("High correlation between holdings reduces diversification benefits")
        elif correlation_score < 60:
            risks.append("Moderate correlation detected - consider uncorrelated assets")
        
        # Existing sector concentration check
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        tech_weight = sum(portfolio.get(ticker, 0) for ticker in tech_tickers)
        if tech_weight > 0.7:
            risks.append("High technology sector concentration")
        
        if not risks:
            risks.append("Risk levels appear well-managed")
        
        return risks[:3]  # Keep max 3 risks
    
    def _generate_recommendations_enhanced(self, score: float, max_weight: float, 
                                         n_positions: int, correlation_score: float) -> List[str]:
        """Enhanced recommendations including correlation insights"""
        recommendations = []
        
        # Existing concentration recommendations
        if max_weight > 0.4:
            recommendations.append("Reduce largest position to under 25%")
        
        # Existing diversification recommendations
        if n_positions < 5:
            recommendations.append("Add more positions for better diversification")
        
        # NEW: Correlation-based recommendations
        if correlation_score < 50:
            recommendations.append("Add assets from different sectors or asset classes to reduce correlation")
        elif correlation_score < 70:
            recommendations.append("Consider adding bonds or international assets for better correlation balance")
        
        # Existing general recommendations
        if score < 70:
            recommendations.append("Consider adding defensive assets for stability")
        
        if not recommendations:
            recommendations.append("Portfolio structure looks good - maintain current approach")
        
        return recommendations[:3]  # Keep max 3 recommendations
    
    def assess_regime_fitness(self, portfolio: Dict[str, float]) -> float:
        """Assess how well portfolio fits current market regime - Enhanced version"""
        
        # Your original logic (simplified and effective)
        portfolio_size = len(portfolio)
        max_weight = max(portfolio.values()) if portfolio else 0
        
        # General heuristics for regime fitness (your original logic)
        if max_weight < 0.4 and portfolio_size >= 4:
            return 85  # Well-diversified portfolio
        elif max_weight > 0.6:
            return 40  # High concentration risk
        else:
            return 70  # Moderate fitness
        
    def assess_factor_balance(self, portfolio: Dict[str, float]) -> float:
        """Assess factor balance (simplified) - Your original logic"""
        
        # Simplified factor balance assessment
        # In practice, this would analyze actual factor exposures
        
        num_positions = len(portfolio)
        max_weight = max(portfolio.values()) if portfolio else 0
        
        # More positions generally means better factor balance (your original logic)
        position_score = min(40, num_positions * 8)
        
        # Lower concentration generally means better factor balance (your original logic)
        concentration_score = max(0, 60 - (max_weight - 0.2) * 200)
        
        return min(100, position_score + concentration_score)
    


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
        
        try:
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
        except Exception as e:
            # Fallback responses if there's any processing error
            if 'risk' in question_lower:
                return "Your portfolio shows some concentration that could increase risk during market volatility. Consider diversifying across different sectors and asset types."
            elif 'health' in question_lower:
                return "Your portfolio health reflects how well-diversified your holdings are. The score considers concentration risk and overall balance."
            elif 'crash' in question_lower or 'stress' in question_lower:
                return "During market stress, portfolio losses depend on concentration and diversification. More concentrated portfolios typically see larger losses."
            elif 'improve' in question_lower:
                return "To improve your portfolio, consider reducing concentration in large positions and adding diversification across sectors or asset classes."
            else:
                return "I can help explain your portfolio's risk characteristics, health score, or suggest improvements. What specific aspect interests you?"
    
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
        try:
            health_metrics = data.get('health_metrics')
            
            # Handle both dataclass and dictionary formats safely
            if health_metrics and hasattr(health_metrics, 'overall_score'):
                # health_metrics is a HealthMetrics dataclass
                score = health_metrics.overall_score
                level = health_metrics.health_level
            elif health_metrics and isinstance(health_metrics, dict):
                # health_metrics is a dictionary
                score = health_metrics.get('overall_score', 65)
                level = health_metrics.get('health_level', 'Fair')
            else:
                # Fallback values
                score = 65
                level = 'Fair'
            
            return f"Your portfolio health score is {score:.0f}/100 ({level}). This reflects how well-diversified and balanced your holdings are. {'Great job!' if score > 80 else 'Some room for improvement.' if score > 60 else 'Consider rebalancing for better health.'}"
        
        except Exception as e:
            return "Your portfolio health score reflects how well-diversified your holdings are. A higher score indicates better balance and lower concentration risk. Consider diversifying if you have large positions in single stocks."
    
    def _explain_stress_test(self, data: Dict) -> str:
        """Explain stress test results"""
        try:
            stress_tests = data.get('stress_tests', {})
            
            if stress_tests and isinstance(stress_tests, dict):
                # Find the scenario with the highest loss
                max_loss = 0
                worst_scenario_name = "market stress"
                
                for scenario_name, scenario_data in stress_tests.items():
                    if isinstance(scenario_data, dict):
                        loss_pct = abs(scenario_data.get('loss_percentage', 0))
                        if loss_pct > max_loss:
                            max_loss = loss_pct
                            worst_scenario_name = scenario_name.replace('_', ' ').lower()
                
                if max_loss > 0:
                    return f"In a {worst_scenario_name}, your portfolio could lose around {max_loss:.1%}. This is based on how similar portfolios performed during past crises."
                else:
                    return "Stress testing shows your portfolio could face losses during market crises. More concentrated portfolios typically see larger losses than diversified ones."
            else:
                return "During market stress, your portfolio could see significant losses depending on its concentration and diversification. More balanced portfolios typically weather crises better."
        
        except Exception as e:
            return "During market stress, your portfolio could see significant losses depending on its concentration and diversification. More balanced portfolios typically weather crises better."
    
    def _suggest_improvements(self, data: Dict) -> str:
        """Suggest portfolio improvements"""
        try:
            health_metrics = data.get('health_metrics')
            portfolio = data.get('portfolio', {})
            
            # Handle both dataclass and dictionary formats
            if health_metrics and hasattr(health_metrics, 'recommendations'):
                # health_metrics is a HealthMetrics dataclass
                recommendations = health_metrics.recommendations
                score = health_metrics.overall_score
            elif health_metrics and isinstance(health_metrics, dict):
                # health_metrics is a dictionary
                recommendations = health_metrics.get('recommendations', [])
                score = health_metrics.get('overall_score', 65)
            else:
                # No health metrics available, generate basic recommendations
                recommendations = []
                score = 65
            
            if recommendations:
                return f"To improve your portfolio: {recommendations[0]}. This would help reduce concentration risk and improve diversification."
            else:
                # Generate basic recommendations based on portfolio
                if portfolio:
                    max_weight = max(portfolio.values()) if portfolio.values() else 0
                    if max_weight > 0.4:
                        return "Consider reducing your largest position to under 30% and adding more diverse holdings for better risk management."
                    elif len(portfolio) < 5:
                        return "Consider adding more positions across different sectors or asset classes to improve diversification."
                    else:
                        return "Your portfolio structure looks solid. Consider periodic rebalancing to maintain your target allocation."
                else:
                    return "Consider diversifying across different sectors, asset classes, and geographic regions for better risk management."
        
        except Exception as e:
            return "To improve your portfolio, consider reducing concentration in large positions and adding diversification across different sectors or asset types like bonds, REITs, or international stocks."
    
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
    st.markdown("### 🏥 Enhanced Portfolio Health")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Enhanced health gauge
        health_score = health_metrics.overall_score if hasattr(health_metrics, 'overall_score') else 65
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            title={'text': f"Health Score: {health_metrics.health_level}"},
            delta={'reference': 75, 'position': "top"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkgreen" if health_score >= 75 else 
                              "orange" if health_score >= 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Enhanced health component scores  
        st.metric("Concentration Risk", 
                 f"{health_metrics.concentration_risk:.0f}/100",
                 delta="Lower is better")
        
        st.metric("Diversification", 
                 f"{health_metrics.diversification_score:.0f}/100",
                 delta="Higher is better")
        
        # Show correlation if available
        if hasattr(health_metrics, 'correlation_score'):
            st.metric("Correlation Health", 
                     f"{health_metrics.correlation_score:.0f}/100",
                     delta="50-80 optimal")
    
    with col3:
        # Portfolio stats
        max_position = max(portfolio.values()) * 100 if portfolio else 0
        st.metric("Largest Position", f"{max_position:.1f}%")
        st.metric("Number of Holdings", len(portfolio))

        # NEW: Add regime fitness metric
        if hasattr(health_metrics, 'regime_fitness_score'):
            st.metric("Regime Fitness", 
                     f"{health_metrics.regime_fitness_score:.0f}/100",
                     delta="Higher is better")
        
        if hasattr(health_metrics, 'factor_balance_score'):
            st.metric("Factor Balance", 
                     f"{health_metrics.factor_balance_score:.0f}/100",
                     delta="Higher is better")
        
        # Risk level indicator
        if health_score >= 80:
            st.success("🟢 Low Risk")
        elif health_score >= 65:
            st.info("🟡 Moderate Risk")
        elif health_score >= 50:
            st.warning("🟠 Elevated Risk")
        else:
            st.error("🔴 High Risk")
    
    # Enhanced expandable sections
    with st.expander("🚨 Risk Analysis & Recommendations", expanded=bool(health_score < 65)):
        
        col_risks, col_tips = st.columns(2)
        
        with col_risks:
            st.markdown("**🎯 Key Risk Factors:**")
            for risk in health_metrics.key_risks:
                if "concentration" in risk.lower():
                    st.warning(f"⚠️ {risk}")
                elif "diversification" in risk.lower():
                    st.info(f"ℹ️ {risk}")
                else:
                    st.write(f"• {risk}")
        
        with col_tips:
            st.markdown("**💡 Improvement Actions:**")
            for i, rec in enumerate(health_metrics.recommendations, 1):
                st.write(f"**{i}.** {rec}")
    
    # Simple portfolio insights
    with st.expander("📊 Portfolio Insights", expanded=False):
        st.markdown("**📈 Portfolio Breakdown:**")
        
        # Create a simple breakdown chart
        components = ['Concentration\n(Inverted)', 'Diversification', 'Correlation', 'Regime Fitness', 'Factor Balance']
        scores = [
            100 - health_metrics.concentration_risk,  # Invert so higher is better
            health_metrics.diversification_score,
            health_metrics.correlation_score,
            health_metrics.regime_fitness_score,
            health_metrics.factor_balance_score
        ]
        
        fig_components = go.Figure(data=[
            go.Bar(
                x=components,
                y=scores,
                text=[f"{score:.0f}" for score in scores],
                textposition='auto',
                marker_color=['red' if s < 60 else 'orange' if s < 75 else 'green' for s in scores]
            )
        ])
        
        fig_components.update_layout(
            title="All Health Components (Higher = Better)",
            yaxis_title="Score (0-100)",
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig_components, use_container_width=True)
        
        # Quick insights
        st.markdown("**💡 Quick Insights:**")
        if max_position > 40:
            st.write(f"• 🎯 **Action**: Consider reducing {max_position:.1f}% position")
        elif len(portfolio) < 4:
            st.write(f"• 🎯 **Suggestion**: Add more holdings for diversification")
        else:
            st.write(f"• ✅ **Status**: Portfolio structure looks balanced")
    
    
    # Stress testing
    st.markdown("### 🔥 Stress Test Results")
    
    try:
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
        
        # Stress test summary
        st.markdown("**Crisis Scenario Summary:**")
        for scenario, data in stress_tests.items():
            loss_pct = abs(data['loss_percentage'])
            loss_amount = data['loss_amount']
            st.write(f"• **{scenario}**: {loss_pct:.1%} loss (${loss_amount:,.0f})")
            
    except Exception as e:
        st.warning("Stress test visualization temporarily unavailable. Analysis results above are still valid.")
        # Fallback display
        if stress_tests:
            st.markdown("**Crisis Scenario Summary:**")
            for scenario, data in stress_tests.items():
                st.write(f"• **{scenario}**: Potential significant losses during crisis")
        else:
            st.info("Stress testing will be available with next analysis update.")
    
    # AI Assistant
    st.markdown("### 🤖 AI Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💡 Why is my portfolio risky?", key="risky_btn", use_container_width=True):
            try:
                response = analyzer.ai_assistant.process_question("Why is my portfolio risky?", results)
            except Exception:
                response = "Your portfolio shows some concentration that could increase risk during market volatility. The main drivers are typically large positions in similar types of investments that tend to move together."
            st.session_state.chat_history.append(("Why is my portfolio risky?", response))
            st.rerun()
        
        if st.button("🏥 Explain my health score", key="health_btn", use_container_width=True):
            try:
                response = analyzer.ai_assistant.process_question("Explain my health score", results)
            except Exception:
                response = "Your portfolio health score reflects how well-diversified your holdings are. A higher score indicates better balance and lower concentration risk."
            st.session_state.chat_history.append(("Explain my health score", response))
            st.rerun()
    
    with col2:
        if st.button("💥 What if markets crash?", key="crash_btn", use_container_width=True):
            try:
                response = analyzer.ai_assistant.process_question("What if markets crash?", results)
            except Exception:
                response = "During market stress, your portfolio could see significant losses depending on its concentration and diversification. More balanced portfolios typically weather crises better."
            st.session_state.chat_history.append(("What if markets crash?", response))
            st.rerun()
        
        if st.button("🛠️ How can I improve?", key="improve_btn", use_container_width=True):
            try:
                response = analyzer.ai_assistant.process_question("How can I improve?", results)
            except Exception:
                response = "To improve your portfolio, consider reducing concentration in large positions and adding diversification across different sectors or asset types like bonds, REITs, or international stocks."
            st.session_state.chat_history.append(("How can I improve?", response))
            st.rerun()
    
    # Custom question
    with st.expander("💬 Ask a Custom Question"):
        custom_question = st.text_input("Your question:", placeholder="Ask about your portfolio...")
        if st.button("Ask AI", key="custom_ai_btn") and custom_question:
            try:
                response = analyzer.ai_assistant.process_question(custom_question, results)
            except Exception:
                response = "I can help explain your portfolio's risk characteristics, health score, or suggest improvements. What specific aspect interests you?"
            st.session_state.chat_history.append((custom_question, response))
            st.rerun()
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("**💬 Conversation:**")
        for question, answer in st.session_state.chat_history[-3:]:  # Show last 3
            st.markdown(f"**🙋 You:** {question}")
            st.markdown(f"**🤖 AI:** {answer}")
            st.markdown("---")

if __name__ == "__main__":
    main()
