#!/usr/bin/env python3
"""
Portfolio Risk Analyzer MVP - Quick Fix Version
Professional-grade portfolio analysis with real market data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io
import base64
import uuid

# Import your portfolio analysis engine
try:
    from real_data_portfolio import RealDataPortfolioAgent
    ENGINE_AVAILABLE = True
except ImportError:
    st.error("Portfolio analysis engine not found. Make sure real_data_portfolio.py is in the same directory.")
    ENGINE_AVAILABLE = False

from dataclasses import dataclass
from typing import List,Dict

@dataclass
class PortfolioHealthMetrics:
    """Comprehensive portfolio health assessment"""
    overall_score: float  # 0-100
    concentration_risk: float  # 0-100
    correlation_health: float  # 0-100  
    regime_fitness: float  # 0-100
    stress_resilience: float  # 0-100
    factor_balance: float  # 0-100
    
    health_level: str  # "Excellent", "Good", "Fair", "Poor"
    key_risks: List[str]
    improvement_priorities: List[str]

class EnhancedPortfolioHealthMonitor:
    """Professional-grade portfolio health monitoring"""
    
    def __init__(self):
        self.health_weights = {
            'concentration_risk': 0.25,
            'correlation_health': 0.20,
            'regime_fitness': 0.20,
            'stress_resilience': 0.20,
            'factor_balance': 0.15
        }
        
    def calculate_portfolio_health(self, portfolio: Dict[str, float], 
                                 analysis_results: Dict) -> PortfolioHealthMetrics:
        """Calculate comprehensive portfolio health score"""
        
        # Individual health components
        concentration = self._assess_concentration_risk(portfolio)
        correlation = self._assess_correlation_health(portfolio, analysis_results)
        regime_fit = self._assess_regime_fitness(portfolio, analysis_results)
        stress_resilience = self._assess_stress_resilience(analysis_results)
        factor_balance = self._assess_factor_balance(portfolio)
        
        # Calculate weighted overall score
        component_scores = {
            'concentration_risk': concentration,
            'correlation_health': correlation,
            'regime_fitness': regime_fit,
            'stress_resilience': stress_resilience,
            'factor_balance': factor_balance
        }
        
        overall_score = sum(
            score * self.health_weights[component] 
            for component, score in component_scores.items()
        )
        
        # Determine health level
        health_level = self._determine_health_level(overall_score)
        
        # Identify key risks and improvements
        key_risks = self._identify_key_risks(component_scores)
        improvements = self._generate_improvement_priorities(component_scores)
        
        return PortfolioHealthMetrics(
            overall_score=overall_score,
            concentration_risk=concentration,
            correlation_health=correlation,
            regime_fitness=regime_fit,
            stress_resilience=stress_resilience,
            factor_balance=factor_balance,
            health_level=health_level,
            key_risks=key_risks,
            improvement_priorities=improvements
        )
    
    def _assess_concentration_risk(self, portfolio: Dict[str, float]) -> float:
        """Assess concentration risk (higher score = less concentrated)"""
        if not portfolio:
            return 0
        
        max_weight = max(portfolio.values())
        num_positions = len(portfolio)
        
        # Penalize high concentration
        concentration_penalty = max(0, (max_weight - 0.25) * 200)  # Penalty for >25% positions
        
        # Reward diversification
        diversification_bonus = min(50, num_positions * 8)  # Bonus for more positions
        
        # Calculate score (0-100)
        score = max(0, 100 - concentration_penalty + diversification_bonus)
        return min(100, score)
    
    def _assess_correlation_health(self, portfolio: Dict[str, float], 
                                 analysis_results: Dict) -> float:
        """Assess correlation health of portfolio"""
        try:
            # Extract correlation data from analysis results
            correlation_matrix = analysis_results.get('model_parameters', {}).get('correlation_matrix')
            
            if not correlation_matrix:
                return 60  # Neutral score if no correlation data
            
            correlation_matrix = np.array(correlation_matrix)
            
            # Calculate average correlation (excluding diagonal)
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            avg_correlation = np.mean(correlation_matrix[mask])
            
            # Optimal correlation is around 0.3-0.5
            if 0.3 <= avg_correlation <= 0.5:
                score = 100
            elif avg_correlation < 0.3:
                score = 70 + (avg_correlation * 100)  # Low correlation is still good
            else:
                score = max(0, 100 - (avg_correlation - 0.5) * 200)  # Penalize high correlation
            
            return min(100, max(0, score))
            
        except Exception:
            return 60  # Neutral score on error
    
    def _assess_regime_fitness(self, portfolio: Dict[str, float], 
                             analysis_results: Dict) -> float:
        """Assess how well portfolio fits current market regime"""
        
        # Enhanced regime fitness assessment
        portfolio_size = len(portfolio)
        max_weight = max(portfolio.values()) if portfolio else 0
        
        # Check for sector concentration (basic heuristic)
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        tech_exposure = sum(portfolio.get(ticker, 0) for ticker in tech_tickers)
        
        # Assess based on portfolio characteristics
        base_score = 70  # Default regime fitness
        
        # Penalize high concentration
        if max_weight > 0.5:
            base_score -= 30
        elif max_weight > 0.3:
            base_score -= 15
        
        # Penalize excessive tech concentration
        if tech_exposure > 0.7:
            base_score -= 20
        
        # Reward diversification
        if portfolio_size >= 5:
            base_score += 15
        
        return max(20, min(100, base_score))
    
    def _assess_stress_resilience(self, analysis_results: Dict) -> float:
        """Assess portfolio's resilience to stress scenarios"""
        try:
            stress_tests = analysis_results.get('stress_tests', {})
            
            if not stress_tests:
                return 60  # Neutral if no stress test data
            
            # Average stress test losses
            stress_losses = []
            for scenario, results in stress_tests.items():
                var_loss = abs(results.get('var_95', 0))
                stress_losses.append(var_loss)
            
            if stress_losses:
                avg_stress_loss = np.mean(stress_losses)
                
                # Score based on stress loss severity
                if avg_stress_loss < 0.15:  # Less than 15% loss
                    return 90
                elif avg_stress_loss < 0.25:  # 15-25% loss
                    return 70
                elif avg_stress_loss < 0.35:  # 25-35% loss
                    return 50
                else:  # >35% loss
                    return 30
            
            return 60
            
        except Exception:
            return 60
    
    def _assess_factor_balance(self, portfolio: Dict[str, float]) -> float:
        """Assess factor balance across portfolio"""
        
        if not portfolio:
            return 0
            
        # Simple factor balance assessment based on portfolio characteristics
        num_positions = len(portfolio)
        max_weight = max(portfolio.values())
        
        # Check for sector diversity (basic implementation)
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        financial_tickers = ['JPM', 'BAC', 'WFC', 'C', 'GS']
        bond_tickers = ['BND', 'TLT', 'AGG', 'IEF']
        
        tech_exposure = sum(portfolio.get(ticker, 0) for ticker in tech_tickers)
        financial_exposure = sum(portfolio.get(ticker, 0) for ticker in financial_tickers)
        bond_exposure = sum(portfolio.get(ticker, 0) for ticker in bond_tickers)
        
        # Base score from position count
        position_score = min(40, num_positions * 6)
        
        # Concentration penalty
        concentration_penalty = max(0, (max_weight - 0.2) * 100)
        
        # Sector balance bonus
        sector_balance = 0
        if tech_exposure < 0.5:  # Not too tech heavy
            sector_balance += 15
        if bond_exposure > 0.1:  # Some defensive allocation
            sector_balance += 15
        if financial_exposure > 0.05:  # Some financial exposure
            sector_balance += 10
        
        total_score = position_score + sector_balance - concentration_penalty
        return max(0, min(100, total_score))
    
    def _determine_health_level(self, overall_score: float) -> str:
        """Determine health level from overall score"""
        if overall_score >= 85:
            return "Excellent"
        elif overall_score >= 70:
            return "Good"
        elif overall_score >= 50:
            return "Fair"
        else:
            return "Poor"
    
    def _identify_key_risks(self, component_scores: Dict[str, float]) -> List[str]:
        """Identify the most significant portfolio risks"""
        risks = []
        
        if component_scores['concentration_risk'] < 60:
            risks.append("High concentration risk - consider diversifying large positions")
        
        if component_scores['correlation_health'] < 60:
            risks.append("High correlation between holdings - limited diversification benefit")
        
        if component_scores['stress_resilience'] < 60:
            risks.append("Poor stress test performance - vulnerable to market downturns")
        
        if component_scores['factor_balance'] < 60:
            risks.append("Imbalanced factor exposures - concentrated investment style risk")
        
        if component_scores['regime_fitness'] < 60:
            risks.append("Poor current market regime fit - consider tactical adjustments")
        
        return risks[:3]  # Return top 3 risks
    
    def _generate_improvement_priorities(self, component_scores: Dict[str, float]) -> List[str]:
        """Generate prioritized improvement recommendations"""
        improvements = []
        
        # Sort components by score (lowest first = highest priority)
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1])
        
        improvement_map = {
            'concentration_risk': "Reduce position sizes and add more holdings",
            'correlation_health': "Add uncorrelated assets to improve diversification", 
            'stress_resilience': "Include defensive assets for downside protection",
            'factor_balance': "Balance growth/value and size exposures",
            'regime_fitness': "Adjust allocation for current market conditions"
        }
        
        for component, score in sorted_components:
            if score < 70:  # Only suggest improvements for weak areas
                improvements.append(improvement_map[component])
        
        return improvements[:3]  # Return top 3 priorities

def create_performance_charts(analysis_data, portfolio_value, portfolio):
    """
    Create comprehensive performance charts - NEW USER-REQUESTED FEATURE!
    """
    
    st.markdown("### 📈 Portfolio Performance Analysis")
    st.info("💡 **New Feature!** Added based on user feedback: 'Can I plot returns and drawdowns?'")
    
    # Extract portfolio returns from analysis
    base_case = analysis_data.get('base_case', {})
    portfolio_returns = base_case.get('portfolio_returns', [])
    
    if len(portfolio_returns) == 0:
        st.warning("No return data available for charting")
        return
    
    # Create tabs for different chart views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cumulative Returns", "📉 Drawdown Analysis", "📋 Performance Stats", "🔄 Rolling Metrics"])
    
    with tab1:
        display_cumulative_returns_chart(portfolio_returns, portfolio)
    
    with tab2:
        display_drawdown_chart(portfolio_returns, portfolio)
    
    with tab3:
        display_performance_statistics(portfolio_returns, portfolio_value, analysis_data)
    
    with tab4:
        display_rolling_metrics_chart(portfolio_returns, portfolio)

def display_cumulative_returns_chart(portfolio_returns, portfolio):
    """Display cumulative returns over time"""
    
    st.markdown("#### 📈 Cumulative Portfolio Returns")
    
    # Calculate cumulative returns
    cumulative_returns = np.cumsum(portfolio_returns)
    cumulative_pct = (np.exp(cumulative_returns) - 1) * 100
    
    # Create the chart
    fig_returns = go.Figure()
    
    # Add portfolio returns line
    fig_returns.add_trace(go.Scatter(
        x=list(range(len(cumulative_pct))),
        y=cumulative_pct,
        mode='lines',
        name='Portfolio Returns',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='Day %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line for reference
    fig_returns.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Break-even")
    
    # Highlight best and worst periods
    if len(cumulative_pct) > 10:
        max_return_idx = np.argmax(cumulative_pct)
        min_return_idx = np.argmin(cumulative_pct)
        
        # Mark peak
        fig_returns.add_trace(go.Scatter(
            x=[max_return_idx],
            y=[cumulative_pct[max_return_idx]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='Peak Return',
            hovertemplate=f'Peak: {cumulative_pct[max_return_idx]:.2f}%<extra></extra>'
        ))
        
        # Mark trough
        fig_returns.add_trace(go.Scatter(
            x=[min_return_idx],
            y=[cumulative_pct[min_return_idx]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle'),
            name='Lowest Point',
            hovertemplate=f'Trough: {cumulative_pct[min_return_idx]:.2f}%<extra></extra>'
        ))
    
    # Update layout
    fig_returns.update_layout(
        title="Cumulative Portfolio Returns Over Time",
        xaxis_title="Trading Days",
        yaxis_title="Cumulative Return (%)",
        height=450,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_returns, use_container_width=True)
    
    # Key insights
    total_return = cumulative_pct[-1]
    best_day = np.max(np.diff(cumulative_pct)) if len(cumulative_pct) > 1 else 0
    worst_day = np.min(np.diff(cumulative_pct)) if len(cumulative_pct) > 1 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Return", f"{total_return:.2f}%")
    with col2:
        st.metric("Best Day", f"{best_day:.2f}%", delta_color="normal")
    with col3:
        st.metric("Worst Day", f"{worst_day:.2f}%", delta_color="inverse")

def display_drawdown_chart(portfolio_returns, portfolio):
    """Display underwater/drawdown chart"""
    
    st.markdown("#### 📉 Portfolio Drawdown Analysis")
    st.info("Drawdown shows how far the portfolio is below its previous peak")
    
    # Calculate drawdowns
    cumulative_returns = np.cumsum(portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    drawdown_pct = drawdown * 100  # Convert to percentage
    
    # Create underwater chart
    fig_drawdown = go.Figure()
    
    # Add drawdown area chart (underwater curve)
    fig_drawdown.add_trace(go.Scatter(
        x=list(range(len(drawdown_pct))),
        y=drawdown_pct,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red', width=2),
        name='Drawdown',
        hovertemplate='Day %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Add zero line
    fig_drawdown.add_hline(y=0, line_dash="solid", line_color="black", 
                          annotation_text="Peak Level")
    
    # Highlight maximum drawdown
    max_drawdown_idx = np.argmin(drawdown_pct)
    max_drawdown_value = drawdown_pct[max_drawdown_idx]
    
    fig_drawdown.add_trace(go.Scatter(
        x=[max_drawdown_idx],
        y=[max_drawdown_value],
        mode='markers',
        marker=dict(color='darkred', size=12, symbol='x'),
        name='Max Drawdown',
        hovertemplate=f'Max Drawdown: {max_drawdown_value:.2f}%<extra></extra>'
    ))
    
    # Update layout
    fig_drawdown.update_layout(
        title="Portfolio Drawdown (Underwater Chart)",
        xaxis_title="Trading Days",
        yaxis_title="Drawdown (%)",
        height=450,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_drawdown, use_container_width=True)
    
    # Drawdown statistics
    max_drawdown = abs(max_drawdown_value)
    
    # Calculate recovery time (simplified)
    underwater_periods = np.where(drawdown_pct < -0.5)[0]  # Periods with >0.5% drawdown
    avg_underwater_time = len(underwater_periods) / len(drawdown_pct) * 100 if len(underwater_periods) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
    with col2:
        st.metric("Time Underwater", f"{avg_underwater_time:.1f}%", 
                 help="Percentage of time portfolio was below its peak")
    with col3:
        current_drawdown = abs(drawdown_pct[-1]) if drawdown_pct[-1] < 0 else 0
        st.metric("Current Drawdown", f"{current_drawdown:.2f}%", delta_color="inverse")

# DAY 2 AFTERNOON: Performance Statistics (3 hours)

def display_performance_statistics(portfolio_returns, portfolio_value, analysis_data):
    """Display comprehensive performance statistics table"""
    
    st.markdown("#### 📋 Detailed Performance Statistics")
    
    # Calculate performance metrics
    total_days = len(portfolio_returns)
    cumulative_returns = np.cumsum(portfolio_returns)
    
    # Return metrics
    total_return = (np.exp(cumulative_returns[-1]) - 1) * 100
    annualized_return = ((1 + total_return/100) ** (252/total_days) - 1) * 100 if total_days > 0 else 0
    
    # Risk metrics  
    daily_vol = np.std(portfolio_returns) * 100
    annualized_vol = daily_vol * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = (annualized_return - 2) / annualized_vol if annualized_vol > 0 else 0  # Assuming 2% risk-free rate
    
    # Drawdown metrics
    cumulative_returns_series = np.cumsum(portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns_series)
    drawdown = cumulative_returns_series - running_max
    max_drawdown = abs(np.min(drawdown)) * 100
    
    # Win/Loss metrics
    positive_days = np.sum(portfolio_returns > 0)
    negative_days = np.sum(portfolio_returns < 0)
    win_rate = (positive_days / total_days) * 100 if total_days > 0 else 0
    
    # Best/Worst days
    best_day = np.max(portfolio_returns) * 100 if len(portfolio_returns) > 0 else 0
    worst_day = np.min(portfolio_returns) * 100 if len(portfolio_returns) > 0 else 0
    
    # Create performance metrics table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Return Metrics**")
        st.write(f"• **Total Return**: {total_return:.2f}%")
        st.write(f"• **Annualized Return**: {annualized_return:.2f}%")
        st.write(f"• **Best Day**: {best_day:.2f}%")
        st.write(f"• **Worst Day**: {worst_day:.2f}%")
        st.write(f"• **Win Rate**: {win_rate:.1f}%")
        
        st.markdown("**⚡ Risk Metrics**")
        st.write(f"• **Daily Volatility**: {daily_vol:.2f}%")
        st.write(f"• **Annualized Volatility**: {annualized_vol:.2f}%")
        st.write(f"• **Maximum Drawdown**: {max_drawdown:.2f}%")
    
    with col2:
        st.markdown("**🎯 Risk-Adjusted Metrics**")
        st.write(f"• **Sharpe Ratio**: {sharpe_ratio:.2f}")
        
        # Add Calmar ratio (return/max drawdown)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        st.write(f"• **Calmar Ratio**: {calmar_ratio:.2f}")
        
        # Add some portfolio composition stats
        st.markdown("**📊 Portfolio Stats**")
        st.write(f"• **Analysis Period**: {total_days} trading days")
        st.write(f"• **Positive Days**: {positive_days} ({win_rate:.1f}%)")
        st.write(f"• **Negative Days**: {negative_days} ({100-win_rate:.1f}%)")
    
    # Performance grade
    st.markdown("#### 🏆 Performance Grade")
    
    # Simple grading system
    grade_score = 0
    if annualized_return > 15:
        grade_score += 25
    elif annualized_return > 10:
        grade_score += 20
    elif annualized_return > 5:
        grade_score += 15
    
    if sharpe_ratio > 1.5:
        grade_score += 25
    elif sharpe_ratio > 1.0:
        grade_score += 20
    elif sharpe_ratio > 0.5:
        grade_score += 15
    
    if max_drawdown < 10:
        grade_score += 25
    elif max_drawdown < 20:
        grade_score += 20
    elif max_drawdown < 30:
        grade_score += 15
    
    if win_rate > 60:
        grade_score += 25
    elif win_rate > 55:
        grade_score += 20
    elif win_rate > 50:
        grade_score += 15
    
    # Assign letter grade
    if grade_score >= 90:
        grade = "A+ (Excellent)"
        grade_color = "success"
    elif grade_score >= 80:
        grade = "A (Very Good)"
        grade_color = "success"
    elif grade_score >= 70:
        grade = "B (Good)"
        grade_color = "info"
    elif grade_score >= 60:
        grade = "C (Average)"
        grade_color = "warning"
    else:
        grade = "D (Below Average)"
        grade_color = "error"
    
    if grade_color == "success":
        st.success(f"**Portfolio Performance Grade: {grade}**")
    elif grade_color == "info":
        st.info(f"**Portfolio Performance Grade: {grade}**")
    elif grade_color == "warning":
        st.warning(f"**Portfolio Performance Grade: {grade}**")
    else:
        st.error(f"**Portfolio Performance Grade: {grade}**")

def display_rolling_metrics_chart(portfolio_returns, portfolio):
    """Display rolling performance metrics"""
    
    st.markdown("#### 🔄 Rolling Performance Metrics")
    st.info("Rolling metrics show how performance changes over time")
    
    if len(portfolio_returns) < 30:
        st.warning("Need at least 30 days of data for rolling metrics")
        return
    
    # Calculate rolling metrics (30-day window)
    window = min(30, len(portfolio_returns) // 3)
    
    rolling_returns = []
    rolling_volatility = []
    rolling_sharpe = []
    
    for i in range(window, len(portfolio_returns)):
        window_returns = portfolio_returns[i-window:i]
        
        # Rolling return (annualized)
        cum_return = np.sum(window_returns)
        ann_return = (np.exp(cum_return) - 1) * (252/window) * 100
        rolling_returns.append(ann_return)
        
        # Rolling volatility (annualized)
        vol = np.std(window_returns) * np.sqrt(252) * 100
        rolling_volatility.append(vol)
        
        # Rolling Sharpe ratio
        sharpe = (ann_return - 2) / vol if vol > 0 else 0
        rolling_sharpe.append(sharpe)
    
    # Create rolling metrics chart
    fig_rolling = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Rolling Returns (%)', 'Rolling Volatility (%)', 'Rolling Sharpe Ratio'),
        vertical_spacing=0.1
    )
    
    x_axis = list(range(window, len(portfolio_returns)))
    
    # Rolling returns
    fig_rolling.add_trace(
        go.Scatter(x=x_axis, y=rolling_returns, name='Rolling Returns', 
                  line=dict(color='blue')), row=1, col=1
    )
    
    # Rolling volatility
    fig_rolling.add_trace(
        go.Scatter(x=x_axis, y=rolling_volatility, name='Rolling Volatility',
                  line=dict(color='orange')), row=2, col=1
    )
    
    # Rolling Sharpe
    fig_rolling.add_trace(
        go.Scatter(x=x_axis, y=rolling_sharpe, name='Rolling Sharpe',
                  line=dict(color='green')), row=3, col=1
    )
    
    # Add zero line for Sharpe ratio
    fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig_rolling.update_layout(height=800, showlegend=False)
    fig_rolling.update_xaxes(title_text="Trading Days", row=3, col=1)
    
    st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Rolling metrics summary
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_rolling_return = np.mean(rolling_returns)
        st.metric("Avg Rolling Return", f"{avg_rolling_return:.1f}%")
    with col2:
        avg_rolling_vol = np.mean(rolling_volatility)
        st.metric("Avg Rolling Volatility", f"{avg_rolling_vol:.1f}%")
    with col3:
        avg_rolling_sharpe = np.mean(rolling_sharpe)
        st.metric("Avg Rolling Sharpe", f"{avg_rolling_sharpe:.2f}")

# ============================================================================
# ANALYTICS TRACKING FUNCTIONS
# ============================================================================

def initialize_session():
    """Initialize session tracking"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if 'usage_analytics' not in st.session_state:
        st.session_state.usage_analytics = []
    
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()

def track_usage(event_type, portfolio_data=None, error_data=None):
    """Track user interactions for market validation"""
    
    event = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'session_id': st.session_state.get('session_id', 'unknown'),
        'session_duration': (datetime.now() - st.session_state.get('session_start_time', datetime.now())).total_seconds(),
        'portfolio_data': portfolio_data,
        'error_data': error_data
    }
    
    st.session_state.usage_analytics.append(event)
    
    # For development - remove in production
    print(f"📊 Analytics: {event_type} | Session: {event['session_id']}")

def add_feedback_section():
    """Add simple feedback collection"""
    
    st.markdown("---")
    st.markdown("### 💡 Help Us Improve This Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 This was helpful!", key="positive_feedback"):
            track_usage('positive_feedback')
            st.success("Thanks for the feedback! 🙏")
    
    with col2:
        if st.button("👎 Needs improvement", key="negative_feedback"):
            track_usage('negative_feedback')  
            st.success("Thanks! We're working on improvements. 🔧")
    
    # Feature request section
    st.markdown("**What feature would be most valuable to you?**")
    feature_request = st.text_area(
        "Feature Request",
        height=80,
        placeholder="e.g., Portfolio optimization suggestions, more asset classes, better visualizations, comparison tools...",
        key="feature_request_input",
        label_visibility="hidden"
    )
    
    if st.button("📝 Submit Feature Request", key="submit_feature") and feature_request.strip():
        track_usage('feature_request', {'request': feature_request.strip()})
        st.success("Feature request recorded! This helps us prioritize development. 🚀")

def display_analytics_summary():
    """Display analytics summary for admin only"""
    
    # Admin password protection
    admin_password = st.sidebar.text_input("Admin Access", type="password", key="admin_pw")
    
    if admin_password == "Gertie78*":  # Change this password!
        if st.sidebar.checkbox("🔍 Show Analytics (Admin)", value=False):
            with st.sidebar.expander("Analytics Data"):
                analytics = st.session_state.get('usage_analytics', [])
                st.write(f"**Session ID**: {st.session_state.get('session_id', 'None')}")
                st.write(f"**Events Tracked**: {len(analytics)}")
                
                if analytics:
                    event_types = {}
                    for event in analytics:
                        event_type = event['event_type']
                        event_types[event_type] = event_types.get(event_type, 0) + 1
                    
                    st.write("**Event Summary**:")
                    for event_type, count in event_types.items():
                        st.write(f"• {event_type}: {count}")
                    
                    # Add download button for analytics
                    if st.button("📥 Download Analytics Report", key="download_analytics"):
                        create_analytics_download(analytics)
                    
                    # Add manual copy option as backup
                    if st.button("📋 Copy Analytics Report", key="copy_analytics"):
                        display_analytics_text(analytics)
    elif admin_password:
        st.sidebar.error("❌ Invalid admin password")

# ============================================================================
# ORIGINAL APP CODE WITH TRACKING INTEGRATED
# ============================================================================

# Page config
st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session tracking
    initialize_session()
    
    # Track page view
    track_usage('page_view')
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Portfolio Risk Analyzer</h1>
        <p>Professional-grade analysis with real market data in 30 seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if engine is available
    if not ENGINE_AVAILABLE:
        track_usage('engine_unavailable')
        st.error("❌ Portfolio analysis engine not available. Please ensure real_data_portfolio.py is in the same directory.")
        st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **Professional Portfolio Analysis**
        
        ✅ Real market data from Yahoo Finance
        ✅ Institutional-quality risk metrics
        ✅ Value-at-Risk calculations
        ✅ Stress testing scenarios
        ✅ Portfolio health scoring
        🪙 **Crypto portfolio support**
        """)
        
        # Clear session button
        if st.button("🔄 Start Fresh"):
            track_usage('session_reset')
            # Clear all session state except analytics
            analytics_backup = st.session_state.get('usage_analytics', [])
            session_id_backup = st.session_state.get('session_id', None)
            session_start_backup = st.session_state.get('session_start_time', None)
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Restore analytics
            st.session_state.usage_analytics = analytics_backup
            st.session_state.session_id = session_id_backup
            st.session_state.session_start_time = session_start_backup
            
            st.rerun()
        
        if st.button("Conservative"):
            track_usage('example_portfolio_selected', {'type': 'conservative'})
            st.session_state.portfolio_input = "50% SPY, 30% BND, 20% VTI"
            st.rerun()
        if st.button("Growth Tech"):
            track_usage('example_portfolio_selected', {'type': 'growth_tech'})
            st.session_state.portfolio_input = "40% AAPL, 30% MSFT, 20% GOOGL, 10% AMZN"
            st.rerun()
        if st.button("Balanced ETF"):
            track_usage('example_portfolio_selected', {'type': 'balanced_etf'})
            st.session_state.portfolio_input = "Equal weight SPY QQQ IWM"
            st.rerun()
        
        # NEW: Crypto portfolios
        st.markdown("**🪙 Crypto Portfolios**")
        if st.button("Crypto Core"):
            track_usage('example_portfolio_selected', {'type': 'crypto_core'})
            st.session_state.portfolio_input = "60% BTC-USD, 40% ETH-USD"
            st.rerun()
        if st.button("Crypto Diversified"):
            track_usage('example_portfolio_selected', {'type': 'crypto_diversified'})
            st.session_state.portfolio_input = "40% BTC-USD, 30% ETH-USD, 20% SOL-USD, 10% ADA-USD"
            st.rerun()
        if st.button("Stocks + Crypto"):
            track_usage('example_portfolio_selected', {'type': 'stocks_crypto'})
            st.session_state.portfolio_input = "40% SPY, 30% QQQ, 20% BTC-USD, 10% ETH-USD"
            st.rerun()
        
        # Analytics summary (for debugging)
        display_analytics_summary()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Enter Your Portfolio")
        
        # Portfolio input
        portfolio_input = st.text_area(
            "Portfolio Holdings",
            value=st.session_state.get('portfolio_input', ''),
            height=120,
            placeholder="Enter your portfolio in one of these formats:\n\n• 40% AAPL, 30% MSFT, 20% GOOGL, 10% BND\n• Equal weight SPY QQQ VTI\n• 50% VOO, 50% BND\n• 60% BTC-USD, 40% ETH-USD (crypto)",  # ← ADD CRYPTO EXAMPLE
            help="Use percentage format or 'Equal weight' followed by ticker symbols. Crypto: use -USD suffix (BTC-USD, ETH-USD)"
        )
        
        # Track portfolio input changes
        if portfolio_input != st.session_state.get('last_portfolio_input', ''):
            if portfolio_input.strip():  # Only track non-empty inputs
                track_usage('portfolio_input_changed', {'input_length': len(portfolio_input)})
            st.session_state.last_portfolio_input = portfolio_input
        
        # Portfolio value input
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=1000,
            value=1000000,
            step=10000,
            format="%d",
            help="Total value of your portfolio for dollar impact calculations"
        )
        
        # Track portfolio value changes
        if portfolio_value != st.session_state.get('last_portfolio_value', 1000000):
            track_usage('portfolio_value_changed', {'value': portfolio_value})
            st.session_state.last_portfolio_value = portfolio_value
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Analyze Portfolio",
            type="primary",
            use_container_width=True
        )
        
        # Input validation feedback
        if portfolio_input:
            st.markdown("**Preview:**")
            if "%" in portfolio_input:
                st.success("✅ Percentage format detected")
                track_usage('input_format_detected', {'format': 'percentage'})
            elif "equal" in portfolio_input.lower():
                st.success("✅ Equal weight format detected")
                track_usage('input_format_detected', {'format': 'equal_weight'})
            else:
                st.warning("⚠️ Format not recognized - try examples above")
                track_usage('input_format_unrecognized', {'input': portfolio_input[:50]})
    
    with col2:
        st.header("📊 Analysis Results")
        
        if analyze_button and portfolio_input:
            # Track analysis attempt
            track_usage('analysis_started', {
                'portfolio_input': portfolio_input[:100],  # Truncate for privacy
                'portfolio_value': portfolio_value
            })
            run_portfolio_analysis(portfolio_input, portfolio_value)
        else:
            st.info("""
            👈 **Enter your portfolio to get started**
            
            Your analysis will include:
            - Real-time risk metrics
            - Portfolio health score
            - Stress test scenarios
            - Visual risk breakdown
            - 🪙 **Crypto portfolio analysis**
            - Downloadable report
            """)
            
    
    # Add feedback section at the bottom
    add_feedback_section()

def run_portfolio_analysis(portfolio_input, portfolio_value):
    """Run the portfolio analysis and display results"""
    
    with st.spinner("🔄 Analyzing your portfolio with real market data..."):
        try:
            # Initialize the analysis engine
            agent = RealDataPortfolioAgent(force_real_data=True)
            
            # Run analysis
            start_time = time.time()
            response = agent.process_message("streamlit_user", portfolio_input)
            analysis_time = time.time() - start_time
            
            # Track successful analysis
            track_usage('analysis_completed', {
                'portfolio_value': portfolio_value,
                'analysis_time': analysis_time,
                'success': True,
                'response_length': len(response)
            })
            
            # Get the context from the agent
            context = agent.user_contexts.get("streamlit_user")
            
            # Display success message
            st.success(f"✅ Analysis completed in {analysis_time:.1f} seconds")
            
            # Display results based on what we have
            if context and context.last_analysis:
                track_usage('detailed_analysis_displayed')
                display_detailed_analysis(context, portfolio_value, response)
            else:
                # Fallback to text response
                track_usage('text_analysis_displayed')
                st.markdown("### 📋 Analysis Summary")
                clean_response = response.replace('*', '').replace('•', '•')
                st.markdown(clean_response)
                
                # Simple download option
                if st.button("📥 Download Text Report"):
                    track_usage('text_report_downloaded')
                    create_simple_download(portfolio_input, portfolio_value, response)
            
        except Exception as e:
            # Track analysis failure
            track_usage('analysis_failed', error_data={
                'error_message': str(e)[:200],  # Truncate error for storage
                'error_type': type(e).__name__,
                'portfolio_input': portfolio_input[:100]
            })
            
            st.error(f"❌ Analysis failed: {str(e)}")
            
            # Provide helpful error guidance
            if "Invalid tickers" in str(e):
                track_usage('invalid_tickers_error')
                st.info("""
                **💡 Ticker Validation Failed**
                
                Try these verified examples:
                - `50% AAPL, 50% MSFT` (major stocks)
                - `Equal weight SPY QQQ BND` (popular ETFs)
                - `40% VOO, 60% BND` (Vanguard funds)
                """)
            elif "REAL DATA" in str(e):
                track_usage('real_data_error')
                st.info("""
                **💡 Market Data Issue**
                
                This usually means:
                - Market is closed and recent data unavailable
                - Network connectivity issues
                - Try again in a few moments
                """)

def display_detailed_analysis(context, portfolio_value, response):
    """Display detailed analysis with charts and metrics"""
    
    analysis_data = context.last_analysis
    portfolio = context.portfolio
    
    # Track what features are being viewed
    track_usage('detailed_analysis_viewed', {
        'portfolio_positions': len(portfolio) if portfolio else 0,
        'portfolio_value': portfolio_value
    })
    
    # Portfolio Overview
    st.markdown("### 📈 Portfolio Overview")
    
    # Portfolio composition chart
    if portfolio:
        track_usage('portfolio_chart_displayed')
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(portfolio.keys()),
            values=list(portfolio.values()),
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig_pie.update_layout(
            title="Portfolio Allocation",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Display portfolio breakdown
    if portfolio:
        st.markdown("**Portfolio Breakdown:**")
        for ticker, weight in portfolio.items():
            value = weight * portfolio_value
            st.write(f"• **{ticker}**: {weight:.1%} (${value:,.0f})")
    
    # Key Metrics Row
    st.markdown("### 📊 Key Risk Metrics")
    track_usage('risk_metrics_displayed')
    
    base_case = analysis_data.get('base_case', {})
    var_95 = base_case.get('var_95', 0)
    es_95 = base_case.get('es_95', 0)
    max_drawdown = base_case.get('max_drawdown', 0)
    
    # Calculate dollar amounts
    var_dollar = abs(var_95) * portfolio_value
    es_dollar = abs(es_95) * portfolio_value
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Daily VaR (95%)",
            f"${var_dollar:,.0f}",
            f"{var_95:.1%}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Expected Shortfall",
            f"${es_dollar:,.0f}",
            f"{es_95:.1%}",
            delta_color="inverse"
        )
    
    with col3:
        portfolio_returns = base_case.get('portfolio_returns', [])
        if len(portfolio_returns) > 0:
            volatility = np.std(portfolio_returns) * np.sqrt(252)
            st.metric(
                "Annual Volatility",
                f"{volatility:.1%}",
                delta_color="off"
            )
        else:
            st.metric("Annual Volatility", "N/A")
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{abs(max_drawdown):.1%}",
            delta_color="inverse"
        )
    
    # Risk Level Assessment
    risk_level = "High" if abs(var_95) > 0.03 else "Moderate" if abs(var_95) > 0.02 else "Low"
    risk_color = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}
    
    track_usage('risk_level_calculated', {'risk_level': risk_level, 'var_95': abs(var_95)})
    
    st.markdown(f"### {risk_color[risk_level]} Risk Level: {risk_level}")
    
    # Portfolio Health Score
    display_enhanced_health_analysis(portfolio,analysis_data)
    
    # Stress Testing Results
    display_stress_tests(analysis_data, portfolio_value)
    
    # Data Source Information
    st.markdown("### 📅 Analysis Details")
    
    col1, col2 = st.columns(2)
    with col1:
        data_source = analysis_data.get('data_source', 'Unknown')
        data_period = analysis_data.get('data_period', 0)
        st.info(f"**Data Source:** {data_source}\n**Trading Days:** {data_period}")
    
    with col2:
        historical_data = analysis_data.get('historical_data', {})
        start_date = historical_data.get('start_date', 'N/A')
        end_date = historical_data.get('end_date', 'N/A')
        st.info(f"**Period:** {start_date} to {end_date}")
    
    # Download Report Button
    st.markdown("### 📄 Export Results")
    if st.button("📥 Download Analysis Report"):
        track_usage('detailed_report_downloaded')
        create_detailed_download(context, portfolio_value, response)

def display_enhanced_health_analysis(portfolio, analysis_data):
    """Display enhanced portfolio health analysis with 5-factor scoring"""
    st.markdown("### 🏥 Enhanced Portfolio Health Assessment")
    track_usage('enhanced_health_analysis_displayed')
    
    if not portfolio:
        st.warning("No portfolio data available for health analysis")
        return
    
    # Initialize enhanced health monitor
    enhanced_health_monitor = EnhancedPortfolioHealthMonitor()
    health_metrics = enhanced_health_monitor.calculate_portfolio_health(portfolio, analysis_data)
    
    # Health level with emoji
    health_emoji = {
        'Excellent': '🟢',
        'Good': '🟡', 
        'Fair': '🟠',
        'Poor': '🔴'
    }.get(health_metrics.health_level, '🟡')
    
    st.markdown(f"#### {health_emoji} Portfolio Health: {health_metrics.health_level}")
    st.markdown(f"**Overall Health Score: {health_metrics.overall_score:.1f}/100**")
    
    # Enhanced health metrics breakdown with 5 factors
    st.markdown("#### 📊 Professional 5-Factor Health Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Enhanced health score gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_metrics.overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 85], 'color': "lightgreen"},
                    {'range': [85, 100], 'color': "green"}
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
        st.markdown("**Professional 5-Factor Health Breakdown:**")
        
        # Progress bars for each factor
        factors = [
            ("Concentration Risk", health_metrics.concentration_risk, "Position size assessment"),
            ("Correlation Health", health_metrics.correlation_health, "Diversification effectiveness"),
            ("Regime Fitness", health_metrics.regime_fitness, "Market adaptation assessment"),
            ("Stress Resilience", health_metrics.stress_resilience, "Crisis protection level"),
            ("Factor Balance", health_metrics.factor_balance, "Style diversification assessment")
        ]
        
        for factor_name, score, description in factors:
            # Color coding for progress bars
            if score >= 80:
                color = "🟢"
            elif score >= 60:
                color = "🟡"
            else:
                color = "🔴"
            
            st.markdown(f"**{color} {factor_name}**: {score:.0f}/100")
            st.progress(score/100, text=description)
            st.markdown("")  # Add spacing
    
    # Enhanced risk identification
    if health_metrics.key_risks:
        st.markdown("#### ⚠️ Key Health Risks Identified")
        for i, risk in enumerate(health_metrics.key_risks, 1):
            st.markdown(f"{i}. {risk}")
    
    # Enhanced improvement priorities  
    if health_metrics.improvement_priorities:
        st.markdown("#### 🎯 Health Improvement Priorities")
        for i, improvement in enumerate(health_metrics.improvement_priorities, 1):
            st.markdown(f"{i}. {improvement}")
    
    # Professional insights based on score
    st.markdown("#### 💡 Professional Health Insights")
    
    if health_metrics.overall_score >= 85:
        st.success("✅ **Excellent Portfolio Health** - Your portfolio demonstrates professional-grade risk management with strong diversification and appropriate factor balance.")
    elif health_metrics.overall_score >= 70:
        st.info("🟡 **Good Portfolio Health** - Solid foundation with minor optimization opportunities identified.")
    elif health_metrics.overall_score >= 50:
        st.warning("🟠 **Fair Portfolio Health** - Several improvement areas identified that could enhance risk-adjusted returns.")
    else:
        st.error("🔴 **Poor Portfolio Health** - Immediate attention recommended to address concentration and diversification risks.")
    
    # Enhanced recommendations section
    st.markdown("#### 🚀 Enhanced Health Improvement Actions")
    
    # Specific recommendations based on weak areas
    weak_areas = [area for area, score in [
        ('concentration_risk', health_metrics.concentration_risk),
        ('correlation_health', health_metrics.correlation_health),
        ('regime_fitness', health_metrics.regime_fitness),
        ('stress_resilience', health_metrics.stress_resilience),
        ('factor_balance', health_metrics.factor_balance)
    ] if score < 70]
    
    if weak_areas:
        for area in weak_areas:
            if area == 'concentration_risk':
                st.markdown("• **Reduce Concentration**: Consider limiting single positions to <20% of portfolio")
            elif area == 'correlation_health':
                st.markdown("• **Improve Diversification**: Add uncorrelated asset classes (international, bonds, REITs)")
            elif area == 'regime_fitness':
                st.markdown("• **Adjust for Market Regime**: Consider current market conditions in allocation")
            elif area == 'stress_resilience':
                st.markdown("• **Add Defensive Assets**: Include treasury bonds or defensive sectors")
            elif area == 'factor_balance':
                st.markdown("• **Balance Investment Styles**: Mix growth/value and large/small cap exposures")
    else:
        st.markdown("• **Maintain Current Strategy**: Your portfolio health metrics are strong across all factors")
    
    st.markdown("• **Regular Monitoring**: Review health scores monthly and rebalance quarterly")
    st.markdown("• **Professional Consultation**: Consider financial advisor review for complex strategies")

def display_stress_tests(analysis_data, portfolio_value):
    """Display stress testing scenarios"""
    st.markdown("### 🔥 Stress Test Results")
    track_usage('stress_tests_displayed')
    
    stress_tests = analysis_data.get('stress_tests', {})
    
    if stress_tests:
        # Create stress test chart
        scenarios = []
        losses = []
        
        for scenario_name, scenario_data in stress_tests.items():
            scenarios.append(scenario_name.replace('_', ' ').title())
            loss_pct = abs(scenario_data.get('var_95', 0))
            losses.append(loss_pct * 100)  # Convert to percentage
        
        track_usage('stress_test_chart_displayed', {'n_scenarios': len(scenarios)})
        
        fig_stress = go.Figure(data=[
            go.Bar(
                x=scenarios,
                y=losses,
                marker_color=['#e74c3c', '#f39c12', '#3498db'],
                text=[f"{loss:.1f}%" for loss in losses],
                textposition='auto'
            )
        ])
        
        fig_stress.update_layout(
            title="Portfolio Loss in Crisis Scenarios",
            xaxis_title="Scenario",
            yaxis_title="Portfolio Loss (%)",
            height=400
        )
        
        st.plotly_chart(fig_stress, use_container_width=True)
        
        # Stress test summary
        st.markdown("**Stress Test Summary:**")
        for scenario_name, scenario_data in stress_tests.items():
            loss_pct = abs(scenario_data.get('var_95', 0))
            loss_dollar = loss_pct * portfolio_value
            recovery_time = scenario_data.get('recovery_time', 0)
            
            st.write(f"• **{scenario_name.replace('_', ' ').title()}**: {loss_pct:.1%} loss (${loss_dollar:,.0f}), Recovery: {recovery_time:.1f} years")

def create_simple_download(portfolio_input, portfolio_value, response):
    """Create simple text download"""
    track_usage('simple_download_created')
    
    report_text = f"""
PORTFOLIO RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO COMPOSITION
Portfolio: {portfolio_input}
Portfolio Value: ${portfolio_value:,}

ANALYSIS RESULTS
{response}

This report was generated by Portfolio Risk Analyzer MVP
"""
    
    # Create download link
    b64 = base64.b64encode(report_text.encode()).decode()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"portfolio_analysis_{timestamp}.txt"
    
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">📥 Download Report ({filename})</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("✅ Report ready for download!")

def display_analytics_text(analytics):
    """Display analytics as copyable text"""
    
    # Create the same report text as the download
    report_text = f"""
PORTFOLIO RISK ANALYZER - ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {st.session_state.get('session_id', 'Unknown')}

================================================================================
SUMMARY STATISTICS
================================================================================
Total Events Tracked: {len(analytics)}
Session Duration: {(datetime.now() - st.session_state.get('session_start_time', datetime.now())).total_seconds() / 60:.1f} minutes

EVENT TYPE BREAKDOWN:
"""
    
    # Count event types
    event_counts = {}
    feature_requests = []
    errors = []
    feedback_items = []
    
    for event in analytics:
        event_type = event['event_type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Collect specific feedback
        if event_type == 'feature_request' and event.get('portfolio_data'):
            feature_requests.append(event)
        elif event_type == 'analysis_failed' and event.get('error_data'):
            errors.append(event)
        elif event_type in ['positive_feedback', 'negative_feedback']:
            feedback_items.append(event)
    
    # Add event counts
    for event_type, count in sorted(event_counts.items()):
        report_text += f"• {event_type}: {count}\n"
    
    # Add feature requests section
    if feature_requests:
        report_text += f"""

================================================================================
FEATURE REQUESTS ({len(feature_requests)} total)
================================================================================
"""
        for i, request in enumerate(feature_requests, 1):
            timestamp = request['timestamp']
            request_text = request.get('portfolio_data', {}).get('request', 'No text provided')
            report_text += f"{i}. {timestamp}\n   Request: {request_text}\n\n"
    
    # Add detailed event log (last 10 events)
    report_text += f"""

================================================================================
RECENT EVENTS (Last 10)
================================================================================
"""
    
    recent_events = analytics[-10:] if len(analytics) > 10 else analytics
    for i, event in enumerate(recent_events, 1):
        timestamp = event['timestamp']
        event_type = event['event_type']
        report_text += f"{i}. [{timestamp}] {event_type}\n"
    
    # Display in expandable text area
    st.markdown("### 📊 Analytics Report (Copy This Text)")
    st.text_area(
        "Analytics Data",
        value=report_text,
        height=400,
        label_visibility="hidden",
        key="analytics_text_display"
    )
    st.info("💡 Select all text above (Ctrl+A) and copy (Ctrl+C) to save your analytics report")

def create_analytics_download(analytics):
    """Create downloadable analytics report"""
    
    # Create comprehensive analytics report
    report_text = f"""
PORTFOLIO RISK ANALYZER - ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {st.session_state.get('session_id', 'Unknown')}

================================================================================
SUMMARY STATISTICS
================================================================================
Total Events Tracked: {len(analytics)}
Session Duration: {(datetime.now() - st.session_state.get('session_start_time', datetime.now())).total_seconds() / 60:.1f} minutes

EVENT TYPE BREAKDOWN:
"""
    
    # Count event types
    event_counts = {}
    feature_requests = []
    errors = []
    feedback_items = []
    
    for event in analytics:
        event_type = event['event_type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Collect specific feedback
        if event_type == 'feature_request' and event.get('portfolio_data'):
            feature_requests.append(event)
        elif event_type == 'analysis_failed' and event.get('error_data'):
            errors.append(event)
        elif event_type in ['positive_feedback', 'negative_feedback']:
            feedback_items.append(event)
    
    # Add event counts
    for event_type, count in sorted(event_counts.items()):
        report_text += f"• {event_type}: {count}\n"
    
    # Add feature requests section
    if feature_requests:
        report_text += f"""

================================================================================
FEATURE REQUESTS ({len(feature_requests)} total)
================================================================================
"""
        for i, request in enumerate(feature_requests, 1):
            timestamp = request['timestamp']
            request_text = request.get('portfolio_data', {}).get('request', 'No text provided')
            report_text += f"{i}. {timestamp}\n   Request: {request_text}\n\n"
    
    # Add errors section
    if errors:
        report_text += f"""

================================================================================
ERROR ANALYSIS ({len(errors)} total)
================================================================================
"""
        error_types = {}
        for error in errors:
            error_msg = error.get('error_data', {}).get('error_message', 'Unknown error')
            error_type = error.get('error_data', {}).get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        for error_type, count in error_types.items():
            report_text += f"• {error_type}: {count} occurrences\n"
    
    # Add feedback section
    if feedback_items:
        positive = len([f for f in feedback_items if f['event_type'] == 'positive_feedback'])
        negative = len([f for f in feedback_items if f['event_type'] == 'negative_feedback'])
        total_feedback = positive + negative
        satisfaction_rate = (positive / total_feedback * 100) if total_feedback > 0 else 0
        
        report_text += f"""

================================================================================
USER FEEDBACK ANALYSIS
================================================================================
Positive Feedback: {positive}
Negative Feedback: {negative}
Total Feedback: {total_feedback}
Satisfaction Rate: {satisfaction_rate:.1f}%
"""
    
    # Add detailed event log
    report_text += f"""

================================================================================
DETAILED EVENT LOG
================================================================================
"""
    
    for i, event in enumerate(analytics, 1):
        timestamp = event['timestamp']
        event_type = event['event_type']
        duration = event.get('session_duration', 0)
        
        report_text += f"{i}. [{timestamp}] {event_type}"
        
        if duration > 0:
            report_text += f" (Session: {duration:.1f}s)"
        
        # Add specific data for important events
        if event_type == 'analysis_completed':
            portfolio_value = event.get('portfolio_data', {}).get('portfolio_value', 'Unknown')
            analysis_time = event.get('portfolio_data', {}).get('analysis_time', 'Unknown')
            report_text += f" - Portfolio: ${portfolio_value:,}, Time: {analysis_time}s"
        elif event_type == 'feature_request':
            request = event.get('portfolio_data', {}).get('request', '')[:100]
            report_text += f" - Request: {request}..."
        elif event_type == 'analysis_failed':
            error = event.get('error_data', {}).get('error_type', 'Unknown')
            report_text += f" - Error: {error}"
        
        report_text += "\n"
    
    report_text += f"""

================================================================================
INSIGHTS & RECOMMENDATIONS
================================================================================
"""
    
    # Generate insights
    total_analyses_started = event_counts.get('analysis_started', 0)
    total_analyses_completed = event_counts.get('analysis_completed', 0)
    success_rate = (total_analyses_completed / total_analyses_started * 100) if total_analyses_started > 0 else 0
    
    report_text += f"Analysis Success Rate: {success_rate:.1f}%\n"
    
    if success_rate < 70:
        report_text += "⚠️ Consider improving error handling and user guidance\n"
    elif success_rate > 90:
        report_text += "✅ Excellent success rate - users are finding the tool easy to use\n"
    
    if len(feature_requests) > 5:
        report_text += "🚀 High feature request volume - consider prioritizing development\n"
    
    if event_counts.get('positive_feedback', 0) > event_counts.get('negative_feedback', 0):
        report_text += "😊 Positive feedback exceeds negative - good product-market fit signals\n"
    
def create_detailed_download(context, portfolio_value, response):
    """Create detailed analysis download"""
    track_usage('detailed_download_created')
    
    portfolio = context.portfolio
    analysis_data = context.last_analysis
    
    report_text = f"""
COMPREHENSIVE PORTFOLIO RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO COMPOSITION
Total Portfolio Value: ${portfolio_value:,}
"""
    
    if portfolio:
        for ticker, weight in portfolio.items():
            value = weight * portfolio_value
            report_text += f"• {ticker}: {weight:.1%} (${value:,.0f})\n"
    
    base_case = analysis_data.get('base_case', {})
    var_95 = base_case.get('var_95', 0)
    es_95 = base_case.get('es_95', 0)
    
    report_text += f"""

RISK METRICS
• Daily Value-at-Risk (95%): ${abs(var_95) * portfolio_value:,.0f} ({var_95:.1%})
• Expected Shortfall: ${abs(es_95) * portfolio_value:,.0f} ({es_95:.1%})
• Maximum Drawdown: {abs(base_case.get('max_drawdown', 0)):.1%}

STRESS TEST RESULTS
"""
    
    stress_tests = analysis_data.get('stress_tests', {})
    for scenario_name, scenario_data in stress_tests.items():
        loss_pct = abs(scenario_data.get('var_95', 0))
        loss_dollar = loss_pct * portfolio_value
        report_text += f"• {scenario_name.replace('_', ' ').title()}: {loss_pct:.1%} loss (${loss_dollar:,.0f})\n"
    
    report_text += f"""

DATA SOURCE INFORMATION
Data Source: {analysis_data.get('data_source', 'Real Market Data')}
Trading Days: {analysis_data.get('data_period', 'N/A')}
Analysis Period: {analysis_data.get('historical_data', {}).get('start_date', 'N/A')} to {analysis_data.get('historical_data', {}).get('end_date', 'N/A')}

FULL ANALYSIS
{response}

This comprehensive report was generated by Portfolio Risk Analyzer MVP
"""
    
    # Create download link
    b64 = base64.b64encode(report_text.encode()).decode()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"detailed_portfolio_analysis_{timestamp}.txt"
    
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">📥 Download Detailed Report ({filename})</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("✅ Detailed report ready for download!")
    """Create detailed analysis download"""
    track_usage('detailed_download_created')
    
    portfolio = context.portfolio
    analysis_data = context.last_analysis
    
    report_text = f"""
COMPREHENSIVE PORTFOLIO RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO COMPOSITION
Total Portfolio Value: ${portfolio_value:,}
"""
    
    if portfolio:
        for ticker, weight in portfolio.items():
            value = weight * portfolio_value
            report_text += f"• {ticker}: {weight:.1%} (${value:,.0f})\n"
    
    base_case = analysis_data.get('base_case', {})
    var_95 = base_case.get('var_95', 0)
    
    es_95 = base_case.get('es_95', 0)
    
    report_text += f"""

RISK METRICS
• Daily Value-at-Risk (95%): ${abs(var_95) * portfolio_value:,.0f} ({var_95:.1%})
• Expected Shortfall: ${abs(es_95) * portfolio_value:,.0f} ({es_95:.1%})
• Maximum Drawdown: {abs(base_case.get('max_drawdown', 0)):.1%}

STRESS TEST RESULTS
"""
    
    stress_tests = analysis_data.get('stress_tests', {})
    for scenario_name, scenario_data in stress_tests.items():
        loss_pct = abs(scenario_data.get('var_95', 0))
        loss_dollar = loss_pct * portfolio_value
        report_text += f"• {scenario_name.replace('_', ' ').title()}: {loss_pct:.1%} loss (${loss_dollar:,.0f})\n"
    
    report_text += f"""

DATA SOURCE INFORMATION
Data Source: {analysis_data.get('data_source', 'Real Market Data')}
Trading Days: {analysis_data.get('data_period', 'N/A')}
Analysis Period: {analysis_data.get('historical_data', {}).get('start_date', 'N/A')} to {analysis_data.get('historical_data', {}).get('end_date', 'N/A')}

FULL ANALYSIS
{response}

This comprehensive report was generated by Portfolio Risk Analyzer MVP
"""
    
    # Create download link
    b64 = base64.b64encode(report_text.encode()).decode()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"detailed_portfolio_analysis_{timestamp}.txt"
    
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">📥 Download Detailed Report ({filename})</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("✅ Detailed report ready for download!")

if __name__ == "__main__":
    # Run the main app
    main()
