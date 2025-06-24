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
        "",
        height=80,
        placeholder="e.g., Portfolio optimization suggestions, more asset classes, better visualizations, comparison tools...",
        key="feature_request_input"
    )
    
    if st.button("📝 Submit Feature Request", key="submit_feature") and feature_request.strip():
        track_usage('feature_request', {'request': feature_request.strip()})
        st.success("Feature request recorded! This helps us prioritize development. 🚀")
        

def display_analytics_summary():
    """Display analytics summary for debugging (remove in production)"""
    if st.sidebar.checkbox("🔍 Show Analytics (Debug)", value=False):
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
        
        st.header("📋 Example Portfolios")
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
            placeholder="Enter your portfolio in one of these formats:\n\n• 40% AAPL, 30% MSFT, 20% GOOGL, 10% BND\n• Equal weight SPY QQQ VTI\n• 50% VOO, 50% BND",
            help="Use percentage format or 'Equal weight' followed by ticker symbols"
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
    display_health_analysis(portfolio)
    
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

def display_health_analysis(portfolio):
    """Display portfolio health analysis"""
    st.markdown("### 🏥 Portfolio Health Assessment")
    track_usage('health_analysis_displayed')
    
    if not portfolio:
        st.warning("No portfolio data available for health analysis")
        return
    
    # Calculate health score based on portfolio
    max_weight = max(portfolio.values())
    n_positions = len(portfolio)
    
    concentration_score = max(0, 100 - max_weight * 200)
    diversification_score = min(100, n_positions * 15)
    overall_score = (concentration_score * 0.6 + diversification_score * 0.4)
    
    health_level = "Excellent" if overall_score >= 80 else "Good" if overall_score >= 60 else "Fair" if overall_score >= 40 else "Poor"
    
    # Track health score
    track_usage('health_score_calculated', {
        'health_level': health_level,
        'overall_score': overall_score,
        'max_weight': max_weight,
        'n_positions': n_positions
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Health score gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
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
        st.markdown(f"**Overall Health: {health_level}**")
        st.markdown(f"**Score: {overall_score:.0f}/100**")
        
        # Health metrics breakdown
        st.markdown("**Health Breakdown:**")
        st.progress(concentration_score/100, text=f"Concentration Risk: {concentration_score:.0f}/100")
        st.progress(diversification_score/100, text=f"Diversification: {diversification_score:.0f}/100")
        
        # Key insights
        if max_weight > 0.3:
            st.warning(f"⚠️ High concentration: {max_weight:.1%} in single position")
            track_usage('high_concentration_warning', {'max_weight': max_weight})
        if n_positions < 5:
            st.warning("⚠️ Limited diversification - consider adding more positions")
            track_usage('low_diversification_warning', {'n_positions': n_positions})
        if overall_score >= 70:
            st.success("✅ Portfolio health looks good!")

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

if __name__ == "__main__":
    # Run the main app
    main()
