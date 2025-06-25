#!/usr/bin/env python3
"""
Portfolio Risk Analyzer MVP with Barebones AI Assistant
Simplified version with conversational AI features - FIXED VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# Import your existing portfolio analysis engine
try:
    from real_data_portfolio import RealDataPortfolioAgent
    ENGINE_AVAILABLE = True
except ImportError:
    st.error("Portfolio analysis engine not found. Make sure real_data_portfolio.py is in the same directory.")
    ENGINE_AVAILABLE = False

# ============================================================================
# SIMPLE HEALTH MONITOR
# ============================================================================

class SimpleHealthMonitor:
    """Simple portfolio health scoring for MVP"""
    
    def calculate_health(self, portfolio, analysis_results):
        if not portfolio:
            return {
                'health_score': 0,
                'health_level': 'Poor',
                'main_risk': 'No portfolio provided',
                'recommendation': 'Please provide portfolio holdings'
            }
        
        # Simple concentration risk assessment
        max_weight = max(portfolio.values())
        concentration_penalty = max(0, (max_weight - 0.25) * 150)
        
        # Simple diversification assessment
        n_positions = len(portfolio)
        diversification_bonus = min(40, n_positions * 8)
        
        # Calculate simple health score (0-100)
        health_score = max(0, min(100, 100 - concentration_penalty + diversification_bonus))
        
        # Determine health level
        if health_score >= 80:
            health_level = "Excellent"
        elif health_score >= 65:
            health_level = "Good"
        elif health_score >= 50:
            health_level = "Fair"
        else:
            health_level = "Poor"
        
        # Identify main risk
        if max_weight > 0.4:
            main_risk = f"High concentration: {max_weight:.1%} in single position"
        elif n_positions < 4:
            main_risk = "Limited diversification - consider adding more positions"
        elif max_weight > 0.3:
            main_risk = f"Moderate concentration risk: {max_weight:.1%} in largest position"
        else:
            main_risk = "Risk levels appear well-managed"
        
        # Simple recommendation
        if health_score < 50:
            recommendation = "Consider reducing concentration and adding more diversified holdings"
        elif health_score < 70:
            recommendation = "Good foundation - consider minor diversification improvements"
        else:
            recommendation = "Well-diversified portfolio - maintain current approach"
        
        return {
            'health_score': health_score,
            'health_level': health_level,
            'main_risk': main_risk,
            'recommendation': recommendation
        }

# ============================================================================
# AI ASSISTANT FUNCTIONS - FIXED VERSION
# ============================================================================

def clean_ai_response(response, question):
    """Clean AI response for chat display - IMPROVED VERSION"""
    
    # Remove ALL markdown formatting
    clean = response.replace('**', '').replace('*', '').replace('###', '').replace('##', '')
    clean = clean.replace('📊', '').replace('🔴', '').replace('🟡', '').replace('🟢', '')
    clean = clean.replace('•', '').replace('-', '').replace('✅', '').replace('❌', '').replace('⚠️', '')
    
    # Remove section headers and formatting artifacts
    lines = clean.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, headers, and formatting lines
        if (line and 
            not line.startswith('Risk Level:') and
            not line.startswith('Daily Risk Metrics:') and
            not line.startswith('Portfolio') and
            not line.startswith('Analysis') and
            not line.startswith('VaR') and
            not line.startswith('Expected Shortfall:') and
            not line.startswith('Volatility:') and
            not line.startswith('Risk Assessment:') and
            len(line) > 10):  # Only keep substantial lines
            filtered_lines.append(line)
    
    # Create conversational responses based on question type
    if "risky" in question.lower():
        return create_risk_explanation(filtered_lines)
    elif "health" in question.lower():
        return create_health_explanation(filtered_lines)
    elif "crash" in question.lower():
        return create_crash_explanation(filtered_lines)
    elif "reduce risk" in question.lower():
        return create_risk_reduction_explanation(filtered_lines)
    else:
        # Generic cleanup for other questions
        clean_text = ' '.join(filtered_lines)
        return clean_text[:300] + "..." if len(clean_text) > 300 else clean_text

def create_risk_explanation(filtered_lines):
    """Create conversational risk explanation"""
    
    explanations = [
        "Your portfolio has some concentration that's driving up the risk level. This means you might see bigger swings in value during market volatility.",
        "The main risk factor appears to be having a large portion in similar types of investments. This can amplify both gains and losses.",
        "Your portfolio shows elevated risk primarily due to concentration in certain holdings. This means more potential for significant daily movements.",
        "The risk comes from having significant exposure to assets that tend to move together. When they go down, they often go down together."
    ]
    
    return np.random.choice(explanations)

def create_health_explanation(filtered_lines):
    """Create conversational health explanation"""
    
    explanations = [
        "Your portfolio health score reflects how well-diversified your holdings are. A lower score typically means more concentration risk.",
        "The health assessment looks at factors like concentration, diversification, and overall balance. Your score suggests some areas for improvement.",
        "Portfolio health considers how spread out your investments are and whether you're taking unnecessary concentration risk.",
        "The health score evaluates your portfolio's structure. Room for improvement usually means better diversification could help."
    ]
    
    return np.random.choice(explanations)

def create_crash_explanation(filtered_lines):
    """Create conversational crash scenario explanation"""
    
    explanations = [
        "In a market crash, your portfolio could see significant losses due to its current structure. The stress tests show how much you might lose in different crisis scenarios.",
        "During market stress, concentrated portfolios often experience larger drawdowns. Your portfolio's structure suggests you'd feel the impact significantly.",
        "Market crashes affect different portfolios differently. Based on your holdings, you'd likely see substantial losses during major market stress events.",
        "The stress test results show your portfolio would be vulnerable during crisis periods, mainly due to concentration in certain areas."
    ]
    
    return np.random.choice(explanations)

def create_risk_reduction_explanation(filtered_lines):
    """Create conversational risk reduction explanation"""
    
    explanations = [
        "To reduce risk, consider spreading your investments across more diverse holdings. Adding bonds or different sectors could help stabilize returns.",
        "Risk reduction typically involves diversification - adding different types of investments that don't all move in the same direction.",
        "You could lower portfolio risk by reducing concentration in your largest positions and adding more defensive investments like bonds.",
        "Consider rebalancing to smaller position sizes and adding uncorrelated assets. This helps smooth out the portfolio's overall volatility."
    ]
    
    return np.random.choice(explanations)

def handle_quick_question(question, context, portfolio_value):
    """Handle pre-defined quick questions - FIXED VERSION"""
    
    with st.spinner("🤖 AI is analyzing your question..."):
        try:
            # Use your existing agent with the context that already has analysis
            agent = RealDataPortfolioAgent(force_real_data=True)
            agent.user_contexts["streamlit_user"] = context
            
            # Generate contextual response
            response = agent.process_message("streamlit_user", question)
            
            # Clean up response for chat display
            clean_response = clean_ai_response(response, question)
            
            # Add to chat history
            st.session_state.chat_history.append((question, clean_response))
            
        except Exception as e:
            error_message = f"I'm having trouble accessing that information right now. Error: {str(e)[:100]}..."
            st.session_state.chat_history.append((question, error_message))

def handle_custom_question(question, context, portfolio_value):
    """Handle custom user questions - FIXED VERSION"""
    
    # Basic input validation
    if len(question.strip()) < 5:
        st.warning("Please ask a more specific question (at least 5 characters)")
        return
    
    # Check for inappropriate questions
    inappropriate_keywords = ['buy', 'sell', 'invest in', 'financial advice', 'stock pick']
    if any(keyword in question.lower() for keyword in inappropriate_keywords):
        disclaimer_response = "I can't provide specific investment advice, but I can help you understand your portfolio's risk characteristics and general concepts. Try asking about risk levels, diversification, or 'what if' scenarios instead."
        st.session_state.chat_history.append((question, disclaimer_response))
        return
    
    # Use existing agent
    with st.spinner("🤖 AI is thinking about your question..."):
        try:
            agent = RealDataPortfolioAgent(force_real_data=True)
            agent.user_contexts["streamlit_user"] = context
            
            response = agent.process_message("streamlit_user", question)
            clean_response = clean_ai_response(response, question)
            
            st.session_state.chat_history.append((question, clean_response))
            
        except Exception as e:
            error_message = f"I'm having trouble with that question. Try asking about your portfolio's risk level, health score, or diversification instead."
            st.session_state.chat_history.append((question, error_message))

def display_ai_assistant(context, portfolio_value):
    """Barebones AI assistant for portfolio questions - FIXED VERSION"""
    
    st.markdown("---")
    st.markdown("### 🤖 AI Portfolio Assistant")
    st.markdown("*Ask questions about your analysis - get personalized explanations*")
    
    # Suggested questions
    st.markdown("**📋 Quick Questions:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💡 Why is my portfolio risky?", key="why_risky", use_container_width=True):
            handle_quick_question("Why is my portfolio risky?", context, portfolio_value)
        if st.button("🛡️ How can I reduce risk?", key="reduce_risk", use_container_width=True):
            handle_quick_question("How can I reduce risk?", context, portfolio_value)
    
    with col2:
        if st.button("📊 Explain my health score", key="explain_health", use_container_width=True):
            handle_quick_question("Explain my health score", context, portfolio_value)
        if st.button("💥 What if markets crash?", key="crash_scenario", use_container_width=True):
            handle_quick_question("What if markets crash?", context, portfolio_value)
    
    # Custom question input
    with st.expander("💬 Ask a Custom Question", expanded=False):
        st.markdown("*Examples: 'What happens if I add 20% bonds?' or 'Should I sell some AAPL?'*")
        
        user_question = st.text_input(
            "Your question:",
            placeholder="Ask anything about your portfolio...",
            key="custom_question"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Ask AI", key="ask_custom", type="primary") and user_question:
                handle_custom_question(user_question, context, portfolio_value)
        with col2:
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("**💬 Conversation History:**")
        
        # Show last 5 exchanges
        recent_chats = st.session_state.chat_history[-5:]
        
        for i, (question, answer) in enumerate(recent_chats):
            # User question
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>🙋 You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            # AI response
            st.markdown(f"""
            <div style="background-color: #e8f4fd; padding: 10px; border-radius: 10px; margin: 5px 0 15px 0;">
                <strong>🤖 AI:</strong> {answer}
            </div>
            """, unsafe_allow_html=True)
        
        if len(st.session_state.chat_history) > 5:
            st.info(f"Showing last 5 conversations. Total: {len(st.session_state.chat_history)}")

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP FUNCTIONS - FIXED VERSION
# ============================================================================

def run_portfolio_analysis(portfolio_input, portfolio_value):
    """Run portfolio analysis - FIXED VERSION"""
    
    with st.spinner("🔄 Analyzing your portfolio with real market data..."):
        try:
            # Initialize the analysis engine
            agent = RealDataPortfolioAgent(force_real_data=True)
            
            # Run analysis
            start_time = time.time()
            response = agent.process_message("streamlit_user", portfolio_input)
            analysis_time = time.time() - start_time
            
            # Get the context from the agent
            context = agent.user_contexts.get("streamlit_user")
            
            # STORE ANALYSIS RESULTS IN SESSION STATE
            if context and context.last_analysis:
                st.session_state['analysis_context'] = context
                st.session_state['portfolio_value'] = portfolio_value
                st.session_state['analysis_response'] = response
                st.session_state['analysis_time'] = analysis_time
                st.session_state['has_analysis'] = True
            
            # Display success message
            st.success(f"✅ Analysis completed in {analysis_time:.1f} seconds")
            
            # Display results
            if context and context.last_analysis:
                display_analysis_results(context, portfolio_value)
            else:
                st.markdown("### 📋 Analysis Summary")
                st.markdown(response)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.session_state['has_analysis'] = False
            
            # Provide helpful error guidance
            if "Invalid tickers" in str(e):
                st.info("💡 Ticker Validation Failed - Try verified examples like 'Equal weight SPY QQQ BND'")
            elif "REAL DATA" in str(e):
                st.info("💡 Market Data Issue - Try again in a few moments")

def display_analysis_results(context, portfolio_value):
    """Display analysis results WITH AI assistant - FIXED VERSION"""
    
    analysis_data = context.last_analysis
    portfolio = context.portfolio
    
    # Data source info
    data_source = analysis_data.get('data_source', 'Unknown')
    data_period = analysis_data.get('data_period', 0)
    historical_data = analysis_data.get('historical_data', {})
    start_date = historical_data.get('start_date', 'N/A')
    end_date = historical_data.get('end_date', 'N/A')
    
    st.info(f"**Data Source**: {data_source} ({data_period} trading days: {start_date} to {end_date})")
    
    # Portfolio Overview Section
    st.markdown("### 📈 Portfolio Overview")
    
    if portfolio:
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(portfolio.keys()),
            values=list(portfolio.values()),
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig_pie.update_layout(title="Portfolio Allocation", height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Portfolio breakdown
        st.markdown("**Portfolio Holdings:**")
        for ticker, weight in portfolio.items():
            value = weight * portfolio_value
            concentration_warning = ""
            if weight > 0.4:
                concentration_warning = " 🔴 High concentration"
            elif weight > 0.25:
                concentration_warning = " 🟡 Significant position"
            
            st.write(f"• **{ticker}**: {weight:.1%} (${value:,.0f}){concentration_warning}")
    
    # Risk Metrics Section
    st.markdown("### 📊 Risk Analysis")
    
    base_case = analysis_data.get('base_case', {})
    var_95 = base_case.get('var_95', 0)
    es_95 = base_case.get('es_95', 0)
    max_drawdown = base_case.get('max_drawdown', 0)
    portfolio_returns = base_case.get('portfolio_returns', [])
    
    # Calculate dollar amounts and volatility
    var_dollar = abs(var_95) * portfolio_value
    es_dollar = abs(es_95) * portfolio_value
    volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily VaR (95%)", f"${var_dollar:,.0f}", f"{var_95:.1%}", delta_color="inverse")
    
    with col2:
        st.metric("Expected Shortfall", f"${es_dollar:,.0f}", f"{es_95:.1%}", delta_color="inverse")
    
    with col3:
        st.metric("Annual Volatility", f"{volatility:.1%}", delta_color="off")
    
    with col4:
        st.metric("Max Drawdown", f"{abs(max_drawdown):.1%}", delta_color="inverse")
    
    # Risk Level Assessment
    risk_level = "High" if abs(var_95) > 0.03 else "Moderate" if abs(var_95) > 0.02 else "Low"
    risk_color = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}
    
    st.markdown(f"**Risk Level: {risk_color[risk_level]} {risk_level}**")
    
    # Portfolio Health Section
    st.markdown("### 🏥 Portfolio Health")
    
    health_monitor = SimpleHealthMonitor()
    health = health_monitor.calculate_health(portfolio, analysis_data)
    
    health_emoji = {
        'Excellent': '🟢',
        'Good': '🟡', 
        'Fair': '🟠',
        'Poor': '🔴'
    }.get(health['health_level'], '🟡')
    
    # Display health score with gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health['health_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown(f"#### {health_emoji} {health['health_level']} Health")
        st.markdown(f"**Score: {health['health_score']:.0f}/100**")
        st.markdown(f"**Main Risk:** {health['main_risk']}")
        st.markdown(f"**Recommendation:** {health['recommendation']}")
    
    # Stress Testing Section
    st.markdown("### 🔥 Stress Test Results")
    
    stress_tests = analysis_data.get('stress_tests', {})
    
    if stress_tests:
        scenarios = []
        losses = []
        
        for scenario_name, scenario_data in stress_tests.items():
            scenarios.append(scenario_name.replace('_', ' ').title())
            loss_pct = abs(scenario_data.get('var_95', 0))
            losses.append(loss_pct * 100)
        
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
            xaxis_title="Crisis Scenario",
            yaxis_title="Portfolio Loss (%)",
            height=400
        )
        
        st.plotly_chart(fig_stress, use_container_width=True)
        
        # Stress test summary
        st.markdown("**Crisis Scenario Summary:**")
        for scenario_name, scenario_data in stress_tests.items():
            loss_pct = abs(scenario_data.get('var_95', 0))
            loss_dollar = loss_pct * portfolio_value
            
            st.write(f"• **{scenario_name.replace('_', ' ').title()}**: {loss_pct:.1%} loss (${loss_dollar:,.0f})")
    
    # AI ASSISTANT SECTION
    display_ai_assistant(context, portfolio_value)
    
    # Summary
    st.markdown("### 🎯 Summary")
    
    if health['health_score'] < 60:
        st.warning("**Consider**: Reducing concentration and improving diversification")
    elif health['health_score'] < 80:
        st.info("**Consider**: Fine-tuning allocation for better risk management")
    else:
        st.success("**Status**: Portfolio shows good risk management practices")
    
    st.markdown("**💡 Tip**: Use the AI assistant above to ask specific questions about your analysis!")

def main():
    """Main application function - FIXED VERSION"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Portfolio Risk Analyzer</h1>
        <p>Professional portfolio analysis with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if engine is available
    if not ENGINE_AVAILABLE:
        st.error("❌ Portfolio analysis engine not available.")
        st.stop()
    
    # Initialize session state
    if 'has_analysis' not in st.session_state:
        st.session_state['has_analysis'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Quick Examples")
        
        if st.button("Conservative Mix", use_container_width=True):
            st.session_state.portfolio_input = "50% SPY, 30% BND, 20% VTI"
            st.session_state.chat_history = []
            st.session_state['has_analysis'] = False
            st.rerun()
            
        if st.button("Tech Growth", use_container_width=True):
            st.session_state.portfolio_input = "40% AAPL, 30% MSFT, 20% GOOGL, 10% AMZN"
            st.session_state.chat_history = []
            st.session_state['has_analysis'] = False
            st.rerun()
            
        if st.button("Balanced ETF", use_container_width=True):
            st.session_state.portfolio_input = "Equal weight SPY QQQ BND"
            st.session_state.chat_history = []
            st.session_state['has_analysis'] = False
            st.rerun()
        
        st.markdown("---")
        st.info("**AI-Powered Portfolio Analysis** with real market data and personalized AI explanations!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Enter Your Portfolio")
        
        portfolio_input = st.text_area(
            "Portfolio Holdings",
            value=st.session_state.get('portfolio_input', ''),
            height=120,
            placeholder="Enter your portfolio:\n\n• 40% AAPL, 30% MSFT, 20% GOOGL, 10% BND\n• Equal weight SPY QQQ VTI\n• 50% VOO, 50% BND",
            help="Use percentage format or 'Equal weight' followed by ticker symbols"
        )
        
        # Clear analysis if portfolio changes
        if portfolio_input != st.session_state.get('last_portfolio_input', ''):
            st.session_state.chat_history = []
            st.session_state['has_analysis'] = False
            st.session_state.last_portfolio_input = portfolio_input
        
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=1000,
            value=1000000,
            step=10000,
            format="%d"
        )
        
        analyze_button = st.button("🚀 Analyze Portfolio", type="primary", use_container_width=True)
        
        if portfolio_input:
            if "%" in portfolio_input:
                st.success("✅ Percentage format detected")
            elif "equal" in portfolio_input.lower():
                st.success("✅ Equal weight format detected")
            else:
                st.warning("⚠️ Format not recognized - try examples above")
    
    with col2:
        st.header("📊 Analysis Results")
        
        if analyze_button and portfolio_input:
            run_portfolio_analysis(portfolio_input, portfolio_value)
        elif st.session_state.get('has_analysis', False):
            # Show existing analysis results
            context = st.session_state['analysis_context']
            portfolio_value = st.session_state['portfolio_value']
            st.success(f"✅ Analysis completed in {st.session_state['analysis_time']:.1f} seconds")
            display_analysis_results(context, portfolio_value)
        else:
            st.info("👈 **Enter your portfolio to get started** with AI-powered analysis!")

if __name__ == "__main__":
    main()
