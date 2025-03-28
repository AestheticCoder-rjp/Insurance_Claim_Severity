import streamlit as st
import sys
import os
from streamlit_option_menu import option_menu

# Page Configuration
st.set_page_config(
    page_title="Insurance Intelligence", 
    page_icon="",
    layout="wide"
)

# Professional Color Palette and Modern CSS with Enhanced Animations
st.markdown("""
<style>
    /* Modern Color Palette - Sleek and Professional */
    :root {
        --primary-color: #1A5F7A;       /* Deep Teal */
        --secondary-color: #159895;     /* Vibrant Teal */
        --accent-color: #57C5B6;        /* Soft Aqua */
        --background-color: #F8F9FA;    /* Light Gray-White */
        --text-primary: #2C3E50;        /* Dark Charcoal */
        --text-secondary: #6C757D;      /* Muted Gray */
        --white-background: #FFFFFF;    /* Pure White */
        --border-color: #DEE2E6;        /* Light Gray Border */
    }

    /* Global Styling with Smooth Transitions */
    .stApp {
        background-color: var(--background-color);
        font-family: 'Inter', 'Roboto', 'Segoe UI', sans-serif;
        color: var(--text-primary);
        transition: all 0.5s ease-in-out;
    }

    /* Animated Title Container */
    .app-title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px 0;
        margin-bottom: 20px;
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInUp 1s forwards;
    }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .app-title {
        font-size: 2.8em;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        letter-spacing: -1px;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: all 0.5s ease;
    }

    .app-title:hover {
        transform: scale(1.02);
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.2em;
        margin-top: 10px;
        opacity: 0.8;
    }

    /* Sidebar with Glassmorphism Effect */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 0, 0, 0.05);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border-radius: 0 10px 10px 0;
    }

    /* Sidebar Menu Styling with Hover Effects */
    .css-qbe2x1 {
        color: var(--text-primary) !important;
        font-weight: 500;
        transition: all 0.3s ease;
        border-radius: 8px;
        padding: 8px 12px;
        position: relative;
        overflow: hidden;
    }

    .css-qbe2x1::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 0;
        height: 2px;
        background-color: var(--accent-color);
        transition: width 0.3s ease;
    }

    .css-qbe2x1:hover::before {
        width: 100%;
    }

    .css-qbe2x1:hover {
        background-color: rgba(85, 197, 182, 0.1);
        color: var(--primary-color) !important;
        transform: translateX(5px);
    }

    .css-qbe2x1[data-selected="true"] {
        background-color: rgba(85, 197, 182, 0.2);
        color: var(--primary-color) !important;
        font-weight: 600;
    }

    /* Animated Navigation Icons */
    .nav-icon {
        color: var(--accent-color) !important;
        margin-right: 10px;
        transition: transform 0.3s ease, color 0.3s ease;
    }

    .nav-icon:hover {
        transform: rotate(15deg) scale(1.2);
        color: var(--primary-color) !important;
    }

    /* Footer with Subtle Animation */
    .footer {
        background-color: var(--white-background);
        border-top: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 15px;
        text-align: center;
        font-size: 0.9em;
        box-shadow: 0 -4px 6px rgba(0,0,0,0.02);
        transition: all 0.3s ease;
    }

    .footer:hover {
        background-color: rgba(85, 197, 182, 0.05);
    }

    /* Animated Buttons */
    .stButton > button {
        background-color: var(--accent-color) !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: all 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        background-color: var(--primary-color) !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2.2em;
        }
        .app-subtitle {
            font-size: 1em;
        }
    }
</style>
""", unsafe_allow_html=True)

# Rest of the code remains the same as in the original script
# Project directory setup
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import page functions
from overview import run_overview
from eda import run_eda_analysis
from modelling import run_advanced_modeling

# Main App
def main():
    # Spacious Custom Title
    st.markdown("""
    <div class="app-title-container">
        <div>
            <h1 class="app-title">AutoShield Intelligence</h1>
            <p class="app-subtitle">Comprehensive Auto Insurance Analytics Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("---")
        
        # Navigation Menu
        selected = option_menu(
            menu_title="Navigation",
            options=["Overview", "Vehicle Analysis", "Risk Assessment"],
            icons=["car-front", "bar-chart-fill", "shield-check"],
            menu_icon="menu-button-wide",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#3498DB", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "0px 5px",
                    "color": "#2C3E50",
                    "transition": "all 0.3s ease"
                },
                "nav-link-selected": {
                    "background-color": "rgba(44, 62, 80, 0.1)",
                    "color": "#2C3E50",
                    "font-weight": "600"
                },
            }
        )
    
    # Page Routing
    if selected == "Overview":
        run_overview()
    elif selected == "Vehicle Analysis":
        run_eda_analysis()
    elif selected == "Risk Assessment":
        run_advanced_modeling()

    # Footer
    st.markdown("""
    <div class="footer">
        © 2024 AutoShield Intelligence | Comprehensive Auto Insurance Analytics
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()