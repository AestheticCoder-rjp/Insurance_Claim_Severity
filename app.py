import streamlit as st
import sys
import os
from streamlit_option_menu import option_menu

# Get the absolute path of the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import the page functions
from overview import run_overview
from eda import run_eda_analysis
from modelling import run_advanced_modeling

# Page Configuration
st.set_page_config(page_title="Insurance Analysis Dashboard", layout="wide")

# Main Header
st.title("\U0001F697 Insurance Data Analysis Dashboard")

# Horizontal Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Overview", "Exploratory Data Analysis", "Modelling"],
    icons=["house", "bar-chart", "clipboard-data", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# Page routing
if selected == "Overview":
    run_overview()
elif selected == "Exploratory Data Analysis":
    run_eda_analysis()

elif selected == "Modelling":
    run_advanced_modeling()
