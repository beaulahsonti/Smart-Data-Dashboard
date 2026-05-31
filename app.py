import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

# ==========================================
# STEP 1: PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Smart AI Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark/modern dashboard CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Premium Linear Gradient Header Container */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .header-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.95;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* Glassmorphism Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2.5rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        border-color: rgba(75, 108, 183, 0.4);
        background: rgba(75, 108, 183, 0.02);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4b6cb7;
        margin-bottom: 0.25rem;
        background: -webkit-linear-gradient(45deg, #4b6cb7, #182848);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    /* AI Response Premium Container */
    .ai-response-box {
        background: rgba(75, 108, 183, 0.04);
        border-left: 5px solid #4b6cb7;
        padding: 1.75rem;
        border-radius: 0 16px 16px 0;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(75, 108, 183, 0.1);
        border-right: 1px solid rgba(75, 108, 183, 0.1);
        border-bottom: 1px solid rgba(75, 108, 183, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 2.5rem;
        font-size: 0.85rem;
        color: #a0aec0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 5rem;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# STEP 2: DATA INGESTION & CACHING
# ==========================================
@st.cache_data(show_spinner="Analyzing and ingestion dataset...")
def load_data(file_bytes, file_name):
    """
    Safely load CSV or Excel data into a Pandas DataFrame.
    Decorated with @st.cache_data to prevent repetitive disk reads.
    """
    if file_name.endswith('.csv'):
        # CSV parsing
        return pd.read_csv(io.BytesIO(file_bytes))
    elif file_name.endswith(('.xlsx', '.xls')):
        # Excel parsing using openpyxl
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported format. Please upload a .csv or .xlsx file.")


# ==========================================
# STEP 4: AI INTEGRATION FUNCTION
# ==========================================
def query_gemini(api_key, df_summary_str, df_preview_str, columns_info, user_question):
    """
    Query the Google Gemini generative model with the statistical summary and the user's prompt.
    """
    if not api_key:
        return "⚠️ **Gemini API Key is missing.** Please provide a valid key in the sidebar."
    
    if not api_key.startswith("AIzaSy"):
        return "❌ **Invalid API Key format.** The Google Gemini API key should start with 'AIzaSy'. Please verify your key."

    try:
        # Configure Gemini client
        genai.configure(api_key=api_key)
        
        # Instantiate 1.5 Flash for rapid, accurate data analysis
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
You are a highly skilled Senior Data Analyst and Business Intelligence Expert.
Analyze the provided dataset summary statistics and answer the user's specific business or technical question.

---
### Dataset Summary Statistics (df.describe()):
{df_summary_str}

### Dataset Schema & Column Details:
{columns_info}

### Dataset Structure (First 5 Rows):
{df_preview_str}

---
### User's Question:
"{user_question}"

---
Please organize your analysis response using beautiful, structured Markdown:
1. **📊 Executive Summary / Direct Answer**: Answer the question directly and concisely.
2. **💡 Key Analytical Observations**: Highlight 3-4 interesting trends, distribution characteristics, or key patterns visible in the statistics (e.g., standard deviation, ranges, skewed indicators).
3. **🛠️ Actionable Recommendations**: Suggest concrete next steps or decisions based on this analytical data.
4. **🔍 Recommended Follow-Up Queries**: Suggest 2 relevant analytical questions the user might ask next.

Make the response engaging, professional, and visually stunning using clear typography, emojis, bullet points, and clean bolding.
"""
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ **Error occurred while calling Google Gemini API:** {str(e)}\n\nPlease ensure your API Key is correct, has proper billing/quota allocations, and that your internet connection is active."


# ==========================================
# SIDEBAR SETUP
# ==========================================
st.sidebar.markdown("### 📊 Navigation & Settings")
st.sidebar.markdown("Configure your dataset source and authentication credentials below.")

# API Key input section
st.sidebar.subheader("🔑 API Credentials")
api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    placeholder="AIzaSy...",
    help="Grab your API Key from Google AI Studio (https://aistudio.google.com/)"
)

# File uploader sidebar widget
st.sidebar.subheader("📥 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Accepts .csv, .xlsx, and .xls format spreadsheet files."
)

# Pre-packaged Demo Data option for instant premium testing
load_demo = st.sidebar.checkbox(
    "💡 Use Sample Dataset",
    value=False,
    help="Enable this to explore the dashboard immediately without uploading a file."
)


# ==========================================
# MAIN CONTAINER - TITLE HEADER
# ==========================================
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📊 Smart AI Data Dashboard</h1>
        <p class="header-subtitle">
            Ingest structured spreadsheets, generate premium custom visualizations instantly, 
            and tap into state-of-the-art Google Gemini AI models for automatic statistical insights.
        </p>
    </div>
""", unsafe_allow_html=True)


# Load dataset based on user input (Upload vs. Demo)
df = None
file_name_label = ""

if uploaded_file is not None:
    try:
        # Load the bytes content safely
        file_bytes = uploaded_file.read()
        file_name_label = uploaded_file.name
        df = load_data(file_bytes, file_name_label)
    except Exception as err:
        st.error(f"⚠️ **Error reading file '{uploaded_file.name}':** {str(err)}")
        st.info("💡 Please verify the spreadsheet is not corrupted and contains standard rows and columns.")
elif load_demo:
    # Beautiful Mock Analytics Dataset for Instant Verification
    demo_data = {
        "Sales ($)": [1250, 3400, 2900, 4800, 1500, 6200, 5300, 4100, 7100, 8000, 9500, 3100],
        "Customers": [45, 120, 95, 150, 50, 210, 180, 130, 240, 280, 310, 105],
        "Store Rating": [4.2, 4.7, 3.9, 4.5, 4.0, 4.8, 4.6, 4.3, 4.9, 4.7, 4.8, 4.1],
        "Operational Costs ($)": [800, 1500, 1100, 2000, 900, 2500, 2200, 1800, 3000, 3500, 4000, 1300]
    }
    df = pd.DataFrame(demo_data)
    file_name_label = "Demo Store Performance Dataset"
    st.info("ℹ️ Using the pre-packaged sample dataset. You can toggle this off in the sidebar to upload your own files.")


# ==========================================
# MAIN CONTAINER - BUSINESS LOGIC
# ==========================================
if df is not None:
    # ------------------------------------------
    # DATA METRICS & SUMMARY
    # ------------------------------------------
    st.subheader("📋 Dataset Overview")
    
    # Identify numeric and string columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Render premium glassmorphism metric cards
    st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(numeric_cols)}</div>
                <div class="metric-label">Numeric Attributes</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(categorical_cols)}</div>
                <div class="metric-label">Categorical Attributes</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Interactive dataframe preview (First 5 rows)
    st.markdown("### 🔍 Dataset Preview (First 5 Rows)")
    st.dataframe(df.head(5), use_container_width=True)
    
    # ------------------------------------------
    # STEP 3: AUTOMATED VISUALIZATIONS
    # ------------------------------------------
    st.markdown("---")
    st.subheader("📈 Automated Visualizations")
    
    if len(numeric_cols) > 0:
        col_select_container, chart_type_container = st.columns([2, 1])
        
        with col_select_container:
            selected_col = st.selectbox(
                "Select a numeric column to analyze:",
                options=numeric_cols,
                help="The chart will update dynamically to represent the distribution and spread of this variable."
            )
            
        with chart_type_container:
            chart_type = st.selectbox(
                "Select Chart Type:",
                options=["Histogram (with Box Plot)", "Simple Histogram", "Violin/Distribution Plot", "Trend Line"],
                help="Different visualizations help highlight distinct shapes and properties in your columns."
            )
        
        # Render dynamic Plotly chart based on choice
        try:
            if chart_type == "Histogram (with Box Plot)":
                fig = px.histogram(
                    df, 
                    x=selected_col, 
                    marginal="box",
                    title=f"Detailed Spread and Outliers of: {selected_col}",
                    color_discrete_sequence=["#2a5298"],
                    template="plotly_white"
                )
            elif chart_type == "Simple Histogram":
                fig = px.histogram(
                    df, 
                    x=selected_col, 
                    title=f"Histogram of: {selected_col}",
                    color_discrete_sequence=["#4b6cb7"],
                    template="plotly_white"
                )
            elif chart_type == "Violin/Distribution Plot":
                fig = px.violin(
                    df, 
                    y=selected_col, 
                    box=True, 
                    points="all",
                    title=f"Violin Spread of: {selected_col}",
                    color_discrete_sequence=["#182848"],
                    template="plotly_white"
                )
            else:
                fig = px.line(
                    df, 
                    y=selected_col,
                    title=f"Trend Sequence of: {selected_col}",
                    color_discrete_sequence=["#4b6cb7"],
                    template="plotly_white"
                )
                
            # Elegant styling matching modern themes
            fig.update_layout(
                margin=dict(l=40, r=40, t=60, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.02)",
                font=dict(family="Plus Jakarta Sans", size=12),
                title=dict(font=dict(size=18, family="Plus Jakarta Sans")),
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as plot_err:
            st.error(f"⚠️ **Could not generate chart for {selected_col}:** {str(plot_err)}")
    else:
        st.warning("⚠️ No numeric attributes found in the dataset to visualize.")

    # ------------------------------------------
    # STEP 4: AI INTEGRATION ("Ask the AI")
    # ------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Ask the AI Assistant")
    st.markdown(
        "Leverage the reasoning capabilities of **Google Gemini** to extract instant insights from your statistical parameters."
    )
    
    # Statistical summary strings to pass to prompt
    # Check if the dataframe exists and is not empty before describing it
    if df is not None and not df.empty and len(df.columns) > 0:
        df_summary = df.describe().to_string()
    # Proceed with your Gemini API call using df_summary here...
        df_summary = df.describe().to_string()

        df_preview = df.head().to_string()

        columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
    # Pre-crafted common questions for fast user selection
    example_questions = [
        "Select an example question or write your own below...",
        "Perform a comprehensive statistical health check and summary of the data.",
        "What are the key insights and takeaways we can draw from the descriptive statistics?",
        "Are there any signs of outliers, skewness, or strange distributions?",
        "If this represented business performance data, what would be your top recommendations?"
    ]
    
    selected_example = st.selectbox(
        "💡 Quick Prompt Templates", 
        options=example_questions,
        help="Select any typical data question to populate the question box automatically."
    )
    
    # Keep query box synchronized with selection
    default_text = ""
    if selected_example != example_questions[0]:
        default_text = selected_example
        
    user_question = st.text_area(
        "Ask a question about this dataset:",
        value=default_text,
        height=100,
        placeholder="e.g., Which metric shows the most fluctuation, and does it correlate with operational costs?"
    )
    
    # Trigger Gemini Action
    if st.button("🚀 Analyze with Gemini AI", use_container_width=True):
        if not api_key:
            st.error("🔑 **API Key Required:** Please provide your Google Gemini API Key in the sidebar to run the analysis.")
        elif not user_question.strip():
            st.warning("❓ **Empty Question:** Please type a question or select an template prompt from above.")
        else:
            with st.spinner("Analyzing statistics & generating your expert response..."):
                response_text = query_gemini(
                    api_key=api_key, 
                    df_summary_str=df_summary, 
                    df_preview_str=df_preview, 
                    columns_info=columns_info, 
                    user_question=user_question
                )
                
                # Render inside a beautiful modern AI block
                st.markdown("### ✨ AI Analyst Insights")
                st.markdown(
                    f'<div class="ai-response-box">{response_text}</div>', 
                    unsafe_allow_html=True
                )
                
else:
    # ------------------------------------------
    # NO FILE UPLOADED STATE (Premium Welcome Page)
    # ------------------------------------------
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: rgba(255,255,255,0.01); border: 1px dashed rgba(255,255,255,0.1); border-radius: 16px; margin-top: 2rem;">
            <span style="font-size: 5rem;">📥</span>
            <h2 style="font-weight: 700; margin-top: 1rem; color: #4b6cb7;">Awaiting Your Dataset</h2>
            <p style="color: #718096; max-width: 600px; margin: 0.5rem auto 2rem auto; font-size: 1.05rem; line-height: 1.6;">
                Upload a CSV or Excel (.xlsx/.xls) spreadsheet in the sidebar to visualize parameters, inspect columns, and get AI-powered insights on your metrics instantly.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Quick Start Checklist")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔐 **1. Configure API Key**\nEnter your Google Gemini API key securely in the sidebar password box.")
    with col2:
        st.info("📊 **2. Drop a Data Sheet**\nUpload any table structured file or simply tick the 'Use Sample Dataset' box to test immediately.")
    with col3:
        st.info("🤖 **3. Query the Analyst**\nAsk custom questions about trends, distributions, anomalies, or business conclusions.")

# Footer element
st.markdown("""
    <div class="footer">
        Smart AI Data Dashboard • Crafted with Streamlit, Plotly & Google Gemini
    </div>
""", unsafe_allow_html=True)
