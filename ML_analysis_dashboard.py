import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import streamlit_card as stc
import altair as alt
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="ML Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more attractive design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background-color: #f5f7ff;
        padding: 20px;
    }
    
    h1, h2, h3 {
        color: #1e3d59;
    }
    
    .stButton>button {
        background-color: #1e3d59;
        color: white;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #276fbf;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: #f0f5ff;
        border-left: 5px solid #276fbf;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    .success-metric {
        border-left: 5px solid #28a745;
    }
    
    .warning-metric {
        border-left: 5px solid #ffc107;
    }
    
    .danger-metric {
        border-left: 5px solid #dc3545;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        padding: 10px 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    
    .tab-button {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background-color: #e5e9f2;
        cursor: pointer;
        font-weight: 500;
    }
    
    .active-tab {
        background-color: white;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    
    .stDataFrame {
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Function to create card UI
def create_card(title, content, icon="📈"):
    st.markdown(f"""
    <div class="card">
        <h3>{icon} {title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Function to create metric display
def display_metric(title, value, status="info"):
    st.markdown(f"""
    <div class="metric-card {status}-metric">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

# Clean data function
def clean_data(df, selected_steps, fill_option=None):
    before_rows = len(df)
    before_nulls = df.isnull().sum().sum()
    
    if "Handle Missing Values" in selected_steps:
        if fill_option == "Fill with Average":
            df = df.fillna(df.mean())
        elif fill_option == "Fill with Forward Fill":
            df = df.fillna(method='ffill')
        elif fill_option == "Drop Rows":
            df = df.dropna()
    
    if "Remove Duplicates" in selected_steps:
        df = df.drop_duplicates()
    
    after_rows = len(df)
    after_nulls = df.isnull().sum().sum()
    
    metrics = {
        "Rows Removed": before_rows - after_rows,
        "Missing Values Fixed": before_nulls - after_nulls,
        "Duplicates Removed": before_rows - after_rows if "Remove Duplicates" in selected_steps else 0
    }
    
    return df, metrics

# Function to generate interactive visualizations
def create_visualization(df, viz_type, x_var, y_var=None, color_var=None):
    if viz_type == "Histogram":
        fig = px.histogram(df, x=x_var, color=color_var, marginal="rug",
                           title=f"Distribution of {x_var}",
                           template="plotly_white")
        return fig
    
    elif viz_type == "Scatter Plot":
        fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                         title=f"{y_var} vs {x_var}",
                         template="plotly_white", 
                         trendline="ols" if y_var else None)
        return fig
    
    elif viz_type == "Box Plot":
        fig = px.box(df, x=x_var, y=y_var, color=color_var,
                     title=f"Box Plot of {y_var} by {x_var}",
                     template="plotly_white")
        return fig
    
    elif viz_type == "Bar Chart":
        fig = px.bar(df, x=x_var, y=y_var, color=color_var,
                     title=f"{y_var} by {x_var}",
                     template="plotly_white")
        return fig
    
    elif viz_type == "Line Chart":
        fig = px.line(df, x=x_var, y=y_var, color=color_var,
                      title=f"{y_var} over {x_var}",
                      template="plotly_white")
        return fig
    
    elif viz_type == "Correlation Heatmap":
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Correlation Matrix",
                        template="plotly_white")
        return fig
    
    elif viz_type == "3D Scatter":
        z_var = st.selectbox("Select Z-axis variable:", df.columns)
        fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var, color=color_var,
                            title=f"3D Plot of {x_var}, {y_var}, and {z_var}",
                            template="plotly_white")
        return fig

# Function for ML models
def run_model(df, model_type, target, features):
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Create prediction plot
        y_pred = model.predict(X_test)
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        fig = px.scatter(results_df, x='Actual', y='Predicted',
                         title=f"Actual vs Predicted {target}",
                         template="plotly_white")
        fig.add_shape(type='line', x0=min(y_test), y0=min(y_test),
                      x1=max(y_test), y1=max(y_test),
                      line=dict(color='red', dash='dash'))
        
        return {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'plot': fig,
            'feature_importance': dict(zip(features, model.coef_)) if len(features) > 1 else {features[0]: model.coef_[0]}
        }
    
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        return {
            'model': model,
            'train_score': train_score,
            'test_score': test_score
        }
    
    elif model_type == "K-Means Clustering":
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = model.fit_predict(X)
        
        if len(features) >= 2:
            fig = px.scatter(df, x=features[0], y=features[1], color='cluster',
                            title=f"K-Means Clustering ({features[0]} vs {features[1]})",
                            template="plotly_white")
        else:
            fig = px.histogram(df, x=features[0], color='cluster',
                              title=f"K-Means Clustering by {features[0]}",
                              template="plotly_white")
        
        return {
            'model': model,
            'plot': fig,
            'cluster_centers': model.cluster_centers_
        }

# Main app function
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>📊 Machine Learning Analysis Dashboard</h1>
        <p>Upload, clean, visualize, and analyze your data with just a few clicks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for navigation
    if 'nav' not in st.session_state:
        st.session_state['nav'] = "Home"
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🧹 Data Cleaning"):
        st.session_state['nav'] = "Data Cleaning"
    if col2.button("📈 Data Visualization"):
        st.session_state['nav'] = "Data Visualization"
    if col3.button("🧠 Data Analysis"):
        st.session_state['nav'] = "Data Analysis"
    if col4.button("🔮 Data Prediction"):
        st.session_state['nav'] = "Data Prediction"
    
    # Home page
    if st.session_state['nav'] == "Home":
        st.markdown("""
        <div class="card">
            <h2>Welcome to the ML Analysis Dashboard!</h2>
            <p>This tool helps you explore and analyze your data through a simple workflow:</p>
            <ol>
                <li><strong>Data Cleaning</strong> - Upload and clean your dataset</li>
                <li><strong>Data Visualization</strong> - Explore your data with interactive charts</li>
                <li><strong>Data Analysis</strong> - Apply machine learning models to find patterns</li>
                <li><strong>Data Prediction</strong> - Make predictions using your trained models</li>
            </ol>
            <p>Click on any of the buttons above to get started!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample dataset section
        st.markdown("""
        <div class="card">
            <h3>🔍 Sample Datasets</h3>
            <p>Don't have a dataset? Try one of these examples:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        if col1.button("Iris Dataset"):
            st.session_state['data'] = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
            st.session_state['nav'] = "Data Cleaning"
        if col2.button("Titanic Dataset"):
            st.session_state['data'] = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
            st.session_state['nav'] = "Data Cleaning"
        if col3.button("Housing Dataset"):
            st.session_state['data'] = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv')
            st.session_state['nav'] = "Data Cleaning"
    
    # Data Cleaning page
    elif st.session_state['nav'] == "Data Cleaning":
        st.markdown("<h2>🧹 Data Cleaning</h2>", unsafe_allow_html=True)
        
        # File uploader if no data in session
        if 'data' not in st.session_state:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is not None:
                st.session_state['data'] = pd.read_csv(uploaded_file)
        
        # If data is available, display and clean
        if 'data' in st.session_state:
            df = st.session_state['data']
            
            # Display data overview
            col1, col2 = st.columns(2)
            with col1:
                create_card("Dataset Information", f"""
                <p>Rows: {df.shape[0]}</p>
                <p>Columns: {df.shape[1]}</p>
                <p>Missing Values: {df.isnull().sum().sum()}</p>
                <p>Duplicates: {df.duplicated().sum()}</p>
                """, icon="📋")
            
            with col2:
                create_card("Data Types", f"""
                <p>Numerical Columns: {len(df.select_dtypes(include=['number']).columns)}</p>
                <p>Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}</p>
                <p>Date Columns: {len(df.select_dtypes(include=['datetime']).columns)}</p>
                """, icon="🔢")
            
            # Display data sample
            with st.expander("Preview Data", expanded=True):
                st.dataframe(df.head(5))
            
            # Cleaning options
            st.markdown("<h3>Select Cleaning Options</h3>", unsafe_allow_html=True)
            cleaning_steps = []
            
            if df.isnull().values.any():
                cleaning_steps.append("Handle Missing Values")
            
            if df.duplicated().any():
                cleaning_steps.append("Remove Duplicates")
            
            selected_steps = st.multiselect("Choose cleaning steps:", cleaning_steps, 
                                           default=cleaning_steps)
            
            fill_option = None
            if "Handle Missing Values" in selected_steps:
                fill_option = st.radio("How to handle missing values:", 
                                      ["Fill with Average", "Fill with Forward Fill", "Drop Rows"])
            
            if st.button("Clean Data"):
                with st.spinner("Cleaning data..."):
                    cleaned_df, metrics = clean_data(df, selected_steps, fill_option)
                    st.session_state['clean_data'] = cleaned_df
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    display_metric("Rows Removed", metrics["Rows Removed"], "warning")
                with col2:
                    display_metric("Missing Values Fixed", metrics["Missing Values Fixed"], "success")
                with col3:
                    display_metric("Duplicates Removed", metrics["Duplicates Removed"], "info")
                
                # Display cleaned data
                st.markdown("<h3>Cleaned Data</h3>", unsafe_allow_html=True)
                st.dataframe(cleaned_df.head(10))
                
                st.success("✅ Data cleaned successfully! You can now proceed to Data Visualization.")
    
    # Data Visualization page
    elif st.session_state['nav'] == "Data Visualization":
        st.markdown("<h2>📈 Data Visualization</h2>", unsafe_allow_html=True)
        
        if 'clean_data' in st.session_state:
            df = st.session_state['clean_data']
            
            # Visualization options
            viz_types = ["Histogram", "Scatter Plot", "Box Plot", "Bar Chart", 
                         "Line Chart", "Correlation Heatmap", "3D Scatter"]
            
            viz_type = st.selectbox("Select Visualization Type:", viz_types)
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X-axis Variable:", df.columns)
            
            with col2:
                if viz_type not in ["Histogram", "Correlation Heatmap"]:
                    y_var = st.selectbox("Select Y-axis Variable:", df.columns)
                else:
                    y_var = None
            
            # Optional color variable
            color_var = st.selectbox("Color by (optional):", 
                                     ["None"] + list(df.select_dtypes(include=['object', 'category']).columns))
            if color_var == "None":
                color_var = None
            
            # Create visualization
            if st.button("Generate Visualization"):
                with st.spinner("Creating visualization..."):
                    fig = create_visualization(df, viz_type, x_var, y_var, color_var)
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state['last_viz'] = fig
            
            # Save visualization option
            if 'last_viz' in st.session_state:
                if st.button("Save Visualization"):
                    st.session_state['saved_viz'] = st.session_state['last_viz']
                    st.success("Visualization saved for the report!")
        else:
            st.warning("Please complete the data cleaning step first.")
            if st.button("Go to Data Cleaning"):
                st.session_state['nav'] = "Data Cleaning"
    
    # Data Analysis page
    elif st.session_state['nav'] == "Data Analysis":
        st.markdown("<h2>🧠 Data Analysis</h2>", unsafe_allow_html=True)
        
        if 'clean_data' in st.session_state:
            df = st.session_state['clean_data']
            
            # Model selection
            model_types = ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
            model_type = st.selectbox("Select Analysis Type:", model_types)
            
            # Target and feature selection
            if model_type != "K-Means Clustering":
                target = st.selectbox("Select Target Variable:", df.columns)
                features = st.multiselect("Select Features:", 
                                         [col for col in df.columns if col != target],
                                         default=[col for col in df.select_dtypes(include=['number']).columns[:3] 
                                                 if col != target])
            else:
                features = st.multiselect("Select Features for Clustering:", 
                                         df.select_dtypes(include=['number']).columns,
                                         default=df.select_dtypes(include=['number']).columns[:2])
                target = None
            
            # Run model
            if st.button("Run Analysis"):
                if not features:
                    st.error("Please select at least one feature.")
                elif model_type != "K-Means Clustering" and target is None:
                    st.error("Please select a target variable.")
                else:
                    with st.spinner("Running analysis..."):
                        results = run_model(df, model_type, target, features)
                        st.session_state['model_results'] = results
                        
                        # Display results
                        if model_type != "K-Means Clustering":
                            col1, col2 = st.columns(2)
                            with col1:
                                display_metric("Training Score", f"{results['train_score']:.4f}", "success")
                            with col2:
                                display_metric("Test Score", f"{results['test_score']:.4f}", 
                                              "success" if results['test_score'] > 0.7 else "warning")
                            
                            if 'feature_importance' in results:
                                st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
                                imp_df = pd.DataFrame({
                                    'Feature': results['feature_importance'].keys(),
                                    'Importance': results['feature_importance'].values()
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(imp_df, x='Feature', y='Importance',
                                            title="Feature Importance",
                                            template="plotly_white")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        if 'plot' in results:
                            st.plotly_chart(results['plot'], use_container_width=True)
                            
                        st.success("Analysis complete! The model is now ready for making predictions.")
        else:
            st.warning("Please complete the data cleaning step first.")
            if st.button("Go to Data Cleaning"):
                st.session_state['nav'] = "Data Cleaning"
    
    # Data Prediction page
    elif st.session_state['nav'] == "Data Prediction":
        st.markdown("<h2>🔮 Data Prediction</h2>", unsafe_allow_html=True)
        
        if 'model_results' in st.session_state and 'clean_data' in st.session_state:
            df = st.session_state['clean_data']
            results = st.session_state['model_results']
            
            st.markdown("""
            <div class="card">
                <h3>Make Predictions</h3>
                <p>Enter values for each feature to get a prediction from your model.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create input fields based on model features
            if 'feature_importance' in results:
                features = list(results['feature_importance'].keys())
                input_values = {}
                
                for feature in features:
                    # Determine a sensible default value
                    if df[feature].dtype in ['int64', 'float64']:
                        default = float(df[feature].mean())
                        input_values[feature] = st.number_input(f"{feature}:", value=default)
                    else:
                        options = df[feature].unique().tolist()
                        input_values[feature] = st.selectbox(f"{feature}:", options)
                
                if st.button("Predict"):
                    # Create input dataframe
                    input_df = pd.DataFrame([input_values])
                    
                    # Make prediction
                    prediction = results['model'].predict(input_df)[0]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="card">
                        <h3>Prediction Result</h3>
                        <div class="metric-card success-metric">
                            <h4>Predicted Value</h4>
                            <h2>{prediction:.4f}</h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("This model type doesn't support direct predictions through this interface.")
        else:
            st.warning("Please run a model analysis first.")
            if st.button("Go to Data Analysis"):
                st.session_state['nav'] = "Data Analysis"

if __name__ == "__main__":
    main()