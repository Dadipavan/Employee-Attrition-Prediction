import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">👥 Employee Attrition Prediction System</h1>', unsafe_allow_html=True)
st.markdown("**Predict employee attrition risk using machine learning models**")

# Check if models exist
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    st.error("❌ Models directory not found! Please run the Jupyter notebook first to train the models.")
    st.info("💡 Run all cells in `Employee_Attrition_Prediction.ipynb` to generate the required model files.")
    st.stop()

# Load models and data
@st.cache_resource
def load_models():
    try:
        models = {}
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and f != 'feature_info.pkl' and f != 'preprocessor.pkl']
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            models[model_name] = joblib.load(os.path.join(MODEL_DIR, model_file))
        
        feature_info = joblib.load(os.path.join(MODEL_DIR, 'feature_info.pkl'))
        
        # Load performance data if available
        performance_data = None
        if os.path.exists(os.path.join(MODEL_DIR, 'model_performance.csv')):
            performance_data = pd.read_csv(os.path.join(MODEL_DIR, 'model_performance.csv'), index_col=0)
        
        return models, feature_info, performance_data
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

models, feature_info, performance_data = load_models()

if models is None:
    st.stop()

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a page:", ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Performance", "ℹ️ About"])

if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to the Employee Attrition Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Purpose")
        st.write("This system helps HR teams identify employees at risk of leaving the company, enabling proactive retention strategies.")
    
    with col2:
        st.markdown("### 🤖 Models Available")
        for model_name in models.keys():
            st.write(f"• {model_name}")
    
    with col3:
        st.markdown("### 📊 Features")
        st.write("• Single employee prediction")
        st.write("• Batch prediction from CSV")
        st.write("• Model performance comparison")
        st.write("• Risk level assessment")
    
    if performance_data is not None:
        st.markdown("### 📈 Model Performance Overview")
        
        # Create performance visualization
        fig = make_subplots(rows=2, cols=2, 
                          subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(x=performance_data.index, y=performance_data[metric], 
                      marker_color=colors[i], name=metric),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Single Prediction":
    st.markdown('<h2 class="sub-header">Single Employee Attrition Prediction</h2>', unsafe_allow_html=True)
    
    # Model selection
    selected_model = st.selectbox("Select Model:", list(models.keys()))
    
    st.markdown("### Employee Information")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Personal Information**")
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=50, value=5)
        
        with col2:
            st.markdown("**Job Information**")
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", [
                "Sales Executive", "Research Scientist", "Laboratory Technician", 
                "Manufacturing Director", "Healthcare Representative", "Manager", 
                "Sales Representative", "Research Director", "Human Resources"
            ])
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        
        with col3:
            st.markdown("**Compensation & Experience**")
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
            hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("**Satisfaction & Work-Life**")
            job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
            environment_satisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])
            work_life_balance = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])
            relationship_satisfaction = st.selectbox("Relationship Satisfaction (1-4)", [1, 2, 3, 4])
        
        with col5:
            st.markdown("**Additional Information**")
            overtime = st.selectbox("Over Time", ["Yes", "No"])
            education = st.selectbox("Education Level (1-5)", [1, 2, 3, 4, 5])
            education_field = st.selectbox("Education Field", [
                "Life Sciences", "Medical", "Marketing", "Technical Degree", 
                "Other", "Human Resources"
            ])
            num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
        
        # Additional fields (with default values for simplicity)
        daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800)
        monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=30000, value=15000)
        percent_salary_hike = st.number_input("Percent Salary Hike (%)", min_value=10, max_value=25, value=15)
        performance_rating = st.selectbox("Performance Rating (1-4)", [1, 2, 3, 4])
        stock_option_level = st.selectbox("Stock Option Level (0-3)", [0, 1, 2, 3])
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=3)
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=2)
        years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20, value=3)
        job_involvement = st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4])
        
        submit_button = st.form_submit_button("🔮 Predict Attrition Risk")
    
    if submit_button:
        # Create employee data
        employee_data = pd.DataFrame({
            'Age': [age],
            'BusinessTravel': [business_travel],
            'DailyRate': [daily_rate],
            'Department': [department],
            'DistanceFromHome': [distance_from_home],
            'Education': [education],
            'EducationField': [education_field],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'Gender': [gender],
            'HourlyRate': [hourly_rate],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobRole': [job_role],
            'JobSatisfaction': [job_satisfaction],
            'MaritalStatus': [marital_status],
            'MonthlyIncome': [monthly_income],
            'MonthlyRate': [monthly_rate],
            'NumCompaniesWorked': [num_companies_worked],
            'OverTime': [overtime],
            'PercentSalaryHike': [percent_salary_hike],
            'PerformanceRating': [performance_rating],
            'RelationshipSatisfaction': [relationship_satisfaction],
            'StockOptionLevel': [stock_option_level],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_times_last_year],
            'WorkLifeBalance': [work_life_balance],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'YearsWithCurrManager': [years_with_curr_manager]
        })
        
        try:
            # Make prediction
            model = models[selected_model]
            prediction = model.predict(employee_data)[0]
            probability = model.predict_proba(employee_data)[0]
            
            # Determine risk level
            attrition_prob = probability[1]
            if attrition_prob > 0.7:
                risk_level = "High"
                risk_class = "high-risk"
                risk_color = "#f44336"
            elif attrition_prob > 0.3:
                risk_level = "Medium"
                risk_class = "medium-risk"
                risk_color = "#ff9800"
            else:
                risk_level = "Low"
                risk_class = "low-risk"
                risk_color = "#4caf50"
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "Will Leave" if prediction == 1 else "Will Stay")
            
            with col2:
                st.metric("Attrition Probability", f"{attrition_prob:.1%}")
            
            with col3:
                st.metric("Risk Level", risk_level)
            
            # Detailed results box
            st.markdown(f"""
            <div class="prediction-box {risk_class}">
                <h3>📊 Detailed Analysis</h3>
                <p><strong>Model Used:</strong> {selected_model}</p>
                <p><strong>Attrition Probability:</strong> {attrition_prob:.1%}</p>
                <p><strong>Retention Probability:</strong> {probability[0]:.1%}</p>
                <p><strong>Risk Assessment:</strong> {risk_level} Risk</p>
                <p><strong>Recommendation:</strong> 
                {'Immediate intervention recommended' if risk_level == 'High' 
                else 'Monitor closely and consider retention strategies' if risk_level == 'Medium'
                else 'Continue regular engagement'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(x=['Will Stay', 'Will Leave'], 
                      y=[probability[0], probability[1]],
                      marker_color=['#4caf50', '#f44336'])
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                xaxis_title="Outcome"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

elif page == "📊 Batch Prediction":
    st.markdown('<h2 class="sub-header">Batch Employee Attrition Prediction</h2>', unsafe_allow_html=True)
    
    # Model selection
    selected_model = st.selectbox("Select Model:", list(models.keys()))
    
    st.markdown("### Upload Employee Data")
    st.info("💡 Upload a CSV file with employee data. The file should contain the same columns as the training data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            st.markdown("### 📋 Data Preview")
            st.dataframe(batch_data.head())
            
            if st.button("🔮 Predict for All Employees"):
                # Make predictions
                model = models[selected_model]
                predictions = model.predict(batch_data)
                probabilities = model.predict_proba(batch_data)
                
                # Add predictions to dataframe
                results_df = batch_data.copy()
                results_df['Attrition_Prediction'] = ['Will Leave' if p == 1 else 'Will Stay' for p in predictions]
                results_df['Attrition_Probability'] = probabilities[:, 1]
                results_df['Risk_Level'] = pd.cut(probabilities[:, 1], 
                                                bins=[0, 0.3, 0.7, 1.0], 
                                                labels=['Low', 'Medium', 'High'])
                
                st.markdown("### 🎯 Prediction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Employees", len(results_df))
                
                with col2:
                    high_risk = len(results_df[results_df['Risk_Level'] == 'High'])
                    st.metric("High Risk", high_risk)
                
                with col3:
                    medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium'])
                    st.metric("Medium Risk", medium_risk)
                
                with col4:
                    low_risk = len(results_df[results_df['Risk_Level'] == 'Low'])
                    st.metric("Low Risk", low_risk)
                
                # Risk distribution chart
                risk_counts = results_df['Risk_Level'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                           title="Risk Level Distribution",
                           color_discrete_map={'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("### 📊 Detailed Results")
                st.dataframe(results_df[['Attrition_Prediction', 'Attrition_Probability', 'Risk_Level']])
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="attrition_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif page == "📈 Model Performance":
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    if performance_data is not None:
        st.markdown("### 📊 Performance Metrics")
        st.dataframe(performance_data.round(4))
        
        # Interactive performance charts
        st.markdown("### 📈 Performance Visualization")
        
        metric_to_plot = st.selectbox("Select Metric to Visualize:", 
                                    ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
        
        fig = px.bar(x=performance_data.index, y=performance_data[metric_to_plot],
                    title=f"{metric_to_plot} by Model",
                    labels={'x': 'Model', 'y': metric_to_plot})
        fig.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = performance_data['ROC-AUC'].idxmax()
        best_score = performance_data.loc[best_model, 'ROC-AUC']
        
        st.success(f"🏆 **Best Model:** {best_model} with ROC-AUC score of {best_score:.4f}")
        
    else:
        st.warning("⚠️ Performance data not available. Please run the Jupyter notebook to generate performance metrics.")

elif page == "ℹ️ About":
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This Employee Attrition Prediction System is designed to help HR departments identify employees 
    who are at risk of leaving the company. By using machine learning models trained on historical 
    employee data, the system can predict the likelihood of attrition for individual employees or 
    entire teams.
    
    ### 🤖 Machine Learning Models
    The system includes multiple trained models:
    - **Logistic Regression**: A linear model that provides interpretable results
    - **Random Forest**: An ensemble method that handles complex patterns
    - **Gradient Boosting**: Advanced boosting technique for high accuracy
    - **Support Vector Machine**: Effective for complex decision boundaries
    - **XGBoost**: State-of-the-art gradient boosting (if available)
    
    ### 📊 Key Features
    - **Single Prediction**: Predict attrition risk for individual employees
    - **Batch Prediction**: Process multiple employees from CSV files
    - **Risk Assessment**: Categorize employees into Low, Medium, and High risk groups
    - **Model Comparison**: Compare performance of different algorithms
    - **Interactive Interface**: User-friendly web application
    
    ### 📈 Business Value
    - **Proactive Retention**: Identify at-risk employees before they leave
    - **Cost Savings**: Reduce recruitment and training costs
    - **Strategic Planning**: Make data-driven HR decisions
    - **Employee Engagement**: Improve workplace satisfaction
    
    ### 🔧 Technical Details
    - **Framework**: Streamlit for web application
    - **ML Library**: Scikit-learn for machine learning
    - **Visualization**: Plotly for interactive charts
    - **Data Processing**: Pandas and NumPy
    
    ### 👥 Final Year Project
    This application was developed as part of a comprehensive Final Year Project on 
    Employee Attrition Prediction using Machine Learning techniques.
    
    ---
    
    **💡 Usage Tips:**
    1. Run the Jupyter notebook first to train models
    2. Use Single Prediction for individual assessments
    3. Use Batch Prediction for team analysis
    4. Monitor high-risk employees closely
    5. Implement retention strategies based on predictions
    """)

# Footer
st.markdown("---")
st.markdown("**Employee Attrition Prediction System** | Final Year Project | Built with ❤️ using Streamlit")