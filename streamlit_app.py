import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- إعداد الصفحة ---
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# --- تحميل الموديل (مع الكاش عشان ميعملش تحميل كل شوية) ---
@st.cache_resource
def load_model():
    model_path = 'ensemble_attrition_model.pkl'
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
OPTIMAL_THRESHOLD = 0.43

# 🛑 قائمة الأعمدة (زي ما هي بالظبط)
FEATURE_COLS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyRate', 
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 
    'BusinessTravel_Travel_Rarely', 'Department_Research & Development', 'Department_Sales', 
    'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 
    'EducationField_Other', 'EducationField_Technical Degree', 'JobRole_Human Resources', 
    'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 
    'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive', 
    'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single'
]

# --- العنوان ---
st.title("👔 Employee Attrition Prediction")
st.markdown("Enter employee details to predict if they are likely to leave the company.")

# --- الفورم (البيانات) ---
with st.form("attrition_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=800)
        distance = st.number_input("Distance From Home", min_value=1, max_value=30, value=5)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        gender = st.radio("Gender", ["Male", "Female"])
        
    with col2:
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=50)
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=30000, value=15000)
        num_companies = st.number_input("Num Companies Worked", min_value=0, max_value=10, value=1)
        over_time = st.radio("Over Time", ["Yes", "No"])

    with col3:
        percent_hike = st.number_input("Percent Salary Hike", min_value=10, max_value=30, value=15)
        perf_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        rel_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        stock_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10)
        training_times = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])

    st.markdown("### Experience & History")
    col4, col5 = st.columns(2)
    with col4:
        years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
        years_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
    with col5:
        years_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
        years_manager = st.number_input("Years With Curr Manager", min_value=0, max_value=20, value=2)

    st.markdown("### Categorical Details")
    col6, col7 = st.columns(2)
    with col6:
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    with col7:
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

    # زرار التنبؤ
    submit_button = st.form_submit_button("🚀 Predict Attrition")

# --- منطق المعالجة والتنبؤ ---
if submit_button:
    if model:
        # 1. تجميع البيانات في Dictionary
        data = {
            'Age': age, 'DailyRate': daily_rate, 'DistanceFromHome': distance, 'Education': education,
            'EnvironmentSatisfaction': env_satisfaction, 'Gender': 1 if gender == "Male" else 0,
            'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement, 'JobLevel': job_level,
            'JobSatisfaction': job_satisfaction, 'MonthlyRate': monthly_rate, 
            'NumCompaniesWorked': num_companies, 'OverTime': 1 if over_time == "Yes" else 0,
            'PercentSalaryHike': percent_hike, 'PerformanceRating': perf_rating,
            'RelationshipSatisfaction': rel_satisfaction, 'StockOptionLevel': stock_level,
            'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times,
            'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_current_role, 'YearsSinceLastPromotion': years_promotion,
            'YearsWithCurrManager': years_manager,
            # سيتم معالجتها بالأسفل
            'BusinessTravel': business_travel, 'Department': department,
            'EducationField': education_field, 'JobRole': job_role, 'MaritalStatus': marital_status
        }

        # 2. إنشاء DataFrame
        data_df = pd.DataFrame([data])

        # 3. One-Hot Encoding (يدوي لضمان التطابق مع الموديل)
        # الطريقة دي أضمن في Streamlit عشان نتفادى مشاكل الـ dummies
        
        # تجهيز الداتا فريم بكل الأعمدة بقيم صفرية
        final_df = pd.DataFrame(0, index=[0], columns=FEATURE_COLS)
        
        # تعبئة القيم الرقمية والمباشرة
        for col in FEATURE_COLS:
            if col in data:
                final_df[col] = data[col]

        # تعبئة الـ One-Hot Encoded يدوياً
        # Business Travel
        if f'BusinessTravel_{business_travel}' in FEATURE_COLS:
            final_df[f'BusinessTravel_{business_travel}'] = 1
            
        # Department
        if f'Department_{department}' in FEATURE_COLS:
            final_df[f'Department_{department}'] = 1
            
        # Education Field
        if f'EducationField_{education_field}' in FEATURE_COLS:
            final_df[f'EducationField_{education_field}'] = 1
            
        # Job Role
        if f'JobRole_{job_role}' in FEATURE_COLS:
            final_df[f'JobRole_{job_role}'] = 1
            
        # Marital Status
        if f'MaritalStatus_{marital_status}' in FEATURE_COLS:
            final_df[f'MaritalStatus_{marital_status}'] = 1

        # 4. التنبؤ
        try:
            probability = model.predict_proba(final_df)[0][1]
            prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
            
            st.divider()
            if prediction == 1:
                st.error(f"⚠️ Prediction: Likely to LEAVE (Attrition: Yes)")
                st.write(f"Probability: **{probability:.2%}**")
            else:
                st.success(f"✅ Prediction: Likely to STAY (Attrition: No)")
                st.write(f"Probability of leaving: **{probability:.2%}**")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.warning("Model is not loaded.")