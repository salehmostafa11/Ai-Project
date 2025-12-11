import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- إعداد الصفحة ---
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# --- تحميل الموديل ---
@st.cache_resource
def load_model():
    if os.path.exists('ensemble_attrition_model.pkl'):
        return joblib.load('ensemble_attrition_model.pkl')
    return None

model = load_model()
OPTIMAL_THRESHOLD = 0.43

# 🛑 قائمة الأعمدة (نفس ترتيب الموديل بالظبط)
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

if not model:
    st.error("❌ Model file not found! Please make sure 'ensemble_attrition_model.pkl' is in the repo.")
    st.stop()

# --- الفورم (تجميع البيانات) ---
with st.form("attrition_form"):
    col1, col2, col3 = st.columns(3)
    
    # نستخدم نفس أسماء الـ HTML Form القديمة عشان نستخدم نفس دالة المعالجة
    with col1:
        age = st.number_input("Age", 18, 80, 30)
        daily_rate = st.number_input("Daily Rate", 100, 2000, 800)
        distance = st.number_input("Distance From Home", 1, 30, 5)
        education = st.selectbox("Education", [1, 2, 3, 4, 5])
        env_sat = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        gender = st.radio("Gender", ["Male", "Female"])
        
    with col2:
        hourly_rate = st.number_input("Hourly Rate", 30, 100, 50)
        job_inv = st.selectbox("Job Involvement", [1, 2, 3, 4])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_sat = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        monthly_rate = st.number_input("Monthly Rate", 2000, 30000, 15000)
        num_comp = st.number_input("Number of Companies Worked in", 0, 10, 1) # الاسم القديم
        over_time = st.radio("Over Time", ["Yes", "No"])

    with col3:
        percent_hike = st.number_input("Percent Salary Hike", 10, 30, 15)
        perf_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        rel_sat = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        stock_opt = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_years = st.number_input("Total Working Years", 0, 40, 10)
        training_times = st.number_input("Training Times Last Year", 0, 6, 2)
        work_life = st.selectbox("Work Life Balance", [1, 2, 3, 4])

    st.markdown("### 📅 Experience")
    c4, c5 = st.columns(2)
    with c4:
        years_comp = st.number_input("Years At Company", 0, 40, 5)
        years_role = st.number_input("Years In Current Role", 0, 20, 2)
    with c5:
        years_promo = st.number_input("Years Since Last Promotion", 0, 20, 1)
        years_manager = st.number_input("Years With Curr Manager", 0, 20, 2)

    st.markdown("### 📋 Categorical")
    c6, c7 = st.columns(2)
    with c6:
        bus_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        dept = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        edu_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    with c7:
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

    submit = st.form_submit_button("🚀 Predict")

# --- دالة المعالجة (نفس منطق Flask بالظبط) ---
def preprocess_data(input_dict):
    df = pd.DataFrame([input_dict])
    
    # 1. توحيد الأسماء (نفس القاموس اللي نجح في Flask)
    rename_map = {
        'Daily Rate': 'DailyRate',
        'Distance From Home': 'DistanceFromHome',
        'Environment Satisfaction': 'EnvironmentSatisfaction',
        'Hourly Rate': 'HourlyRate',
        'Job Involvement': 'JobInvolvement',
        'Job Level': 'JobLevel',
        'Job Satisfaction': 'JobSatisfaction',
        'Monthly Rate': 'MonthlyRate',
        'Number of Companies Worked in': 'NumCompaniesWorked', # تصحيح الاسم
        'Percent Salary Hike': 'PercentSalaryHike',
        'Performance Rating': 'PerformanceRating',
        'Relationship Satisfaction': 'RelationshipSatisfaction',
        'Stock Option Level': 'StockOptionLevel',
        'Total Working Years': 'TotalWorkingYears',
        'Training Times Last Year': 'TrainingTimesLastYear',
        'Work Life Balance': 'WorkLifeBalance',
        'Years At Company': 'YearsAtCompany',
        'Years In Current Role': 'YearsInCurrentRole',
        'Years Since Last Promotion': 'YearsSinceLastPromotion',
        'Years With Curr Manager': 'YearsWithCurrManager',
        'Over Time': 'OverTime',
        'Business Travel': 'BusinessTravel',
        'Education Field': 'EducationField',
        'Job Role': 'JobRole',
        'Marital Status': 'MaritalStatus'
    }
    df = df.rename(columns=rename_map)
    
    # 2. Binary Encoding
    binary_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    if 'Gender' in df.columns: df['Gender'] = df['Gender'].map(binary_map)
    if 'OverTime' in df.columns: df['OverTime'] = df['OverTime'].map(binary_map)
    
    # 3. One Hot Encoding
    ohe_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], prefix_sep='_')
    
    # 4. تنظيف الأسماء (لأن pd.get_dummies أحياناً بتسيب مسافات)
    # الموديل متدرب على 'Department_Research & Development' (بمسافات)، فمش هنشيلها
    
    # 5. Reindexing (أهم خطوة)
    final_df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return final_df

if submit:
    # تجميع الداتا بنفس مفاتيح الـ Form القديمة
    raw_data = {
        'Age': age, 'Daily Rate': daily_rate, 'Distance From Home': distance, 'Education': education,
        'Environment Satisfaction': env_sat, 'Gender': gender, 'Hourly Rate': hourly_rate,
        'Job Involvement': job_inv, 'Job Level': job_level, 'Job Satisfaction': job_sat,
        'Monthly Rate': monthly_rate, 'Number of Companies Worked in': num_comp,
        'Over Time': over_time, 'Percent Salary Hike': percent_hike, 'Performance Rating': perf_rating,
        'Relationship Satisfaction': rel_sat, 'Stock Option Level': stock_opt,
        'Total Working Years': total_years, 'Training Times Last Year': training_times,
        'Work Life Balance': work_life, 'Years At Company': years_comp,
        'Years In Current Role': years_role, 'Years Since Last Promotion': years_promo,
        'Years With Curr Manager': years_manager, 'Business Travel': bus_travel,
        'Department': dept, 'Education Field': edu_field, 'Job Role': job_role,
        'Marital Status': marital
    }
    
    final_df = preprocess_data(raw_data)
    
    try:
        prob = model.predict_proba(final_df)[0][1]
        pred = 1 if prob >= OPTIMAL_THRESHOLD else 0
        
        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if pred == 1:
                st.error("### ⚠️ Likely to LEAVE")
            else:
                st.success("### ✅ Likely to STAY")
        
        with col_res2:
            st.metric("Attrition Probability", f"{prob:.2%}", delta_color="inverse")
            st.caption(f"Threshold used: {OPTIMAL_THRESHOLD}")

        # --- Debug info (عشان تتأكد إن الداتا صح) ---
        with st.expander("🔍 Show Debug Data (Data sent to model)"):
            st.write(final_df)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")