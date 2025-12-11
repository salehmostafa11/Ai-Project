import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# --- إعدادات أساسية ---
MODEL_PATH = 'ensemble_attrition_model.pkl'
OPTIMAL_THRESHOLD = 0.43 

# 🛑 القائمة النهائية والوحيدة الصحيحة للأعمدة الـ 43 بالترتيب الدقيق المطلوب
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
# --- تهيئة تطبيق Flask وتحميل الموديل ---
app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model {MODEL_PATH} loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 

# --- مسار معالجة البيانات ---

# def preprocess_input(data_json):
#     """
#     يضمن هذا المنطق أن يكون DataFrame النهائي مطابقاً لـ FEATURE_COLS بالضبط.
#     """
#     # 1. إنشاء DataFrame 
#     data_df = pd.DataFrame([data_json])
    
#     # 2. إسقاط الأعمدة (Monthly Income)
#     data_df = data_df.drop('Monthly Income', axis=1, errors='ignore')
    
#     # 3. الترميز الثنائي (Gender, Over Time)
#     binary_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
#     # يجب أن تتطابق أسماء الأعمدة هنا مع الـ keys في JSON (من الـ form)
#     data_df['Gender'] = data_df['Gender'].map(lambda x: binary_map.get(x, 0))
#     data_df['Over Time'] = data_df['Over Time'].map(lambda x: binary_map.get(x, 0))

#     # 4. الترميز الأحادي الساخن (OHE) - بدون إسقاط Drop First
#     OHE_COLS_WITH_SPACES = ['Business Travel', 'Department', 'Education Field', 'Job Role', 'Marital Status']
#     data_df = pd.get_dummies(data_df, columns=OHE_COLS_WITH_SPACES, drop_first=False)
    
#     # 5. تنظيف أسماء الأعمدة بعد OHE لمطابقة أسماء الموديل
#     data_df.columns = data_df.columns.str.replace(' ', '')
#     data_df.columns = data_df.columns.str.replace('-', '_')
    
#     # 6. 🛑 النقطة الحاسمة: إعادة الفهرسة لضمان الترتيب الصحيح
#     final_df = data_df.reindex(columns=FEATURE_COLS, fill_value=0)
    
#     return final_df
# --- مسار معالجة البيانات ---
def preprocess_input(data_json):
    # 1. إنشاء DataFrame من البيانات القادمة من الـ Form
    data_df = pd.DataFrame([data_json])
    
    # 2. قاموس لتوحيد الأسماء: (الاسم في HTML) -> (الاسم في الموديل)
    # هذا يحل مشكلة "Number of Companies Worked in" ومشكلة المسافات
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
    
    # تطبيق تغيير الأسماء
    data_df = data_df.rename(columns=rename_map)

    # 3. إسقاط الأعمدة التي لا يحتاجها الموديل (Monthly Income غير موجود في قائمة الموديل)
    #
    data_df = data_df.drop('Monthly Income', axis=1, errors='ignore')

    # 4. معالجة البيانات النصية (Encoding)
    
    # Binary Encoding
    binary_map = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    if 'Gender' in data_df.columns:
        data_df['Gender'] = data_df['Gender'].map(binary_map)
    if 'OverTime' in data_df.columns:
        data_df['OverTime'] = data_df['OverTime'].map(binary_map)

    # One-Hot Encoding
    # نحدد الأعمدة التي تحتاج تحويل (بأسمائها الجديدة بعد التعديل)
    ohe_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    
    # نتأكد أن الأعمدة موجودة قبل عمل get_dummies
    cols_to_encode = [c for c in ohe_cols if c in data_df.columns]
    data_df = pd.get_dummies(data_df, columns=cols_to_encode, prefix=cols_to_encode, prefix_sep='_', drop_first=False)

    # 5. 🛑 الخطوة الأهم: إعادة ترتيب الأعمدة (Reindexing)
    # هذه الخطوة تضمن أن الجدول النهائي يحتوي على الـ 43 عمود بالضبط بنفس ترتيب الموديل
    # وأي عمود ناقص (بسبب الـ One-Hot Encoding) سيتم إنشاؤه وتعبئته بـ 0
    final_df = data_df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return final_df
# --- home page end point (GET) ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- (API Endpoint - POST) ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
    try:
        data = request.get_json(force=True) 
        processed_data = preprocess_input(data)
        
        if processed_data.shape[1] != 43:
             #only appears if columns are not valid
             return jsonify({"error": f"Feature count mismatch after processing. Expected 43, got {processed_data.shape[1]}. Please ensure all 28 fields are submitted."}), 400

        # التنبؤ
        probability = model.predict_proba(processed_data)[0][1]
        prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
        result_label = "Likely to leave (Yes) 😟" if prediction == 1 else "Likely to stay (No) 😊"
        
        return jsonify({
            'attrition_prediction': result_label,
            'probability_of_attrition': f"{probability:.4f}",
            'threshold_used': OPTIMAL_THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}. Check that you sent all 28 fields in the correct format (JSON)."}), 400

#---------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)