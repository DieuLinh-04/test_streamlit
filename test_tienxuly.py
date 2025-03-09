
import pandas as pd
import pickle
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# Đọc dữ liệu từ file CSV
dataset = pd.read_csv('D:\AI\Code\itde-ai-is-54-a-2025\cityu10c_train_dataset.csv')

# Xử lý dữ liệu
print(dataset['EmploymentStatus'].unique())

dataset = dataset.dropna(axis=1, thresh=int(0.8 * len(dataset)))
dataset.fillna(dataset.select_dtypes(include=['number']).median(), inplace=True)
categorical_cols = dataset.select_dtypes(include=['object']).columns
for col in categorical_cols:
    dataset[col] = dataset[col].fillna('Unknown')

# Chọn các cột đặc trưng và nhãn
features = ['Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel', 'LoanAmount', 'LoanDuration']
target = ['LoanApproved']
X = dataset[features]
y = dataset[target]

# Định nghĩa các đặc trưng số và đặc trưng phân loại
categorical_features = ['EmploymentStatus', 'EducationLevel']
numerical_features = ['Age', 'AnnualIncome', 'CreditScore', 'LoanAmount', 'LoanDuration']

# Tạo bộ biến đổi cho dữ liệu số và dữ liệu phân loại
numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Kết hợp các bộ biến đổi
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Tạo pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Huấn luyện pipeline
pipeline.fit(X, y.values.ravel())

# Lưu pipeline vào file pickle
with open('decision_tree_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Pipeline trained and saved to decision_tree_pipeline.pkl")

# === Tạo ứng dụng Streamlit ===
st.title("Loan Approval Prediction App")

# Load pipeline từ file pickle
with open('decision_tree_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Nhận dữ liệu từ người dùng
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
education_level = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
loan_duration = st.number_input("Loan Duration (months)", min_value=1, value=12)

# Nút dự đoán
if st.button("Predict"):
    new_data = pd.DataFrame({
        'Age': [age],
        'AnnualIncome': [annual_income],
        'CreditScore': [credit_score],
        'EmploymentStatus': [employment_status],
        'EducationLevel': [education_level],
        'LoanAmount': [loan_amount],
        'LoanDuration': [loan_duration]
    })

    # Dự đoán
    prediction = loaded_pipeline.predict(new_data)
    
    # Hiển thị kết quả
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected.")

# Chạy ứng dụng Streamlit trên VS Code bằng lệnh: streamlit run filename.py
