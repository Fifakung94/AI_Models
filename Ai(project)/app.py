import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. ตั้งค่า Config และหัวข้อเว็บ ---
st.set_page_config(page_title="Project IS 2568 - AI Model", layout="wide")

# --- 2. ฟังก์ชันโหลดโมเดล (ดึงจากโฟลเดอร์ models) ---
@st.cache_resource
def load_all_models():
    # โหลดโมเดล Diabetes
    with open('Ai(project)/models/diabetes_model.pkl', 'rb') as f:
        d_model = pickle.load(f)
    with open('Ai(project)/models/scaler_diabetes.pkl', 'rb') as f:
        d_scaler = pickle.load(f)

    # โหลดโมเดล Telco Churn
    with open('Ai(project)/models/churn_model.pkl', 'rb') as f:
        c_model = pickle.load(f)
    with open('Ai(project)/models/scaler_churn.pkl', 'rb') as f:
        c_scaler = pickle.load(f)

    return d_model, d_scaler, c_model, c_scaler

# โหลดโมเดลมาเก็บไว้ในตัวแปร
try:
    d_model, d_scaler, c_model, c_scaler = load_all_models()
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.info("ตรวจสอบว่าไฟล์ในโฟลเดอร์ models มีครบ (diabetes_model.pkl, scaler_diabetes.pkl, churn_model.pkl, scaler_churn.pkl)")

# --- 3. Sidebar Menu (แบ่ง 4 หน้าตาม PDF) ---
st.sidebar.title("📑 เมนูหลัก")

st.sidebar.divider()
st.sidebar.markdown("### 👤 ผู้จัดทำ")
st.sidebar.write("ชื่อ: ภูมิพัฒน์ มากถาวร")
st.sidebar.write("รหัสนักศึกษา: 6704062617181")
st.sidebar.divider()

page = st.sidebar.radio("เลือกหน้า:", [
    "1. ทฤษฎี: Diabetes (Ensemble)", 
    "2. ทดสอบ: Diabetes Prediction", 
    "3. ทฤษฎี: Telco (Neural Network)", 
    "4. ทดสอบ: Churn Prediction"
])

# ==========================================
# หน้าที่ 1: อธิบายโมเดล Diabetes (Ensemble)
# ==========================================
if page == "1. ทฤษฎี: Diabetes (Ensemble)":
    st.title("🩺 อธิบายโมเดลทำนายโรคเบาหวาน (Ensemble Learning)")
    st.divider()
    
    st.subheader("🧹 การเตรียมข้อมูล (Preprocessing)")
    st.write("ข้อมูลชุดนี้มีความไม่สมบูรณ์ (Incompleteness) โดยพบว่าค่า Glucose, BMI และ Blood Pressure มีค่าเป็น '0' ในหลายแถว")
    st.info("แนวทางแก้ไข: ทำการเปลี่ยนค่า 0 เป็นค่าเฉลี่ย (Mean) ของแต่ละคอลัมน์เพื่อให้ข้อมูลสมบูรณ์")
    
    st.subheader("🤖 ทฤษฎีอัลกอริทึม")
    st.write("ใช้เทคนิค **Voting Classifier** โดยรวมเอา 3 อัลกอริทึมมาตัดสินใจร่วมกัน:")
    st.markdown("- **Random Forest:** ใช้ Decision Tree หลายต้นโหวตกัน\n- **AdaBoost:** เน้นการเรียนรู้จากจุดที่ผิดพลาด\n- **Logistic Regression:** ใช้หาความน่าจะเป็นเชิงเส้น")
    
    st.subheader("⚙️ ขั้นตอนการพัฒนาโมเดล")
    st.markdown("""
    1. นำเข้าข้อมูล : โหลด Dataset และตรวจสอบหาค่าที่ผิดปกติหรือขาดหายไป
    2. เตรียมข้อมูล (Preprocessing) : แทนที่ค่า 0 ในคอลัมน์ที่ผิดปกติด้วยค่าเฉลี่ย และทำ Standard Scaling
    3. สร้างและประกอบโมเดล (Ensemble) : สร้าง Random Forest, AdaBoost, Logistic Regression และนำมารวมร่างกันด้วย Voting Classifier
    4. ฝึกสอนและบันทึก (Train & Save) : นำข้อมูลเข้าฝึกสอน และบันทึกโมเดลเป็นไฟล์ `.pkl` เพื่อนำมาใช้บนเว็บ
    """)
    st.markdown("🔗 **แหล่งข้อมูล Dataset :** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)")

# ==========================================
# หน้าที่ 2: ทดสอบโมเดล Diabetes
# ==========================================
elif page == "2. ทดสอบ: Diabetes Prediction":
    st.title("🧪 ทดสอบการทำนายโรคเบาหวาน")
    st.write("กรุณากรอกข้อมูลเพื่อวิเคราะห์ความเสี่ยง")
    
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            preg = st.number_input("จำนวนครั้งที่ตั้งครรภ์", 0, 20, 0)
            glucose = st.number_input("ระดับน้ำตาล (Glucose)", 0, 200, 100)
            bp = st.number_input("ความดันโลหิต (Blood Pressure)", 0, 150, 80)
            stk = st.number_input("ความหนาผิวหนัง (Skin Thickness)", 0, 100, 20)
        with col2:
            ins = st.number_input("ระดับอินซูลิน (Insulin)", 0, 900, 80)
            bmi = st.number_input("ดัชนีมวลกาย (BMI)", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
            age = st.number_input("อายุ", 1, 100, 30)
            
        if st.form_submit_button("วิเคราะห์ผล"):
            input_data = np.array([[preg, glucose, bp, stk, ins, bmi, dpf, age]])
            scaled_data = d_scaler.transform(input_data)
            prediction = d_model.predict(scaled_data)
            
            if prediction[0] == 1:
                st.error("🚨 ผลการทำนาย: มีความเสี่ยงเป็นโรคเบาหวาน")
            else:
                st.success("✅ ผลการทำนาย: ไม่มีความเสี่ยงเป็นโรคเบาหวาน")

# ==========================================
# หน้าที่ 3: อธิบายโมเดล Telco (Neural Network)
# ==========================================
elif page == "3. ทฤษฎี: Telco (Neural Network)":
    st.title("🧠 อธิบายโมเดลทำนายการย้ายค่าย (Neural Network)")
    st.divider()
    
    st.subheader("🧹 การเตรียมข้อมูล (Preprocessing)")
    st.warning("ตรวจพบความไม่สมบูรณ์: คอลัมน์ TotalCharges มีค่าว่างเป็น String (' ')")
    st.write("แนวทางแก้ไข: ทำการแปลงเป็นตัวเลข (Numeric) และเติมค่า 0 ในตำแหน่งที่ว่าง พร้อมทั้งทำ Feature Scaling")
    
    st.subheader("🕸️ โครงสร้างโมเดล (MLPClassifier)")
    st.write("ใช้ Multi-Layer Perceptron (MLP) จาก Scikit-learn เพื่อความรวดเร็ว")
    st.code("""
    - Input Layer: รับ 3 Features หลัก (Tenure, MonthlyCharges, TotalCharges)
    - Hidden Layer 1: 16 Nodes (ReLU)
    - Hidden Layer 2: 8 Nodes (ReLU)
    - Output Layer: 1 Node (Log-Loss / Probability)
    """)
    
    st.subheader("⚙️ ขั้นตอนการพัฒนาโมเดล")
    st.markdown("""
    1. นำเข้าข้อมูล : โหลด Dataset การใช้งานของลูกค้า (Telco)
    2. เตรียมข้อมูล (Preprocessing) : จัดการค่าว่าง (String ว่าง) ใน TotalCharges โดยแปลงเป็นตัวเลขและเติม 0 จากนั้นทำ Standard Scaling
    3. สร้างโมเดล (Model Building) : กำหนดโครงสร้าง Neural Network แบบ 2 Hidden Layers ตามที่ออกแบบไว้
    4. ฝึกสอนและบันทึก (Train & Save) : นำข้อมูลเข้าฝึกสอนโมเดล และบันทึกเป็นไฟล์ `.pkl` เพื่อนำมาใช้บนเว็บ
    """)
    
    st.markdown("🔗 **แหล่งข้อมูล Dataset :** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")
    

# ==========================================
# หน้าที่ 4: ทดสอบโมเดล Telco Churn
# ==========================================
elif page == "4. ทดสอบ: Churn Prediction":
    st.title("🧪 ทดสอบการทำนายการเลิกใช้งาน (Churn)")
    
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("ระยะเวลาที่เป็นลูกค้า (เดือน)", 0, 72, 12)
    with col2:
        monthly = st.number_input("ค่าบริการรายเดือน ($)", 0.0, 200.0, 50.0)
        total = st.number_input("ค่าบริการรวมทั้งหมด ($)", 0.0, 10000.0, 500.0)
    
    if st.button("วิเคราะห์โอกาสย้ายค่าย", type="primary"):
        input_data = np.array([[tenure, monthly, total]])
        scaled_data = c_scaler.transform(input_data)
        
        # สำหรับ MLPClassifier ใช้ predict_proba เพื่อหาความน่าจะเป็น
        prob = c_model.predict_proba(scaled_data)[0][1]
        
        st.metric("โอกาสที่ลูกค้าจะย้ายค่าย", f"{prob*100:.2f} %")
        if prob > 0.5:
            st.warning("⚠️ ลูกค้ามีแนวโน้มจะเลิกใช้บริการสูง")
        else:
            st.success("💖 ลูกค้ามีแนวโน้มจะใช้บริการต่อ")