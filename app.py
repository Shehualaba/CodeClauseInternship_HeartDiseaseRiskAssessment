# Import relevant libraries
import pandas as pd
import streamlit as st
import numpy as np
import pickle
import os
import joblib
import hashlib
from Username_store import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Disease Mortality Prediction </h1>
		<h5 style="color:white;text-align:center;">Hepatitis B </h5>
		</div>
		"""

# Avatar Image using a url
avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Healthy Diet</li>
		<li style="text-align:justify;color:black;padding:10px">Regular Exercise</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Weight Management</li>
		<li style="text-align:justify;color:black;padding:10px">Limit Alcohol</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Management</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Medication Adherence</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Definition</h3>
		<p>Hepatitis B is a viral infection that attacks the liver and can cause both acute and chronic disease.</p>
	</div>
	"""
	

def change_avatar(sex):
	if sex == "male":
		avatar_img = 'male_avatar.png'
	else:
		avatar_img = 'female_avatar.png'
	return avatar_img
# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False


def main():
    """Heart Disease Assessment"""
    

    menu= ["Home","Login","Sign up"]
    submenu = ["Information on Heart Disease", "Early Heart Disease Diagnosis"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.sidebar.markdown("**About Developer**")
        st.sidebar.image("WhatsApp Image 2024-05-01 at 15.47.41_f86b7e0a.jpg", width=200)
        st.sidebar.markdown("""
                                **Name**: Shehu Alaba Rasheed  
                                **Linkedin Username**: [Shehu Alaba](https://www.linkedin.com/in/shehu-alaba/)  
                                **Twitter Username**:[ShehuAlaba](https://twitter.com/ShehuAlaba)  
                                **Medium**:[Shehualaba ](https://medium.com/@shehualaba74)
        
        """)

        # CSS for styling
        st.markdown("""
            <style>
                /* Customize fonts and colors */
                body {
                    font-family: Arial, sans-serif;
                    color: #F8F8F2; /* Text color */
                    background-color: #212750; /* Background color */
                }
                .container {
                    max-width: 800px;
                    margin: auto;
                    padding: 20px;
                }
                .header {
                    background-color: #5681D0; /* Box background color */
                    color: #F8F8F2; /* Box text color */
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .section-box {
                    background-color: #1A1A3D; /* Box background color */
                    color: #F8F8F2; /* Box text color */
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                h1 {
                    font-size: 36px;
                    font-weight: bold;
                    margin: 0;
                }
            </style>
        """, unsafe_allow_html=True)

        # Header section
        st.markdown("""
            <div class="container">
                <div class="header">
                    <h1>Heart Disease Assessment</h1>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Introduction section
        st.markdown("""
            <div class="container">
                <h2>Welcome to Heart Disease Assessment</h2>
                <p>Heart disease is a leading cause of death globally, encompassing various heart and blood vessel conditions. The danger lies in potentially fatal consequences like heart attacks and strokes. Our web app assesses users' heart disease risk, considering factors like age, gender, cholesterol levels, and lifestyle habits. It empowers individuals to take proactive measures for heart health, aiming to reduce risk and improve well-being</p>
            </div>
        """, unsafe_allow_html=True)

        # About the Web App section
        st.markdown("""
            <div class="container section-box">
                <h3>About the Web App</h3>
                <p>The Heart Disease Assessment app uses machine learning algorithms to analyze user data and provide an assessment of their risk of heart disease.</p>
                <p>It takes into account various factors such as age, gender, cholesterol levels, blood pressure, and exercise habits to generate a risk score.</p>
                <p>Use this app as a tool for early detection and prevention of heart disease.</p>
                <h3>How to use app</h3>
                <ul>
                    <li>If you are new user, you have to sign up to be able to use App. First click the side arrow pointing to the right at the top left conner of the app.To sign up you need to enter a username and password, the password has to be confirmed</li>
                    <li>After signing up, you will need to log in with your password and username</li>
                    <li>After you successfully log in, you can interact with the app.</li>
                    <li>To get your Heart Disease Assessment, Click the "Choose The Action You Wish To Perform" dropdown and select Early Heart Disease Diagnosis</li>
                    <li>After you choose "Early Heart Disease Diagnosis", kindly enter your details</li>
                    <li>Click Submit details, then click predict</li>
                </ul>
                <h4>Note</h4>
                You must compeletely fill the form before you can predict, if you  click predict before completely filling the form you will get error
            </div>
        """, unsafe_allow_html=True)

        
        
        
        
        
        
        
      
    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password",type="password")
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pwd = generate_hashes(password)
            result = login_user(username,verify_hashes(password,hashed_pwd))
            if result:
                st.success("Welcome back {}".format(username))
                activity = st.selectbox("Choose The Action You Wish To Perform", submenu)
                if activity == "Information on Heart Disease":
                    # Define theme colors and font
                    primaryColor = "#5681D0"
                    backgroundColor = "#212750"
                    secondaryBackgroundColor = "#1A1A3D"
                    textColor = "#F8F8F2"
                    font = "sans-serif"
                    # Custom CSS for improved aesthetics
                    custom_css = f"""
                        <style>
                            /* Main background color */
                            .stApp {{
                                background-color: {backgroundColor};
                                color: {textColor};
                                font-family: {font};
                            }}

                            /* Section background color */
                            .css-1mpk29p {{
                                background-color: {secondaryBackgroundColor};
                                color: {textColor};
                                font-family: {font};
                                padding: 15px;
                                border-radius: 10px;
                                margin-bottom: 20px;
                            }}

                            /* Header color */
                            .css-1k6enqb {{
                                color: {primaryColor};
                                font-size: 28px;
                                font-weight: bold;
                                margin-bottom: 10px;
                            }}

                            /* Subheader color */
                            .css-1vrc75a {{
                                color: {textColor};
                                font-size: 20px;
                                font-weight: bold;
                                margin-bottom: 10px;
                            }}

                            /* Image border radius */
                            .stImage {{
                                border-radius: 10px;
                                margin-bottom: 15px;
                            }}

                            /* Horizontal rule color */
                            .horizontal-rule {{
                                border-top: 2px solid {primaryColor};
                                margin: 30px 0;
                            }}
                        </style>
                    """
                    # Header
                    st.title("Understanding Heart Disease: Causes, Symptoms, and Prevention")
                    st.markdown(custom_css, unsafe_allow_html=True)
                    # Introduction
                    st.header("Introduction:")
                    st.write(
                    "Heart disease remains one of the leading causes of death worldwide, affecting millions of individuals annually. "
                    "It encompasses a range of conditions that affect the heart's structure and function, impairing its ability to pump blood effectively. "
                    "Understanding the causes, symptoms, and prevention strategies is crucial in combating this pervasive health issue."
                    )
                    st.markdown("---")

                    # Causes
                    st.header("Causes:")
                    st.write(
                        "Heart disease can develop due to various factors, including:\n"
                        "1. **Lifestyle Choices**\n"
                        "2. **Medical Conditions**\n"
                        "3. **Genetics**\n"
                        "4. **Age and Gender**\n"
                        "5. **Other Factors**\n"
                    )
                    st.markdown("---")
                    # Causes
                    st.header("Causes:")
                    st.markdown(
                            "Heart disease can develop due to various factors, including:\n"
                            "1. **Lifestyle Choices**\n"
                            "2. **Medical Conditions**\n"
                            "3. **Genetics**\n"
                            "4. **Age and Gender**\n"
                            "5. **Other Factors**\n"
                            )
                    st.markdown("---")

                    # Symptoms
                    st.header("Symptoms:")
                    
                    st.write(
                        "Symptoms of heart disease can vary depending on the specific condition but may include:\n"
                        "1. **Chest Pain or Discomfort**\n"
                        "2. **Shortness of Breath**\n"
                        "3. **Fatigue**\n"
                        "4. **Swelling**\n"
                        "5. **Irregular Heartbeat**\n"
                    )
                    st.markdown("---")

                    # Prevention
                    st.header("Prevention:")
                    st.write(
                        "Preventing heart disease involves adopting a heart-healthy lifestyle and managing risk factors effectively. "
                        "Here are some prevention strategies:\n"
                        "1. **Healthy Diet**\n"
                        "2. **Regular Exercise**\n"
                        "3. **Maintain a Healthy Weight**\n"
                        "4. **Manage Stress**\n"
                        "5. **Quit Smoking**\n"
                        "6. **Limit Alcohol**\n"
                        "7. **Regular Health Check-ups**\n"
                    )
                    st.markdown("---")

                    # Conclusion
                    st.header("Conclusion:")
                    st.write(
                        "Heart disease is a complex condition that requires a comprehensive approach to prevention and management. "
                        "By adopting a heart-healthy lifestyle and effectively managing risk factors, individuals can reduce their risk of heart disease and improve their overall heart health."
                    )

                    # Footer
                    st.markdown("---")
                    st.write("This information page is provided for educational purposes only. Please consult a healthcare professional for personalized medical advice.")

                    
                # Early Heart Disease Diagnosis
                elif activity=="Early Heart Disease Diagnosis":
                    st.markdown("##### Please fill out your information to Know if you are at risk of an Heart Disease")
                    # User Input Form
                    form = st.form('data_form')
                    age = form.number_input("Enter Your Age", min_value=18, max_value=77)
                    sex = form.radio("Choose your Gender",["Male","Female"], index=None)
                    chtyp = ["Typical Angina"," Atypical Angina","Non-Anginal Pain"," Asymptomatic"]
                    ChestPainType = form.selectbox("Choose the Type of chest pain you experience", chtyp, index=None)
                    RestingBP = form.number_input("Enter Your Resting Blood Pressure (mmHg)", min_value=92, max_value=200)
                    Cholesterol = form.number_input("Enter Your Serum Cholesterol level (mm/dl)", min_value=85, max_value=603)
                    FastingBS = form.radio("Is your Fasting Blood sugar level > 120mg/dl",["Yes","No"], index=None)
                    RestingECG = form.selectbox("Select Resting Electrocardiogram Results", ["Normal","ST-T wave abnormality", "Left ventricular hypertrophy"], index=None)
                    MaxHR = form.number_input("Enter Your Maximum heart rate achieved", min_value=60, max_value=202)
                    ExerciseAngina = form.radio("Do you experience heart discomfort after an exercise", ["Yes","No"], index=None)
                    Oldpeak = form.number_input("Select ST Depression", min_value=0.0, max_value=6.0, step=0.1)
                    ST_Slope = form.selectbox("What is slope of the peak exercise ST segment",['Up sloping',"Flat","Down Slope"], index=None)                 
                    form.form_submit_button('Submit details')

                    # Use the data to create a dataframe
                    user_data = pd.DataFrame({
                        
                            "Age" : [age],
                            "Sex" :	[sex],
                            "ChestPainType"	: [ChestPainType],
                            "RestingBP"	: [RestingBP],
                            "Cholesterol" : [Cholesterol],
                            "FastingBS"	: [FastingBS],
                            "RestingECG" : [RestingECG],	
                            "MaxHR"	: [MaxHR],
                            "ExerciseAngina" : [ExerciseAngina],	
                            "Oldpeak"	: [Oldpeak],
                            "ST_Slope"	: [ST_Slope]
                        }
                    )
                    st.markdown("### Confirm Your Details")
                    st.dataframe(user_data)
                    user_data["Sex"] = user_data["Sex"].map({"Male":"M", "Female":"F"})
                    user_data["ChestPainType"] = user_data["ChestPainType"].map({"Typical Angina":"TA"," Atypical Angina":"ATA","Non-Anginal Pain":"NAP"," Asymptomatic":"ASY"})
                    user_data["RestingECG"] = user_data["RestingECG"].map({"Normal":"Normal","ST-T wave abnormality":"ST", "Left ventricular hypertrophy":"LVH"})
                    user_data["ExerciseAngina"] = user_data["ExerciseAngina"].map({"No":"N","Yes":"Y"})
                    user_data["ST_Slope"] = user_data["ST_Slope"].map({'Up sloping':'Up',"Flat":'Flat',"Down Slope":"Down"})
                    
                    # Import Model
                    model = joblib.load("model.pkl")
                    # Import Preprocessor
                    preprocessor = joblib.load("preprocessor.pkl")
                    if st.button("Get Prediction"):
                         # Preprocess user data
                        userdata_scaled = preprocessor.transform(user_data)
                        # Get Prediction
                        prediction = model.predict(userdata_scaled)
                        predict_prob = model.predict_proba(userdata_scaled)[0][1] * 100
                        if prediction == 1:
                            st.warning(f"You have a {predict_prob:.2f}% chance of having heart disease.")
                            st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
                                
                
                        else:
                            st.success("You are not at Risk of a Heart Disease")
                            




            else:
                st.warning("Incorrect Username/Password")


    elif choice=="Sign up":
        new_user = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password", type="password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password")
        
        if new_password == confirm_password:
            st.sidebar.success("Password Confirm")
        else:
            st.sidebar.warning("Password is not the same")
        if st.sidebar.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_user, hashed_new_password)
            st.success(f"Congratulations! {new_user}, You have successfully created a new account")
            st.info("Log In to Get Started")







    
if __name__ == '__main__':
	main()
