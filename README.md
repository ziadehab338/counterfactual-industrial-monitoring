# 🔧 Intelligent Predictive Maintenance System with Explainable AI

## 📌 Overview
This project presents a complete intelligent predictive maintenance system that goes beyond traditional machine learning models.

Instead of only predicting machine failure, the system provides:
- Failure prediction
- Failure type classification
- Risk assessment (even before failure happens)
- Actionable insights to prevent failure

The goal is to transform raw sensor data into **clear decisions and preventive actions**.

---

## 🌐 Interactive Web Application (Streamlit)

An interactive web application was built using **Streamlit** to make the system accessible and easy to use.

Through the interface, users can:
- Input machine sensor values
- Get instant failure predictions
- View failure type (if applicable)
- Understand risk levels
- Explore counterfactual explanations visually

> This turns the project from a machine learning model into a real-world decision-support system.

---

## ⚙️ System Capabilities

### 🟢 1. Failure Prediction (Binary Classification)
A Logistic Regression model predicts whether a machine is likely to fail or not based on sensor readings.

---

### 🔴 2. Failure Type Classification (Multi-class)
If a failure is predicted, an XGBoost model identifies the exact failure type:

- Tool Wear Failure (TWF)
- Heat Dissipation Failure (HDF)
- Power Failure (PWF)
- Overstrain Failure (OSF)
- Random Failure (RNF)

---

### ⚠️ 3. Risk Level Assessment
Even if the machine is not currently failing, the system detects **risk levels** by identifying patterns similar to previous failure conditions.

> This enables early warning before actual failure occurs.

---

### 🔄 4. Counterfactual Explanations (What-if Analysis)
Using DiCE, the system generates alternative scenarios showing:

> "What changes are needed to prevent failure?"

This allows engineers to take **preventive actions** instead of reacting after breakdown.

---

### 📊 5. Counterfactual Visualization
The system visualizes:
- Original machine state
- Modified (safe) state

Helping users clearly understand what changed and why.

---

### 📈 6. Sensor Importance Analysis
The system identifies the most influential sensors affecting predictions using model feature importance.

This answers critical questions like:

> Which sensors are driving the failure decision?

Examples:
- Is temperature the main issue?
- Is torque causing overload?
- Is tool wear the key factor?

---

## 🧠 How the System Works

1. Input machine sensor data  
2. Predict failure (Yes / No)  
3. If failure is predicted:
   - Identify failure type using XGBoost  
4. If no failure:
   - Evaluate risk level (early warning)  
5. Generate counterfactual scenarios  
6. Analyze feature importance  
7. Display results via Streamlit interface  

---

## 🛠️ Technologies Used

- Python 🐍
- Scikit-learn
- XGBoost
- DiCE (Explainable AI)
- Streamlit 🌐
- Pandas & NumPy
- Matplotlib

---

## 📊 Dataset

AI4I 2020 Predictive Maintenance Dataset

Includes industrial sensor data:
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

---

## 🎯 Key Highlights

- 🔥 Combines **Prediction + Explanation + Action**
- 🧠 Uses a **two-stage modeling architecture**
- ⚠️ Detects **risk before failure occurs**
- 🌐 Interactive web application (Streamlit)
- 📊 Provides **sensor-level insights**
- 🔄 Generates **real-world preventive scenarios**

---

## 💡 Example Output

Failure: Yes  
Type: Heat Dissipation Failure  

Suggested Fix:
- Reduce temperature
- Adjust torque

OR

No Failure  
Risk Level: Medium  

Key Risk Factors:
- Increasing temperature
- High torque  

---

## 📌 Why This Project?

Traditional systems answer:
> "Will the machine fail?"

This system answers:
> "Why will it fail, how risky is it, and how can we prevent it?"

---
