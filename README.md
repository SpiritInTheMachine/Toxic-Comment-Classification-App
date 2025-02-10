# **Toxic Comment Classification System**

## **Overview**
The **Toxic Comment Classification System** is a **machine learning-based application** designed to **automate content moderation** by detecting and categorizing toxic comments in real-time. Using **Natural Language Processing (NLP)** and **multi-label classification**, this system helps **reduce human moderation efforts**, improve **decision-making**, and enhance **user safety** on online platforms.

This project was built as part of a **capstone initiative**, demonstrating expertise in **data science, software development, and machine learning deployment**.

---

## **Features**
✅ **Real-time toxic comment classification** (Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate)  
✅ **Interactive GUI** built with Tkinter/PyQt5  
✅ **Data visualizations** (Bar Chart, Pie Chart, Heatmap) for insights  
✅ **Standalone executable (.exe)** for easy deployment (no dependencies required)   
✅ **Pre-trained model with real-time prediction capability**  
✅ **Dynamic model training** with new datasets (optional)  

---

## **Technologies Used**
- **Programming Language:** Python  
- **Machine Learning & NLP:** Scikit-learn, NLTK  
- **Data Processing & Visualization:** Pandas, Matplotlib, Seaborn  
- **GUI Development:** Tkinter / PyQt5  
- **Deployment & Packaging:** PyInstaller (Standalone `.exe`)  

---

## **Installation & Usage**
### **1. Running the Executable (.exe) File**
- Download and unzip the packaged folder.
- Double-click `Toxic_Comment_Classifier.exe` to launch the GUI.
- Enter a comment in the text box and click **"Classify"**.
- View **real-time classification results**.
- Click visualization buttons to analyze data insights.

### **2. Running the Python Script (For Developers)**
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- Run the script:
  ```sh
  python TASK_2_toxic_comment_classification_app.py
  ```
- Use the GUI for classification and visualizations.

### **3. Training a New Model (Optional)**
- Modify `train.csv` to include new comments.
- Run `TASK_2_toxic_comment_classification_app.py` to retrain the model dynamically.

---

## **Project Structure**
```
📂 Toxic-Comment-Classification
│── 📂 data/                # Dataset files (train.csv, test.csv)
│── 📂 src/                 # Source code directory
│── 📂 dist/                # Packaged executable output
│── TASK_2_toxic_comment_classification_app.py   # Main application script
│── README.md               # Project documentation
│── requirements.txt        # Dependencies list
│── Toxic_Comment_Classifier.exe   # Standalone executable
```

---

## **Future Improvements**
**Expand dataset** with real-world comments for improved accuracy  
**Overhaul GUI** for better visualization  
**Support multi-language toxicity detection**  
**Deploy as a web-based tool** for platform integration  

---

## **Acknowledgments**
Thanks to **Kaggle** for providing the **Jigsaw Toxic Comment Classification Dataset** for supporting this capstone project.

---
