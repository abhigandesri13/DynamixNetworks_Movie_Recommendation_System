## 🎬 **Movie Recommendation System**  
### **Dynamix Networks – Machine Learning Internship (Task 2)**  

This project was developed as part of the **Machine Learning Internship at Dynamix Networks**.  
A complete **Hybrid Movie Recommendation System** is implemented using the **MovieLens Latest-Small dataset**.  

The system includes:  
- **Collaborative Filtering**  
- **Content-Based Filtering**  
- **Personalized User Recommendations**  
- **RMSE Evaluation**  
- **Local Streamlit User Interface for Demo**  


---

## 📁 **Dataset Information**  

**Dataset:** MovieLens Latest-Small  
**Users:** 610  
**Movies:** 9,742  
**Ratings:** 100,836  
**Source:** https://grouplens.org/datasets/movielens/latest/  

Files used:  
- `ratings.csv`  
- `movies.csv`  
- `tags.csv`  
- `links.csv`  

Folder structure:                                                                             
data/                                                                                           
ratings.csv                                                                 
movies.csv                                                                
tags.csv                                                                   
links.csv                                                                                  

---

## **📌 System Features**

### **1️⃣ Collaborative Filtering**
- Uses rating similarity between movies
- Cosine similarity applied on user–item matrix
- Returns similar movies for a selected title

### **2️⃣ Content-Based Filtering**
- Uses TF-IDF on genres + tags
- Cosine similarity on content vectors
- Returns movies with similar attributes

### **3️⃣ Personalized User Recommendations**
- Predictive score generated from historical user behavior
- Returns top-N recommendations per user

### **4️⃣ RMSE Evaluation**
- Baseline rating prediction model
- Measures prediction error

---

## **🧪 Notebook Execution (Model-Level)**

To run the model notebook:

1. Open `Movie_Recommendation.ipynb` in Google Colab or Jupyter
2. Execute cells step-by-step
3. Review:
   - EDA results
   - CF & Content outputs
   - Personalized recommendations
   - RMSE score

_No additional configuration needed._

---

## **🖥 Frontend Demo (Local Streamlit Execution)**

A **local web interface** is implemented for clean demo recording.

### Installation Requirements (Run Once)                                           
pip install streamlit                                                               
pip install scikit-learn                                                           
pip install pandas                                                                       
pip install numpy                                                                      

### Run the Application                                                      
In project folder:                                                                 
streamlit run app.py                                                                         

Browser will open automatically at:                                                                        
http://localhost:8501                                                                   

### Frontend Modes
- **Similar Movies (Collaborative)**
- **Similar Movies (Content-Based)**
- **Personalized User Recommendations**

### Demo Notes
- This project **uses local frontend only (no deployment)**.
- Video demonstration recorded directly from Streamlit UI.

---

## **📊 Example Demo Actions (For Recording)**

1. Select **Toy Story (1995)** → run Collaborative
2. Select **Jumanji (1995)** → run Content-Based
3. Select **User ID: 1** → show personalized recommendations

Tables update instantly showing recommended titles.

---

## **📁 Project Structure**

DynamixNetworks_Movie_Recommendation_System/                                                       
│                                                                                    
├── data/                                                                            
│ ├── ratings.csv                                                                        
│ ├── movies.csv                                                                         
│ ├── tags.csv                                                                          
│ └── links.csv                                                                         
│                                                                                                       
├── app.py                                                                      
├── Movie_Recommendation.ipynb                                                    
└── README.md                                                                                  

---


## 🔗 LinkedIn Project Announcement

I have officially shared this project and demonstration video on LinkedIn as part of my internship submission.

👉 LinkedIn Post:                               
                                                                                   

## **📌 Internship Deliverables Status**

| Task Requirement | Status |
|------------------|--------|
| Task 2 Completed | ✔ |
| Notebook Execution | ✔ |
| Frontend UI Added | ✔ |
| GitHub Repository | ✔ |
| README Updated Fully | ✔ |
| LinkedIn Post + Video | ✔ |
| Local Demo Recording | ✔ |

---

## **🙏 Acknowledgements**

Dataset by **MovieLens / GroupLens Research**  
Project completed under **Dynamix Networks Machine Learning Internship**

