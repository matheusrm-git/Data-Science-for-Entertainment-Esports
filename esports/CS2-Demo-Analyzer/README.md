#  Demo Analyzer – Extracting insights from CS2 demos

## 📌 Overview
This project analyzes **Counter-Strike 2** professional match demos and extracts information based on metadata collected from .dem files. I've created a module called demo_analyzer.py to transform metadata into heatmaps, kill and death charts, and player statistics. Additionally, using the Streamlit library, it's possible to create interactive dashboards. Teams could use those to study and create team-oriented strategies. 

---

## 🎯 Objectives
- [x] Extract heatmaps from demo files.  
- [x] Extract player statistics from demo files.  
- [x] Create an interactive dashboard for insights.  

---

## 📂 Data
- **Source:** [HLTV demos](https://www.hltv.org/)  
- **Description:** Each demo can have thousands to millions of server ticks, including player, weapon, and match states.  

---

## ⚙️ Tech Stack
- **Language:** Python 3.11.9  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, demoparser2, IPython, Streamlit.  
- **Tools:** Jupyter Notebook, Streamlit.  

---

## 📊 Methodology
1. Metadata collection & preprocessing  
2. Exploratory Data Analysis (EDA)    
3. Visualization & insights  
4. Deployment (Dashboards)  

---

## 🚀 Results

Using the Demoparser2 library to transform .dem files into metadata dataframes, I was able to obtain a great amount of graphs and statistics, such as kills, deaths, assists, flashbang assists, and KDA. For instance, based on my functions, it's possible to analyze players individually, filtering by round, round time, and even afterplant situations.

For illustration purposes, I've built a Streamlit application to simulate an interactive dashboard and a Jupyter Notebook. Both of the demonstrations use a Blast Open London game of Furia against Legacy. Feel free to click on the links or stay with the screenshots.

- [Live Demo on Streamlit](https://cs2-demo-analyzer-dashboard.streamlit.app/)

![Streamlit App 1](https://github.com/matheusrm-git/Data-Science-for-Entertainment-Esports/blob/main/esports/CS2-Demo-Analyzer/demo_analyzer/assets/demonstration_prints/streamlit_app_demo_1.png)
![Streamlit App 2](https://github.com/matheusrm-git/Data-Science-for-Entertainment-Esports/blob/main/esports/CS2-Demo-Analyzer/demo_analyzer/assets/demonstration_prints/streamlit_app_demo_2.png)
![Streamlit App 3](https://github.com/matheusrm-git/Data-Science-for-Entertainment-Esports/blob/main/esports/CS2-Demo-Analyzer/demo_analyzer/assets/demonstration_prints/streamlit_app_demo_3.png)  


- [Jupyter Notebook](https://github.com/matheusrm-git/Data-Science-for-Entertainment-Esports/blob/main/esports/CS2-Demo-Analyzer/Analyzer%20Demonstration.ipynb)

![Jupyter 1](https://github.com/matheusrm-git/Data-Science-for-Entertainment-Esports/blob/main/esports/CS2-Demo-Analyzer/demo_analyzer/assets/demonstration_prints/jupyter_demo_1.png)
![Jupyter 2](https://github.com/matheusrm-git/Data-Science-for-Entertainment-Esports/blob/main/esports/CS2-Demo-Analyzer/demo_analyzer/assets/demonstration_prints/jupyter_demo_2.png)


---

## 📚 Learnings
- Improved my Python, Pandas, and Numpy writing a well-performing module that executes operations in Dataframes with millions of samples.
- Improved my Matplotlib and Seaborn skills by building heatmaps using PNG images.  
- Learned how to use the Streamlit library to create web applications.  

---

## 👤 Author
**Matheus Resende Miranda**  
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/matheus-resende-miranda/)    
- GitHub: [Your GitHub](https://github.com/matheusrm-git)  
