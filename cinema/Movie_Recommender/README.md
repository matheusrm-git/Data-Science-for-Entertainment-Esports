# 🎬 Movie Night ! - Movie Recommendation System with Neural Networks

This project implements a **movie recommendation system** powered by a **neural network built with TensorFlow**, trained on millions of user ratings from the **MovieLens dataset**.  
The system predicts personalized movie preferences and serves recommendations through an interactive **Streamlit web app**.

---

## 🚀 Features

- Deep learning–based recommendation model using TensorFlow/Keras  
- Trained on millions of ratings from the [MovieLens](https://grouplens.org/datasets/movielens/) dataset  
- Web application built with Streamlit for interactive user experience  
- User-based and item-based embeddings for similarity search  
- Real-time recommendation display with dynamic UI  

---

## 🧠 Model Overview

The model uses **embedding layers** to represent users and movies in a shared latent space.  
These embeddings are learned via supervised training on user–movie rating pairs, optimizing the **mean squared error** between predicted and actual ratings.

**Architecture summary:**
- Two input layers (user, movie)  
- Embedding layers for each entity  
- Dot product of embedding vectors  
- Output layer predicting normalized rating (0–5)  

---

## 🛠️ Tech Stack

- **Python** 3.13.9  
- **TensorFlow / Keras** — model training and evaluation  
- **Pandas / NumPy** — data preprocessing  
- **Streamlit** — web application  
- **Scikit-learn** — metrics and data splitting  
- **Matplotlib / Seaborn** — exploratory data analysis  

---

## 💾 Dataset

- **Source:** [MovieLens 33M Dataset](https://grouplens.org/datasets/movielens/latest/)  
- Contains over **33 million ratings** across **86,000 movies** by **330,975 users**  
- Cleaned, encoded, and normalized for training and validation  

---

## 📊 Results

- Model achieved **RMSE = 0.145** on validation set  
- Learned meaningful embeddings — similar movies cluster together in embedding space  
- Deployed via Streamlit for quick user testing and demo  

---

## 🌐 Demo

You can try the live demo here: [🔗 Streamlit App Link](https://movienightapp.streamlit.app/)

![Movie Night !](https://github.com/matheusrm-git/Data_Science_for_Entertainment_Esports/blob/main/cinema/Movie_Recommender/img/app_screanshot.png)

---

## 📂 Repository Structure

```python

📦 Movie_Recommender/
├── csv_files/ # Dataset files (not included in repo)
├── notebooks/ # Exploratory notebooks
├── src/ # Model training and Streamlit app scripts
├── img/ # Images used in the project
├── requirements.txt # Dependencies
└── README.md # Project documentation

```

## 🧩 Future Work

- Add content-based hybrid features (genres, tags, etc.)  
- Implement model monitoring and retraining pipeline  

## 👤 Author
**Matheus Resende Miranda**  
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/matheus-resende-miranda/)    
- GitHub: [Your GitHub](https://github.com/matheusrm-git) 
