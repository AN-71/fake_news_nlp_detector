# Fake News NLP Detector 📰

A simple machine learning classifier built with **scikit-learn** and **NLTK** to detect fake news based on title and text content using **TF-IDF** and **Logistic Regression**.

---

## 📊 Experiments & Insights

On a standard test set of 1,200 samples, the model achieves **97.25% accuracy**. However, a series of provocative experiments revealed interesting limitations:

1. **Absurd News (83.3% — 5 out of 6):** The model successfully catches obvious fiction (e.g., *"Ancient scientists built interstellar spaceships"* → `Fake`), though it can occasionally slip on vivid imagery.

2. **Political Fakes (14.3% — 1 out of 7):** When fake stories are wrapped in familiar political terminology (*"mobilization," "ruling party," "conflict"* — including a tale about *drunk neighbour from the second floor becoming the hero of the Eurasian crisis*), the model often classifies them as `Real`. Stylistic patterns proved more influential than factual plausibility.

3. **Sensitivity to Phrasing:** The same fabricated news story about banning smartphones in Estonian schools received different labels (`Real` or `Fake`) depending solely on how the sentence was structured.

---

## 🛠 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/AN-71/fake_news_nlp_detector.git
cd fake_news_nlp_detector
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn nltk
```

### 3. Download the dataset

1. Download the dataset from Kaggle: [Fake News Detection Dataset (6000+ News Articles)](https://www.kaggle.com/datasets/mobeenfatimah/fake-news-detection-dataset-6000-news-articles).
2. Place the `news_dataset.csv` file in the project's root directory.

### 4. Run the script

```bash
python fake_news_nlp_detector.py
```

The script will train the model, output evaluation metrics (accuracy, classification report, and confusion matrix), and launch an interactive console prompt for testing custom news articles.

---

## 📂 Project Structure

- `fake_news_nlp_detector.py` — Main Python script for training, evaluation, and interactive prediction.
- `news_dataset.csv` — Source news dataset (downloaded separately from Kaggle).
- `README.md` — Project documentation.

---

## 📜 Dataset Attribution & License

### Original Dataset Author

**Mobeen Fatima**

*Computer Science (Specialized AI) Student | Machine Learning & AI Enthusiast*

- **GitHub:** [MobeenFatimaa](https://github.com/MobeenFatimaa)
- **LinkedIn:** [Mobeen Fatima](https://www.linkedin.com/in/mobeen-fatima-599a35347/)
- **Original dataset:** [Fake News Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mobeenfatimah/fake-news-detection-dataset-6000-news-articles)

### License

This project utilizes material licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**. You are free to share and adapt the material, provided that you give appropriate credit to the original author and distribute your contributions under the same license.
