# 1 Introduction

### Motivation:  
In today’s digital world, political communication increasingly takes place on social media platforms such as
Twitter. Politicians, parties, journalists, and citizens all participate in debates using short, emotionally
charged text formats. These messages rarely include clear party identifiers, making it difficult to assign
statements to political affiliations.
Our project explores whether it is possible to predict the party affiliation associated with a tweet using
only the textual content. The goal is to build a machine learning model that classifies tweets by party
labels, uncovering patterns in political language and enabling automated analysis of political discourse

### Research question:  
Can political orientation be detected from tweets using only their text, without additional metadata?

### Goal
- Develop a machine learning model to classify tweets by political party
- Create a web app (Streamlit) for interactive prediction
- Support discourse analysis and digital media literacy with transparent tools

---

# 2 Related Work
Previous research has examined political affiliation prediction using speeches, manifestos, and social network information. In particular, the work by Beese et al. (2022) presented at Konvens explored predicting party affiliation in German tweets using a combination of transformer models and linguistic features. Their results showed the potential and limitations of text-only classification in a political context.

> Beese, L., Rehm, G., & Eckart, T. (2022). Towards the Detection of Political Affiliation in German Tweets. In *Proceedings of the Conference on Natural Language Processing (KONVENS)*. [PDF](https://aclanthology.org/2022.konvens-1.9.pdf)

Inspired by this and similar research, our project focuses on a minimalist approach — using only raw tweet text and classical ML methods for educational and exploratory purposes.

---

# 3 Methodology

## 3.1 General Methodology
We began by loading a dataset of tweets labeled with political party affiliations. We designed an
experimental pipeline that:
- Cleans and processes tweet text
- Extracts relevant linguistic features
- Trains and evaluates multiple classifiers
- Deploys a web-based application 

## 3.2 Data Understanding and Preparation
The dataset consists of German-language tweets labeled with the political party of the author. It was sourced from:

> https://faubox.rrze.uni-erlangen.de/getlink/fi8W52QUEdtmm7LEGLiDBD/twitter-bundestag-2022.tar.gz

Each party's dataset was capped at 1,000 tweets to ensure class balance.

Final preprocessing steps applied:
- Lowercasing
- Removal of URLs, mentions
- Demojize
- Punctuation and stopword removal
- Tokenization
- Vectorization using TF-IDF
- Semantic embeddings using a pretrained German BERT model
- Basic style features (tweet length, emoji usage, hashtag presence)

## 3.3 Modeling and Evaluation
We evaluated multiple models, selecting **XGBoost** for its performance and compatibility with explainability tools. The features were concatenated into sparse and dense representations for optimal results.

To explain model predictions, we integrated **LIME**, which highlights which words contributed most to a classification decision.

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix by party

---

# 4 Results
The final model achieved the following:

- Accuracy: ~35%
- Macro F1-Score: ~0.345

**Insights:**
- AfD, FDP, and CSU were most distinguishable due to clear linguistic patterns
- SPD and Grüne were often confused, likely due to overlapping themes
- Non-affiliated users were hardest to classify due to lack of consistent language

The system successfully classifies tweets with reasonable accuracy, based only on textual features. It
highlights common linguistic patterns used by different political parties and demonstrates the feasibility of
lightweight political orientation prediction, though limitations remain.

---

# 5 Discussion
### Limitations

- Small dataset size, limited to ~1,000 tweets per party
- Sarcasm, irony, and ambiguity affect model performance
- Party identifiers may be context-dependent or implicit
- No metadata or user info included

### Ethical Considerations

- Risk of misclassification in political contexts
- Tools must be transparent and explainable
- Not intended for profiling or automated enforcement

### Future Improvements

- Use larger datasets across more timeframes
- Try more robust models like transformers
- Improve sarcasm and irony detection
- Integrate temporal or user context

---

# 6 Conclusion
This project demonstrates that party affiliation can be predicted from tweet text with moderate accuracy using classical machine learning. Our approach is modular, interpretable, and designed for educational use. The accompanying Streamlit app allows users to explore predictions interactively and understand the language features behind them.

The work supports transparency in political discourse and lays the groundwork for more advanced, context-aware systems.

---

*Created as part of the ML4B course — Machine Learning for Business: Advanced Concepts.*
