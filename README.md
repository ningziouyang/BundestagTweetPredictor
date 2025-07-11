# 1 Introduction
- Motivation:  
In today’s digital world, political communication increasingly takes place on social media platforms such as
Twitter. Politicians, parties, journalists, and citizens all participate in debates using short, emotionally
charged text formats. These messages rarely include clear party identifiers, making it difficult to assign
statements to political affiliations.
Our project explores whether it is possible to predict the party affiliation associated with a tweet using
only the textual content. The goal is to build a machine learning model that classifies tweets by party
labels, uncovering patterns in political language and enabling automated analysis of political discourse
- Research question:  
Can political orientation be detected from tweets using only their text without additional metadata?
# 2 Related Work
Previous projects have used NLP and machine learning to analyze political communication. Some studies relied on parliamentary speeches, party manifestos, or user profiles. Others explored sentiment and stance classification or ideological clustering. Our work is inspired by those approaches but takes a minimalist route: no metadata, no user history, and no time-based features,  just raw tweet content.
# 3 Methodology
## 3.1 General Methodology
We began by loading a dataset of tweets labeled with political party affiliations. We designed an
experimental pipeline that:
- Cleans and processes tweet text
- Extracts relevant linguistic features
- Trains and evaluates multiple classifiers
## 3.2 Data Understanding and Preparation
The dataset consists of German tweets, each labeled with a political party. Key challenges included short and
informal language, use of slang or hashtags, and limited context per tweet.
We applied several preprocessing steps:
- Lowercasing and punctuation removal
- Stopword removal
- Tokenization
- Vectorization using TF-IDF
- Semantic feature extraction using BERT
## 3.3 Modeling and Evaluation
We trained and compared several machine learning models:
- Logistic Regression (baseline)
- Logistic Regression with engineered features (TF-IDF + BERT)
Models were evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score.
The best-performing model was a logistic regression using a combination of TF-IDF and semantic embeddings from
BERT.
# 4 Results
The system successfully classifies tweets with reasonable accuracy, based only on textual features. It
highlights common linguistic patterns used by different political parties and demonstrates the feasibility of
lightweight political orientation prediction.
# 5 Discussion
This project demonstrates promising results, but several limitations remain:
- The dataset is relatively small and may not generalize
- Political messaging can be subtle, ironic, or context-dependent
- Tweets from individuals unaffiliated with parties may introduce noise
- Model performance is sensitive to the quality of preprocessing and feature selection
From an ethical standpoint, political classification systems must be used with caution. Automated labeling can
lead to misinterpretation, manipulation, or profiling. Transparency and explainability of such systems are
crucial.
# 6 Conclusion
Our system shows that political party affiliation can be predicted to a reasonable extent using only tweet
text. It uses a modular and interpretable architecture, making it suitable for educational use or as a
foundation for further political language research.
Future improvements could involve:
- Using more advanced deep learning models
- Expanding datasets to include more parties and domains
- Integrating irony or sarcasm detection
- Analyzing temporal evolution of political language
