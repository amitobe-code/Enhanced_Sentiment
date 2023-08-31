# %%
import joblib
import flair
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

# %%
from sklearn.base import BaseEstimator, TransformerMixin

# %%
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
flair_model = flair.models.TextClassifier.load('en-sentiment')

def process_flair(dialogue):   # returning the flair score
    sentence = flair.data.Sentence(dialogue)
    flair_model.predict(sentence)
    label = sentence.labels[0].value
    score = sentence.labels[0].score
    if label == 'POSITIVE':
        return score
    elif label == 'NEGATIVE':
        return -score



def return_sentiment(txt):  # returning single-sentence BERT score
    encoded_input = tokenizer(txt, return_tensors='pt',padding=True,truncation=True)
    output = bert_model(**encoded_input)
    score = output[0][0].detach().numpy() 
    scores = softmax(score)
    if np.argsort(scores)[2] == 1:
        return 0
    else:
        return (np.argsort(scores)[2]-1)*scores[np.argsort(scores)[2]]
    
def tb_score(txt):
    sen = TextBlob(txt)
    return pd.Series({'tb': sen.sentiment.polarity})

def cal_vader_textblob_bert_flair(txt):
    tb_score = TextBlob(txt).sentiment.polarity
    obj = SentimentIntensityAnalyzer()
    vader_score = obj.polarity_scores(txt)['compound']
    flair_score = process_flair(txt)
    bert_score = return_sentiment(txt)
    #prob = logmodel3.predict_proba([[tb_score,vader_score,flair_score,bert_score]])[0]      
    return np.array([[vader_score, tb_score,bert_score,flair_score]])



# %%
from sklearn.preprocessing import FunctionTransformer

# %%
func_tfmr = FunctionTransformer(func=cal_vader_textblob_bert_flair)

# %%
# This is the wrapper
class PredictionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.model.predict_proba(X)

# %%
def fused_func(arr):
    negative = (arr[0][0]+arr[0][2])/2
    positive = (arr[0][1]+arr[0][3])/2
    #if positive >= negative:
    #    return 1
    #else:
    #    return -1
    if arr[0][1]>=arr[0][0]:
        return 1
    else:
        return -1

# %%
final_pipeline = joblib.load('final_pipe.pkl')

# %%
# Get input as a sentence from the user
input_sentence = input("Enter a sentence: ")



# %%
senti = final_pipeline.transform(input_sentence)

# %%
# Print the input sentence
print("You entered:\"",input_sentence, "\"  .....Sentiment is:", senti)

# %%



