#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('capture', '', '%pylab inline\nimport warnings\nwarnings.filterwarnings(\'ignore\')\nimport pandas as pd\nimport seaborn as sns\nsns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})\nfrom wordcloud import WordCloud, STOPWORDS\nfrom collections import Counter\nnp.random.seed(10)\n# Import required packages\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report\nfrom sklearn.metrics import f1_score\nfrom sklearn.utils import shuffle\nimport hazm\nfrom cleantext import clean\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom tqdm.notebook import tqdm\nimport os\nimport re\nimport json\nimport copy\nimport collections\n# Import required packages\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report\nfrom sklearn.metrics import f1_score\nfrom sklearn.utils import shuffle\nimport hazm\nfrom cleantext import clean\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom tqdm.notebook import tqdm\nimport os\nimport re\nimport json\nimport copy\nimport collections\nimport keras\nfrom keras.models import Sequential , load_model , Model\nfrom keras.layers import Dense , LSTM, Dropout , Conv1D , MaxPooling1D , Input, Reshape , Masking , TimeDistributed\nfrom keras.layers import Concatenate , BatchNormalization , Bidirectional , Activation , GlobalMaxPooling1D\nfrom keras.preprocessing.sequence import TimeseriesGenerator\nfrom keras.callbacks import History , ModelCheckpoint\nfrom keras.preprocessing.text import Tokenizer\nfrom keras.constraints import maxnorm\nfrom keras.regularizers import l1_l2\nfrom keras.utils import pad_sequences\nfrom keras.layers import Embedding\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.metrics import mean_squared_error , precision_score , accuracy_score\nfrom sklearn.metrics import recall_score , confusion_matrix, roc_curve, auc\nfrom nltk.tokenize import TreebankWordTokenizer\nfrom nltk.stem.wordnet import WordNetLemmatizer\nfrom nltk.stem import PorterStemmer\nfrom tensorflow.keras.utils import plot_model\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout')


# In[2]:


# loading stopwords
stopwords = []
file = open('stopwords.txt', encoding='utf-8').read()
[stopwords.append(x) for x in file.split()]
stopwords = set(stopwords)


# In[3]:


# loading the dataset
df1=pd.read_excel('data1.xlsx')
df=df1.drop(["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14", "Unnamed: 15", "Unnamed: 16", "Unnamed: 17", "Unnamed: 18", "Unnamed: 19", "Unnamed: 20"], axis=1, inplace=False)
df.head()


# In[4]:


# calculate the length of comments based on their words
df['comment_len_by_words'] = df['comment'].apply(lambda t: len(hazm.word_tokenize(str(t))) if isinstance(t, str) else np.nan)


# In[5]:


min_max_len = df["comment_len_by_words"].min(), df["comment_len_by_words"].max()
print(f'Min: {min_max_len[0]} \tMax: {min_max_len[1]}')


# In[6]:


def data_gl_than(data, less_than=100.0, greater_than=0.0, col='comment_len_by_words'):
    data_length = data[col].values
    data_glt = sum([1 for length in data_length if greater_than < length <= less_than])
    data_glt_rate = (data_glt / len(data_length)) * 100
    print(f'Texts with word length of greater than {greater_than} and less than {less_than} includes {data_glt_rate:.2f}% of the whole!')


# In[7]:


data_gl_than(df, 300, 3)


# In[8]:


minlim, maxlim = 3, 300


# In[9]:


# remove comments with the length of fewer than three words
df['comment_len_by_words'] = df['comment_len_by_words'].apply(lambda len_t: len_t if minlim < len_t <= maxlim else None)
data = df.dropna(subset=['comment_len_by_words'])
data = df.reset_index(drop=True)


# In[10]:


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# removing stopwords function

def remove_general_stopwords(text):
    filtered_tokens = [token for token in text.split() if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def cleaning(text):
    text = text.strip()
    
    # regular cleaning
    text = clean(text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="0",
        replace_with_currency_symbol="",
    )

    # cleaning htmls
    text = cleanhtml(text)
    
    # normalizing
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)
    
    #removing stopwords
    text = remove_general_stopwords(text)
    # removing wierd patterns
    wierd_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        u"\u200c"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)
    
    text = wierd_pattern.sub(r'', text)
    
    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    
    return text


# In[11]:


# cleaning comments
def cleaning(comment):
    if isinstance(comment, str):
        return comment.strip()
    return ''
data['cleaned_comment'] = data['comment'].apply(cleaning)
# calculate the length of comments based on their words
data['cleaned_comment_len_by_words'] = data['cleaned_comment'].apply(lambda t: len(hazm.word_tokenize(str(t))) if isinstance(t, str) else np.nan)
# remove comments with the length of fewer than three words
data['cleaned_comment_len_by_words'] = data['cleaned_comment_len_by_words'].apply(lambda len_t: len_t if minlim < len_t <= maxlim else len_t)
data = data.dropna(subset=['cleaned_comment_len_by_words'])
data = data.reset_index(drop=True)
data.head()


# In[12]:


from hazm import Normalizer, Lemmatizer
normalizer = Normalizer()
lemmatizer = Lemmatizer()
def preprocess(df, stopwords):
    df['cleaned_comment'] = df['cleaned_comment'].apply(lambda x: re.sub(r'[\s]{2,}', ' ', x))
    df['cleaned_comment'] = df['cleaned_comment'].apply(lambda x: x.replace("\n", " "))
    df['cleaned_comment'] = df['cleaned_comment'].apply(lambda x: ' '.join([normalizer.normalize(word) for word in hazm.word_tokenize(x)]))
#     df['cleaned_comment'] = df['cleaned_comment'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in hazm.word_tokenize(x)]))
    df['cleaned_comment'] = df['cleaned_comment'].apply(lambda x: re.sub('[^\u0600-\u06FF\s]', ' ', x))
    df['cleaned_comment'] = df['cleaned_comment'].apply(lambda x: ' '.join([word for word in hazm.word_tokenize(x) if word not in stopwords]))
    return df

df_new = preprocess(data, stopwords)


# In[13]:


df_new['cleaned_comment_len_by_words'] = data['cleaned_comment'].apply(lambda t: len(hazm.word_tokenize(str(t))) if isinstance(t, str) else np.nan)
df_new['label'] = df_new['polarity']
df_new = df_new.drop(["comment","polarity","sentenceID","comment_len_by_words","cleaned_comment_len_by_words"], axis=1, inplace=False)
df_new.iloc[1]['cleaned_comment']


# In[14]:


# Functions that do text preprocessing

# We use the treebank tokenizer.
treebank_tokenizer = TreebankWordTokenizer()
# lemmatization.
lem = WordNetLemmatizer()
# Stemming.
stemmer = PorterStemmer()

def tokenize(x):
    return ' '.join(treebank_tokenizer.tokenize(x))

def lemmatize(x):
    return ' '.join([lem.lemmatize(s) for s in x.split(' ')])

def stem(x):
    return ' '.join([stemmer.stem(s) for s in x.split(' ')])

# Apply the transformations :
sentences = df_new.cleaned_comment.apply(tokenize)
sentences = sentences.apply(stem)


# # XGBoost

# In[15]:


df_new_1 = df_new.iloc[:20000,:]
# Determine the desired number of rows for each label
desired_count = min(df_new_1['label'].value_counts())

# Create a list to store the selected rows
selected_rows = []
for label in df_new_1['label'].unique():
    # Filter the DataFrame to select only rows with the current label
    label_rows = df_new_1[df_new_1['label'] == label]
    
    # Sample the desired number of rows for this label
    sampled_rows = label_rows.sample(n=desired_count, random_state=42)
    
    # Append the selected rows to the list
    selected_rows.append(sampled_rows)

# Concatenate the selected rows back into a single DataFrame
selected_df = pd.concat(selected_rows)
selected_df = selected_df.reset_index(drop=True)
selected_df.label.value_counts()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from hazm import word_tokenize, Lemmatizer

# Tokenize and lemmatize Persian text

# Split the data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(selected_df['cleaned_comment'], selected_df['label'], test_size=0.2, random_state=42)
labels_train = labels_train + 1
labels_test = labels_test + 1

# Create the TF-IDF vectorizer with additional parameters
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=None, max_df=0.8, min_df=2)  # You can adjust these parameters
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# Perform hyperparameter tuning using GridSearchCV for XGBoost
param_grid_xgboost = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
}
grid_search_xgboost = GridSearchCV(XGBClassifier(), param_grid_xgboost, cv=3)
grid_search_xgboost.fit(X_train, labels_train)
best_xgboost_model = grid_search_xgboost.best_estimator_

# Predict on the test set using XGBoost
predicted_labels_xgboost = best_xgboost_model.predict(X_test)

# Evaluate the XGBoost model
accuracy_xgboost = accuracy_score(labels_test, predicted_labels_xgboost)
print(f"Best XGBoost Model's Accuracy: {accuracy_xgboost:.2f}")

print("Classification Report (XGBoost):")
print(classification_report(labels_test, predicted_labels_xgboost))

print("Confusion Matrix (XGBoost):")
print(confusion_matrix(labels_test, predicted_labels_xgboost))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier

# Create and train the XGBoost model
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)
xgb_model.fit(X_train, labels_train)

# Predict probabilities on the test set using XGBoost
probabilities = xgb_model.predict_proba(X_test)

# Plot ROC curve for each label
plt.figure()

for i, label in enumerate(xgb_model.classes_):
    label_probs = probabilities[:, i]
    fpr, tpr, _ = roc_curve((labels_test == label).astype(int), label_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Label (XGBoost)')
plt.legend(loc='lower right')
plt.show()

