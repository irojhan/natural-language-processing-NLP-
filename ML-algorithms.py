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


fig = go.Figure()

fig.add_trace(go.Histogram(
    x=data['comment_len_by_words']
))

fig.update_layout(
    title_text='Distribution of word counts within comments',
    xaxis_title_text='Word Count',
    yaxis_title_text='Frequency',
    bargap=0.2,
    bargroupgap=0.2)

fig.show()


# In[11]:


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


# In[12]:


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


# In[13]:


data.iloc[1]['cleaned_comment']


# In[14]:


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


# In[15]:


df_new['cleaned_comment_len_by_words'] = data['cleaned_comment'].apply(lambda t: len(hazm.word_tokenize(str(t))) if isinstance(t, str) else np.nan)
df_new['label'] = df_new['polarity']
df_new = df_new.drop(["comment","polarity","sentenceID","comment_len_by_words","cleaned_comment_len_by_words"], axis=1, inplace=False)
df_new.iloc[1]['cleaned_comment']


# In[16]:


#handeling unbalanced data
fig = go.Figure()

groupby_label = df_new.groupby('label')['label'].count()

fig.add_trace(go.Bar(
    x=list(sorted(groupby_label.index)),
    y=groupby_label.tolist(),
    text=groupby_label.tolist(),
    textposition='auto'
))

fig.update_layout(
    title_text='Distribution of label within comments [DATA]',
    xaxis_title_text='Label',
    yaxis_title_text='Frequency',
    bargap=0.2,
    bargroupgap=0.2)

fig.show()


# In[17]:


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


# In[18]:


sentences


# # SVM

# In[19]:


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


# In[20]:


selected_df


# In[21]:


import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from hazm import word_tokenize, Lemmatizer

# Tokenize and lemmatize Persian text

# Split the data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(selected_df['cleaned_comment'], selected_df['label'], test_size=0.2, random_state=42)

# Create the TF-IDF vectorizer with additional parameters
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=None, max_df=0.8, min_df=2)  # You can adjust these parameters
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=3)
grid_search.fit(X_train, labels_train)
best_svm_model = grid_search.best_estimator_

# Predict on the test set
predicted_labels = best_svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(labels_test, predicted_labels)
print(f"Best Model's Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(labels_test, predicted_labels))

print("Confusion Matrix:")
print(confusion_matrix(labels_test, predicted_labels))


# In[22]:


# Create and train the SVM model
svm_model = SVC(kernel='linear', probability=True)  # Set probability=True to enable probability estimates
svm_model.fit(X_train, labels_train)

# Predict probabilities on the test set
probabilities = svm_model.predict_proba(X_test)

# Plot ROC curve for each label
plt.figure()

for i, label in enumerate(svm_model.classes_):
    label_index = np.where(svm_model.classes_ == label)[0][0]
    label_probs = probabilities[:, label_index]
    fpr, tpr, _ = roc_curve(labels_test, label_probs, pos_label=label)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Label')
plt.legend(loc='lower right')
plt.show()


# # Random Forest

# In[23]:


from sklearn.ensemble import RandomForestClassifier
# Create the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],     # You can adjust these values
    'max_depth': [10, 20, 30],            # You can adjust these values
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=3)
grid_search.fit(X_train, labels_train)

# Get the best Random Forest model from grid search
best_rf_model = grid_search.best_estimator_

# Predict probabilities on the test set
probabilities = best_rf_model.predict_proba(X_test)

# Plot ROC curve for each label
plt.figure()

unique_labels = np.unique(labels_test)
for label in unique_labels:
    label_index = np.where(unique_labels == label)[0][0]
    label_probs = probabilities[:, label_index]
    fpr, tpr, _ = roc_curve(labels_test, label_probs, pos_label=label)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Label')
plt.legend(loc='lower right')
plt.show()

# Predict on the test set with the final decision
predicted_labels = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(labels_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(labels_test, predicted_labels))

print("Confusion Matrix:")
print(confusion_matrix(labels_test, predicted_labels))


# # Multinomial NB

# In[24]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix

# Create and train the Multinomial Naive Bayes classifier
nb_model = MultinomialNB()

# Define the parameter grid for grid search
param_grid = {
    'alpha': [0.1, 0.5, 1.0]  # You can adjust these values
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(nb_model, param_grid, cv=3)
grid_search.fit(X_train, labels_train)
best_nb_model = grid_search.best_estimator_

# Predict probabilities on the test set
probabilities = best_nb_model.predict_proba(X_test)

# Plot ROC curve for each label
plt.figure()

for i, label in enumerate(best_nb_model.classes_):
    label_index = np.where(best_nb_model.classes_ == label)[0][0]
    label_probs = probabilities[:, label_index]
    fpr, tpr, _ = roc_curve(labels_test, label_probs, pos_label=label)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Label (Multinomial Naive Bayes)')
plt.legend(loc='lower right')
plt.show()

# Predict on the test set with the final decision
predicted_labels = best_nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(labels_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(labels_test, predicted_labels))

print("Confusion Matrix:")
print(confusion_matrix(labels_test, predicted_labels))


# # Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix

# Define the parameter grid for grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # You can adjust these values
    'penalty': ['l1', 'l2']  # You can also try 'elasticnet' with different l1_ratio
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3)
grid_search.fit(X_train, labels_train)
best_lr_model = grid_search.best_estimator_

# Predict probabilities on the test set
probabilities = best_lr_model.predict_proba(X_test)

# Plot ROC curve for each label
plt.figure()

for i, label in enumerate(best_lr_model.classes_):
    label_index = np.where(best_lr_model.classes_ == label)[0][0]
    label_probs = probabilities[:, label_index]
    fpr, tpr, _ = roc_curve(labels_test, label_probs, pos_label=label)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Label (Logistic Regression)')
plt.legend(loc='lower right')
plt.show()

# Predict on the test set with the final decision
predicted_labels = best_lr_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(labels_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(labels_test, predicted_labels))

print("Confusion Matrix:")
print(confusion_matrix(labels_test, predicted_labels))


# # XGBoost

# In[26]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

class_labels = [-1, 0, 1]  # Include all unique labels from your dataset
# create a key finder based on label 2 id and id to label
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {v: k for k, v in label2id.items()}
mapped_labels = np.array([label2id for label in selected_df['label']])

# Split the data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(selected_df['cleaned_comment'], mapped_labels, test_size=0.2, random_state=42)


# In[27]:


# Create the TF-IDF vectorizer with additional parameters
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=None, max_df=0.8, min_df=0.2)
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)


# In[28]:


# Apply the label mapping to your labels
mapped_labels_train = np.array([mapped_labels for label in labels_train])
mapped_labels_test = np.array([mapped_labels for label in labels_test])


# In[29]:


# Create the TF-IDF vectorizer with additional parameters
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=None, max_df=0.8, min_df=0.2)
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# Create the XGBoost classifier
xgb_model = XGBClassifier(random_state=42, num_class=len(class_labels))

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, cv=3)
grid_search.fit(X_train, mapped_labels_train)

# Get the best XGBoost model from grid search
best_xgb_model = grid_search.best_estimator_

# Predict on the test set with the final decision
predicted_labels_encoded = best_xgb_model.predict(X_test)
predicted_labels = [label for label, index in label_mapping.items() if index in predicted_labels_encoded]
# Evaluate the model
accuracy = accuracy_score(labels_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(labels_test, predicted_labels, target_names=class_labels))

print("Confusion Matrix:")
print(confusion_matrix(labels_test, predicted_labels))


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import numpy as np

labels = [-1,0,1]  # Include all unique labels from your dataset
texts_train, texts_test, labels_train, labels_test = train_test_split(selected_df['cleaned_comment'], selected_df['label'], test_size=0.2, stratify = labels)

# Split the data into training and testing sets
#texts_train, texts_test, labels_train, labels_test = train_test_split(selected_df['cleaned_comment'], selected_df['label'], test_size=0.2, random_state=42)

# Create the TF-IDF vectorizer with additional parameters
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words=None, max_df=0.8, min_df=0.2)
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# Create the XGBoost classifier
xgb_model = XGBClassifier(random_state=42, num_class=len(class_labels))

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, cv=3)
grid_search.fit(X_train, labels_train)  # Use labels_train directly

# Get the best XGBoost model from grid search
best_xgb_model = grid_search.best_estimator_

# Predict on the test set with the final decision
predicted_labels = best_xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(labels_test, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(labels_test, predicted_labels, target_names=class_labels))

print("Confusion Matrix:")
print(confusion_matrix(labels_test, predicted_labels))


# In[ ]:


pip uninstall xgboost 


# In[ ]:


pip install xgboost==0.90

