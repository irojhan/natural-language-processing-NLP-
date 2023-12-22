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


# # balancing

# In[25]:


# General
import numpy as np
import pandas as pd
import codecs
#from google.colab import files
# Word Embedding
from gensim.models import KeyedVectors
# Keras
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import GlobalMaxPool1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences

# Preprocessing
from stopwords_guilannlp import stopwords_output
from hazm import *
# Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from keras.utils import plot_model
# Measuring metrics
from sklearn.metrics import f1_score
# Import required packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import hazm
from cleantext import clean
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm
import os
import re
import json
import copy
import collections
import keras
from keras.models import Sequential , load_model , Model
from keras.layers import Dense , LSTM, Dropout , Conv1D , MaxPooling1D , Input, Reshape , Masking , TimeDistributed
from keras.layers import Concatenate , BatchNormalization , Bidirectional , Activation , GlobalMaxPooling1D
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import History , ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.constraints import maxnorm
from keras.regularizers import l1_l2
from keras.utils import pad_sequences
from keras.layers import Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , precision_score , accuracy_score
from sklearn.metrics import recall_score , confusion_matrix, roc_curve, auc
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


# In[203]:


df_new_1 = df_new.iloc[:80000,:]
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


# In[204]:


x_train, x_test, y_train, y_test = train_test_split(selected_df['cleaned_comment'], selected_df['label'], test_size=0.2, random_state=42)


# In[205]:


# See the data number of sentence in each category 
from collections import Counter
cnt = Counter(y_train)
cnt = dict(cnt)
print(cnt)


# In[206]:


import matplotlib.pyplot as plt
labels = list(cnt.keys())
sizes = list(cnt.values())
colors = ['#3fba36', '#66b3ff','#ffcc99','#ff9999', '#d44444']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
# Decomment following line if you want to save the figure
# plt.savefig('distribution.png')
plt.show()


# In[207]:


puncs = ['،', '.', ',', ':', ';', '"']
normalizer = Normalizer()
lemmatizer = Lemmatizer()

# turn a doc into clean tokens
def clean_doc(doc):
    doc = normalizer.normalize(doc) # Normalize document using Hazm Normalizer
    tokenized = word_tokenize(doc)  # Tokenize text
    tokens = []
    for t in doc:
      temp = t
#       for p in puncs:
#         temp = temp.replace(p, '')
      tokens.append(temp)
    # tokens = [w for w in tokens if not w in stop_set]    # Remove stop words
    tokens = [w for w in tokens if not len(w) <= 1]
    tokens = [w for w in tokens if not w.isdigit()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens] # Lemmatize sentence words using Hazm Lemmatizer
    tokens = ' '.join(tokens)
    return tokenized


# In[208]:


from gensim.models import KeyedVectors

EMBEDDING_FILE = 'wiki.fa.vec'

ft_model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE)


# In[209]:


def import_with_gensim(file_address):
  # Creating the model
  ft_model = KeyedVectors.load_word2vec_format(file_address)
  # Getting the tokens
  ft_words = []

  #for ft_word in ft_model.vocab:
  for ft_word in ft_model.key_to_index: #-- gensim 3.x to gensim 4.00 (Hamid)

      ft_words.append(ft_word)
  return ft_model, ft_words
  
ft_model_1, ft_words_1 = import_with_gensim(EMBEDDING_FILE)


# In[210]:


# FastText embedding dimensionality
embed_size = 300


# In[211]:


# We get the mean and standard deviation of the embedding weights so that we could maintain the
# same statistics for the rest of our own random generated weights.
embedding_list = list()
for w in ft_words_1:
  embedding_list.append(ft_model_1[w])

all_embedding = np.stack(embedding_list)
emb_mean, emb_std = all_embedding.mean(), all_embedding.std()


# In[212]:


# Apply preprocessing step to training data
train_docs = np.empty_like(x_train)
for index, document in enumerate(x_train):
  train_docs[index] = clean_doc(document)


# In[213]:


# Applying preprocessing step to test data
test_docs = np.empty_like(x_test)
for index, document in enumerate(x_test):
  test_docs[index] = clean_doc(document)


# In[214]:


num_words = 2000

# Create the tokenizer
tokenizer = Tokenizer(num_words=num_words)

# fFt the tokenizer on the training documents
tokenizer.fit_on_texts(x_train)
# Find maximum length of training sentences
max_length = max([len(s.split()) for s in x_train])
max_length


# In[215]:


# Embed training sequences
encoded_docs = tokenizer.texts_to_sequences(train_docs)

# Pad embeded training sequences
x_train_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# In[216]:


# Define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index)


# In[217]:


unique_words = list(tokenizer.word_index.keys())

print("Unique Words:", unique_words)


# In[218]:


# We are going to set the embedding size to the pre-trained dimension as we are replicating it
nb_words = len(tokenizer.word_index)

# the size will be Number of Words in Vocab X Embedding Size
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

# With the newly created embedding matrix, we'll fill it up with the words that we have in both
# our own dictionary and loaded pre-trained embedding.
embeddedCount = 0
for word, i in tokenizer.word_index.items():
    i -= 1
    # then we see if this word is in glove's dictionary, if yes, get the corresponding weights
    if word in ft_model.key_to_index:
        embedding_vector = ft_model[word]
        # and store inside the embedding matrix that we will train later on.
        embedding_matrix[i] = embedding_vector
        embeddedCount += 1
    else:   # Unknown words
        embedding_vector = ft_model['subdivision_name']
        embedding_matrix[i] = embedding_vector
        embeddedCount += 1

print('total embedded:', embeddedCount, 'common words')
print('Embedding matrix shape:', embedding_matrix.shape)


# In[219]:


# Embed testing sequences
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# Pad testing sequences
x_test_padded = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# In[220]:


from keras.utils.np_utils import to_categorical

# Assuming y_train and y_test contain labels (-1, 0, 1)
num_classes = 3

# Prepare labels for categorical prediction
categorical_y_train = to_categorical(y_train, num_classes)
categorical_y_test = to_categorical(y_test, num_classes)


# In[221]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Function to create the CNN model
def create_cnn_model(learning_rate=0.001, dropout_rate=0.1):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=16, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(3, activation='softmax'))

    #optimizer = adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[categorical_accuracy])
    return model

# Create a KerasClassifier
cnn_classifier = KerasClassifier(build_fn=create_cnn_model, epochs=10, batch_size=64, verbose=0)

# Define hyperparameters for grid search
param_grid = {
    'learning_rate': [0.001,0.002],
    'dropout_rate': [0.1,0.15]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=cnn_classifier, param_grid=param_grid, cv=3, verbose=2)
grid_search_result = grid_search.fit(x_train_padded, categorical_y_train)

# Print best hyperparameters
print("Best Parameters: ", grid_search_result.best_params_)

# Evaluate the best model on the test set
best_cnn_model = grid_search_result.best_estimator_
# Make predictions on the test set
y_pred = best_cnn_model.predict(x_test_padded)

# Calculate accuracy
accuracy = accuracy_score(categorical_y_test.argmax(axis=1), y_pred)
# loss_cnn, acc_cnn = best_cnn_model.evaluate(x_test_padded, categorical_y_test, verbose=0)
print('Test Accuracy: {:.2f}%'.format(accuracy * 100))

# Predict on the test set
predict_x_cnn = best_cnn_model.predict(x_test_padded)
y_pred_cnn = np.argmax(predict_x_cnn)


# In[222]:


# Evaluate performance metrics
from sklearn.metrics import classification_report, confusion_matrix
# Convert y_pred to multilabel-indicator format
from sklearn.preprocessing import label_binarize

# Assuming y_pred contains class labels (0, 1, 2, etc.)
y_pred_multilabel = label_binarize(y_pred, classes=np.arange(3))  # Adjust the number of classes (3) as needed

# Now you can calculate the classification report
print("Classification Report:")
print(classification_report(categorical_y_test, y_pred_multilabel))

from sklearn.metrics import multilabel_confusion_matrix

# Calculate and print the multilabel confusion matrix
multilabel_cm = multilabel_confusion_matrix(categorical_y_test, y_pred_multilabel)

# Print each confusion matrix for individual labels
for i, label_cm in enumerate(multilabel_cm):
    print(f"Confusion Matrix for Label {i}:")
    print(label_cm)


# In[223]:


# Print the architecture of the best CNN model
best_cnn_model.model.summary()


# In[79]:


#from keras.utils import plot_model

# Plot the architecture of the best CNN model
#plot_model(best_cnn_model.model, to_file='best_cnn_model.png', show_shapes=True, show_layer_names=True)


# In[224]:


from keras.layers import Bidirectional, LSTM

# Function to create a Bi-LSTM model
def create_bi_lstm_model(learning_rate=0.001, dropout_rate=0.1):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[categorical_accuracy])
    return model

# Create a KerasClassifier for the Bi-LSTM model
bi_lstm_classifier = KerasClassifier(build_fn=create_bi_lstm_model, epochs=10, batch_size=64, verbose=0)

# Define hyperparameters for grid search
param_grid = {
    'learning_rate': [0.001, 0.002],
    'dropout_rate': [0.1, 0.15]
}

# Perform grid search with cross-validation
grid_search_bi_lstm = GridSearchCV(estimator=bi_lstm_classifier, param_grid=param_grid, cv=3, verbose=2)
grid_search_result_bi_lstm = grid_search_bi_lstm.fit(x_train_padded, categorical_y_train)

# Print best hyperparameters for Bi-LSTM model
print("Best Parameters (Bi-LSTM): ", grid_search_result_bi_lstm.best_params_)

# Evaluate the best Bi-LSTM model on the test set
best_bi_lstm_model = grid_search_result_bi_lstm.best_estimator_
y_pred_bi_lstm = best_bi_lstm_model.predict(x_test_padded)

# Calculate accuracy for Bi-LSTM model
accuracy_bi_lstm = accuracy_score(categorical_y_test.argmax(axis=1), y_pred_bi_lstm)
print('Test Accuracy (Bi-LSTM): {:.2f}%'.format(accuracy_bi_lstm * 100))


# In[225]:


from keras.layers import Bidirectional, LSTM
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np

# Convert y_pred to multilabel-indicator format
y_pred_multilabel = label_binarize(y_pred_bi_lstm, classes=np.arange(3))  # Adjust the number of classes (3) as needed

# Now you can calculate the classification report
print("Classification Report (Bi-LSTM):")
print(classification_report(categorical_y_test, y_pred_multilabel))

# Calculate and print the multilabel confusion matrix
multilabel_cm = multilabel_confusion_matrix(categorical_y_test, y_pred_multilabel)

# Print each confusion matrix for individual labels
for i, label_cm in enumerate(multilabel_cm):
    print(f"Confusion Matrix for Label {i} (Bi-LSTM):")
    print(label_cm)


# In[226]:


best_bi_lstm_model.model.summary()


# In[ ]:





# In[ ]:





# In[ ]:




