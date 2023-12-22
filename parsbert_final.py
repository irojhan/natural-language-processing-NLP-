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


# # Final DataSet for Analysis

# In[72]:


df_new_1 = df_new.iloc[:50000,:]
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
selected_dataset = pd.concat(selected_rows)
selected_dataset = selected_dataset.reset_index(drop=True)
selected_dataset.label.value_counts()


# In[73]:


x_train, x_test, y_train, y_test = train_test_split(selected_dataset['cleaned_comment'], selected_dataset['label'], test_size=0.2, random_state=42)


# In[74]:


labels = [1, 0, -1]
train, test = train_test_split(selected_dataset, test_size=0.1, random_state=1, stratify=selected_dataset['label'])
train, valid = train_test_split(train, test_size=0.1, random_state=1, stratify=train['label'])

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

x_train, y_train = train['cleaned_comment'].values.tolist(), train['label'].values.tolist()
x_valid, y_valid = valid['cleaned_comment'].values.tolist(), valid['label'].values.tolist()
x_test, y_test = test['cleaned_comment'].values.tolist(), test['label'].values.tolist()

print(train.shape)
print(valid.shape)
print(test.shape)


# In[75]:


# See the data number of sentence in each category 
from collections import Counter
cnt = Counter(y_train)
cnt = dict(cnt)
print(cnt)


# In[76]:


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


# In[41]:


#tensorflow


# In[77]:


from transformers import BertConfig, BertTokenizer
from transformers import TFBertModel, TFBertForSequenceClassification
from transformers import glue_convert_examples_to_features

import tensorflow as tf


# In[78]:


# general config
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

EPOCHS = 3
EEVERY_EPOCH = 1000
LEARNING_RATE = 2e-5
CLIP = 0.0

MODEL_NAME_OR_PATH = 'HooshvareLab/bert-fa-base-uncased'
#OUTPUT_PATH = '/content/bert-fa-base-uncased-sentiment-taaghceh/pytorch_model.bin'
#os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# In[79]:


label2id = {label: i for i, label in enumerate(labels)}
id2label = {v: k for k, v in label2id.items()}

print(f'label2id: {label2id}')
print(f'id2label: {id2label}')


# In[80]:


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
config = BertConfig.from_pretrained(
    MODEL_NAME_OR_PATH, **{
        'label2id': label2id,
        'id2label': id2label,
    })

print(config.to_json_string())


# In[81]:


class InputExample:
    """ A single example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """ Constructs a InputExample. """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def make_examples(tokenizer, x, y=None, maxlen=128, output_mode="classification", is_tf_dataset=True):
    examples = []
    y = y if isinstance(y, list) or isinstance(y, np.ndarray) else [None] * len(x)

    for i, (_x, _y) in tqdm(enumerate(zip(x, y)), position=0, total=len(x)):
        guid = "%s" % i
        label = int(_y)
        
        if isinstance(_x, str):
            text_a = _x
            text_b = None
        else:
            assert len(_x) == 2
            text_a = _x[0]
            text_b = _x[1]
        
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    
    features = glue_convert_examples_to_features(
        examples, 
        tokenizer, 
        maxlen, 
        output_mode=output_mode, 
        label_list=list(np.unique(y)))

    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    all_labels = []

    for f in tqdm(features, position=0, total=len(examples)):
        if is_tf_dataset:
            all_input_ids.append(tf.constant(f.input_ids))
            all_attention_masks.append(tf.constant(f.attention_mask))
            all_token_type_ids.append(tf.constant(f.token_type_ids))
            all_labels.append(tf.constant(f.label))
        else:
            all_input_ids.append(f.input_ids)
            all_attention_masks.append(f.attention_mask)
            all_token_type_ids.append(f.token_type_ids)
            all_labels.append(f.label)

    if is_tf_dataset:
        dataset = tf.data.Dataset.from_tensor_slices(({
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks,
            'token_type_ids': all_token_type_ids
        }, all_labels))

        return dataset, features
    
    xdata = [np.array(all_input_ids), np.array(all_attention_masks), np.array(all_token_type_ids)]
    ydata = all_labels

    return [xdata, ydata], features


# In[82]:


train_dataset_base, train_examples = make_examples(tokenizer, x_train, y_train, maxlen=128)
valid_dataset_base, valid_examples = make_examples(tokenizer, x_valid, y_valid, maxlen=128)

test_dataset_base, test_examples = make_examples(tokenizer, x_test, y_test, maxlen=128)
[xtest, ytest], test_examples = make_examples(tokenizer, x_test, y_test, maxlen=128, is_tf_dataset=False)


# In[83]:


for value in train_dataset_base.take(1):
    print(f'     input_ids: {value[0]["input_ids"]}')
    print(f'attention_mask: {value[0]["attention_mask"]}')
    print(f'token_type_ids: {value[0]["token_type_ids"]}')
    print(f'        target: {value[1]}')


# In[84]:


def get_training_dataset(dataset, batch_size):
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)

    return dataset

def get_validation_dataset(dataset, batch_size):
    dataset = dataset.batch(batch_size)

    return dataset


# In[85]:


train_dataset = get_training_dataset(train_dataset_base, TRAIN_BATCH_SIZE)
valid_dataset = get_training_dataset(valid_dataset_base, VALID_BATCH_SIZE)

train_steps = len(train_examples) // TRAIN_BATCH_SIZE
valid_steps = len(valid_examples) // VALID_BATCH_SIZE

train_steps, valid_steps


# In[86]:


def build_model(model_name, config, learning_rate=3e-5):
    model = TFBertForSequenceClassification.from_pretrained(model_name, config=config)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model


# In[87]:


model = build_model(MODEL_NAME_OR_PATH, config, learning_rate=LEARNING_RATE)


# In[88]:


get_ipython().run_cell_magic('time', '', "\nr = model.fit(\n    train_dataset,\n    validation_data=valid_dataset,\n    steps_per_epoch=train_steps,\n    validation_steps=valid_steps,\n    epochs=3,\n    verbose=1)\n\nfinal_accuracy = r.history['val_accuracy']\nprint('FINAL ACCURACY MEAN: ', np.mean(final_accuracy))")


# In[89]:


final_accuracy


# In[90]:


ev = model.evaluate(test_dataset_base.batch(TEST_BATCH_SIZE))
print()
print(f'Evaluation: {ev}')
print()

predictions = model.predict(xtest)
ypred = predictions[0].argmax(axis=-1).tolist()
labels = ["-1", "0", "1"]
print()
print(classification_report(ytest, ypred, target_names=labels))
print()

print(f'F1: {f1_score(ytest, ypred, average="weighted")}')


# In[ ]:





# In[ ]:





# In[ ]:




