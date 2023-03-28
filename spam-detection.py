# # Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# For spliting dataset
from sklearn.model_selection import train_test_split

# For preprocessing data
import re
import nltk
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For building the model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Embedding, Dropout,
    GlobalAveragePooling1D, Input
)
from tensorflow.keras.models import Model





SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# # Data preprocessing
# ## File path
data_path = "spam.csv"


# ## Read data from csv file
df = pd.read_csv(data_path, encoding='ISO-8859-1')
df = df[['v1', 'v2']]
df.columns = ['Category', 'Content']
df.head()



# ## Text cleaner Fnction
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


def clean_text(text):
    # Remove punctuation from text
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    # Get Origin of words (books -> book)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text


# ## Clean the text
df['Clean_text'] = df['Content'].apply(lambda x: clean_text(x))
df.head()


# ## Category column string to int8
# {ham:0, spam:1}
df['label'] = df['Category'].astype('category').cat.codes
df.head()


# ## Spliting data
X_train,X_test,y_train,y_test=train_test_split(
    df['Clean_text'],df['label'],test_size=0.25,random_state=SEED)


# # Tokenize

# ## Define the parameters
max_len=50 # max words in every input string
trunc_type='post' # if number of words are more than max_len then trunct from the last of the string
padding_type='post' # if number of words are less than max_len then add
oov_token_1='<OOV>'# out of vocabulary token
vocab_size=500


# ## Text to numerical
tokenizer=Tokenizer(num_words=vocab_size,char_level=False,oov_token=oov_token_1)
tokenizer.fit_on_texts(X_train)





print(f"Total token : {len(tokenizer.word_index)}")
for token_word in tokenizer.word_index:
    print(f"{token_word}: {tokenizer.word_index[token_word]}")
    if tokenizer.word_index[token_word] >10:
        break


# ## Tokenize and padding 
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train,maxlen=max_len,padding=padding_type,truncating=trunc_type)
X_test=tokenizer.texts_to_sequences(X_test)
X_test=pad_sequences(X_test,maxlen=max_len,padding=padding_type,truncating=trunc_type)
X_train.shape, X_test.shape


# # Modeling
input_shape = X_train.shape[1:]
input_shape




input = Input(shape=input_shape)
x = Embedding(input_dim=vocab_size, output_dim=12)(input)
x = GlobalAveragePooling1D()(x)
x = Dense(24, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input, outputs=output)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])



model.summary()




# ## Fit the model
num_epochs=50
# If val_loss increases five times it will stop model fitting
early_stop=EarlyStopping(monitor='val_loss',patience=5)
history=model.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    validation_data=(X_test,y_test),
    callbacks=[early_stop]
)


# ## Evaluate
model.evaluate(X_test, y_test)


# ## plot_loss_curves
def plot_loss_curves(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();





plot_loss_curves(history)




