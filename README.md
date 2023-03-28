# Email/Sms Spam Detection

This model can separate spam email or SMS. Its accuracy is more than 98%.[Link](https://www.kaggle.com/code/hsakash/spam-detection)

# Data preprocessing
* Remove punctuation.
* Word stemming. (books -> book)
* Tokenize the string.
* Sentence padding using pad_sequences.

## Ham & smap Distribution
![](https://github.com/HSAkash/Email-Sms-Spam-Detection/raw/main/related_images/distribution.png)

## Model
![](https://github.com/HSAkash/Email-Sms-Spam-Detection/raw/main/related_images/model.png)
<br>
<br>
Layer parameters:

* Input shape (50,)
* Embedding(input_dim=500, output_dim=12)
* Dense(24) Activation=relu
* Dropout 0.2
* Output layer activation sigmoid

## Compile model

Compile the model with the following options:

* Loss function (categorical_crossentropy)
* optimizer (Adam lr=0.001)
* metrics (accuracy)


## Loss Accuracy curves
![](https://github.com/HSAkash/Email-Sms-Spam-Detection/raw/main/related_images/loss_accuracy_curves.png)




# Requirements
* matplotlib 3.5.2
* nltk 3.8.1
* numpy 1.23.5
* pandas 1.5.3
* scikit-learn 1.2.1
* tensorflow 2.11.0

# Demo

Here is how to run the potato disease program using the following command line.

Clone code from github
```
git clone git@github.com:HSAkash/Email-Sms-Spam-Detection.git
```
Install requirements
```
pip install -r requirements.txt
```
Run the code
```
python spam-detection.py
```

# Links (dataset & code)
* [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* [code](https://www.kaggle.com/code/hsakash/spam-detection)

# Author
HSAkash
* [Linkedin](https://www.linkedin.com/in/hemel-akash/)
* [Kaggle](https://www.kaggle.com/hsakash)
* [Facebook](https://www.facebook.com/hemel.akash.7/)
