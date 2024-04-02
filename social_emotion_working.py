import numpy as np
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('train_data.csv')
dataset1.iloc[:,0].value_counts()

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def decontracted(phrase):
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


corpus = []
for i in range(0, len(dataset1)):
    review =decontracted( dataset1['content'][i])
    review = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",review).split())
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = re.sub(r'([a-z])\1+', r'\1', review)
    review = review.split()
    ps = PorterStemmer()
    stwords = stopwords.words('english')
    stwords.remove('not')
    stwords.remove('no')
    review = [ps.stem(word) for word in review if not word in set(stwords)]
    review = ' '.join(review)
    corpus.append(review)



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()

y = dataset1['sentiment'].astype('category')

y1 = dataset1['sentiment'].astype('category')


dummies = pd.get_dummies(y)
y = dummies.values

features = cv.get_feature_names()

feat = pd.DataFrame(features)
#hello = np.random.random_integers(1,4005,4000)

import random
hello1 = random.sample(range(1, 4005), 4000)
feat['value'] = hello1


my_dict= {}
for i in range(len(feat)):
    x = { feat.loc[i,0]: feat.loc[i,'value']}
    my_dict = {**my_dict, **x}


print(my_dict['abt'])
corpus1 = []
for i in range(len(corpus)):
    words  = corpus[i].split()
    list1 = []
    for j in range(len(words)):
        try:
            list1.append(my_dict[words[j]])
        except:
            list1.append(0)
    corpus1.append(list1)

corpus1 = np.array(corpus1)



import numpy as np


np.save('my_file.npy', my_dict) 




#features = cv.get

    
import numpy
from keras.datasets import imdb #A utility to load a dataset
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding #To convert an integer to embedding
from keras.preprocessing import sequence #To convert a variable length sentence into a prespecified length
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest

top_words = 4006


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus1, y, test_size = 0.20, 
                                                    random_state = 17)

print (numpy.unique(y_train))
print (numpy.unique(y_test))

max(X_train.max())


print(X_train[:2])
print(y_train[:2])

# truncate and pad input sequences
max_review_length = 20
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
#model.add(Dense(13,kernel_initializer='uniform', activation='relu'))
model.add(Dense(13,kernel_initializer='uniform', activation='softmax'))
#model.add(Dense(13, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(X_train, y_train, nb_epoch=2, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1]*100))

model.save('Social_Emotion1.h5')




from keras.models import model_from_yaml

model_yaml = model.to_yaml()
with open("model1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")




import json
from keras.models import model_from_json , load_model




model.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
    
    
# Option 1: Load Weights + Architecture
with open('model_architecture.json', 'r') as f:
    new_model_1 = model_from_json(f.read())
new_model_1.load_weights('model_weights.h5')








# Load
read_dictionary = np.load('my_file.npy').item()
#print(read_dictionary['hello']) # displays "world"


def emotion(x):
    review =decontracted(x)
    review = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",review).split())
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = re.sub(r'([a-z])\1+', r'\1', review)
    #review  =" ".join(w for w in nltk.wordpunct_tokenize(review) \
     #   if w.lower() in words or not w.isalpha())
    review = review.split()
    ps = PorterStemmer()
    stwords = stopwords.words('english')
    stwords.remove('not')
    stwords.remove('no')
    review = [ps.stem(word) for word in review if not word in set(stwords)]
    review = ' '.join(review)
    words  = review.split()
    list1 = []
    k = []
    for j in range(len(words)):
        try:
            list1.append(my_dict[words[j]])
        except:
            list1.append(0)
    print(list1)
    list1 = np.array(list1)
    list2= np.reshape(list1,(1,len(list1)))
    list2 = sequence.pad_sequences(list2, maxlen=20)

    k= model.predict(list2)
    
    e1 = np.argmax(k)
    m = k
    m[0,e1] = 0
    e2 = np.argmax(m)
    
    emotion_names = dummies.columns
    
    for i in range(0,13):
        if e1==i:
        
          o1 =   "The Sentence seems to be" +" "+ emotion_names[i]
                      
    for i in range(0,13):
        if e2==i:
            o2 = "Otherwise The Sentence can be" +" "+ emotion_names[i] 
    z=[]
    z.append(o1)
    z.append(o2)
    print(z)
    return(z)


new_sentence = "love"
k1 = emotion(new_sentence)
k1

for j in range(len(words)):
        if words[j] in set(my_dict):
            list1.append(my_dict[words[j]])
        else:
            list1.append(0)
