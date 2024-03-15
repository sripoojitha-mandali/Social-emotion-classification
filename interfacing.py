# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:20:38 2018

@author: prash
"""

import json
from keras.models import model_from_json, load_model
from keras.preprocessing import sequence
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from keras.models import model_from_yaml


with open('model_architecture.json', 'r') as f:
    new_model_1 = model_from_json(f.read())
new_model_1.load_weights('model_weights.h5')


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

# Load
my_dict = np.load('my_file.npy').item()


def emotion():
    doc =text.get("1.0","end-1c")
    review =decontracted(doc)
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
    for j in range(len(words)):
        try:
            list1.append(my_dict[words[j]])
        except:
            list1.append(0)
    list1 = np.array(list1)
    list2= np.reshape(list1,(1,len(list1)))
    list2 = sequence.pad_sequences(list2, maxlen=20)

    z= new_model_1.predict(list2)
    print(z)
    return(z)
new_sentence = "how sad it, see her in this position"
#k = emotion(new_sentence)


import tkinter as t
from tkinter import *
root = t.Tk()
text= t.Text






root = Tk()
root.title("Text Editor")
text= Text()
text.pack()
button = Button(root,text= 'Save', command = emotion)
button.pack()
root.mainloop()



