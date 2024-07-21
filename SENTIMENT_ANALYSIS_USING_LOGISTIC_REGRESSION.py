import string
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
warnings.filterwarnings('ignore')

with open("C:\\Users\\LENOVO\\OneDrive\\Documents\\Data Science\\Projects & Assignmets\\Files\\data\\full_set.txt") as f:
    content = f.readlines()

#Remove leading and trailing white spaces
content = [x.strip() for x in content]
#Seperate the sentences from the labels:
sentences = [x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]

#print(sentences)
#print(labels)

# Transform the labels from '0 versus 1' to '-1 versus 1'
y = np.array(labels, dtype='int8')
y = 2*y - 1
print(y)
#"full_remove" takes a string x and a list of characters
# "removal_list" and returns with all the characters in removal_list
# replaced by ' '
def full_remove(x, removal_list):
    for w in removal_list:
        x = x.replace(w, ' ')
    return x
digits = [str(x) for x in range(10)] # remove digits
digit_less = [full_remove(x,digits) for x in sentences]
punc_less = [full_remove(x,list(string.punctuation)) for x in digit_less]
sents_lower = [x.lower() for x in punc_less]

stop_set = set(['the','a','an','i','he','she','they','to','of','it','from'])
# remove the stop words
sents_split = [x.split() for x in sents_lower]
sents_processed = [" ".join(list(filter(lambda a: a not in stop_set,x))) for x in sents_split]
#[print(w) for w in sents_processed]
sents_processed[0:10]

# Transform to bag of words representation.
vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None, stop_words=None,max_features=4500)
data_features = vectorizer.fit_transform(sents_processed)

# Append '1' to the end of each vector.
data_mat = data_features.toarray()

#Training / test split - We split the data into a training set of 2500 sentences
# and a test set of 500 sentences (of which 250 are positive and 250 negative).
# Split the data into testing and training sets
np.random.seed(0)
test_inds = np.append(np.random.choice((np.where(y==-1))[0],250,
                    replace=False),np.random.choice((np.where(y==1))[0],250,replace=False))
train_inds = list(set(range(len(labels))) - set(test_inds))
train_data = data_mat[train_inds,]
train_labels = y[train_inds]

test_data = data_mat[test_inds,]
test_labels = y[test_inds]

print('Train Data: ',train_data.shape)
print("Test data: ",test_data.shape)

# Fit logistic classifier on training data
clf = SGDClassifier(loss="log_loss")
clf.fit(train_data,train_labels)
# Pull out the parameters (w,b) of the logistic regression model
w = clf.coef_[0,:]
b = clf.intercept_
# getting predictions on training and test data
preds_train = clf.predict(train_data)
preds_test = clf.predict(test_data)
#compute errors
errs_train = np.sum((preds_train > 0.0) != (train_labels > 0.0))
errs_test = np.sum((preds_test > 0.0) != (test_labels > 0.0))

print("Training error: ",float(errs_train)/len(train_labels))
print("Test error: ",float(errs_test)/len(test_labels))
# Return number of test points for which Pr(y=1) lies in [0, 0.5 - gamma) or (0.5 + gamma, 1]
# Compute probability on each test point
# Find data points for which prediction is at least gamma away from 0.5
def margin_counts(clf, test_data, gamma):
    preds = clf.predict_proba(test_data)[:,1]
    margin_inds = np.where((preds > (0.5+gamma)) | (preds < (0.5-gamma)))[0]
    return float(len(margin_inds))
#let us visualize the test set's distribution of margin values.
gammas = np.arange(0,0.5,0.01)
f = np.vectorize(lambda g: margin_counts(clf, test_data, g))
plt.plot(gammas, f(gammas) / 500.0, linewidth=2, color='green')
plt.xlabel("Margin",fontsize=14)
plt.ylabel("Fraction of points above margin",fontsize=14)
plt.show()

#We investigate a natural question: "Are points "x" with larger margin more likely to
# be classified correctly? To address this, we define a function margin_errors
# that computes the fraction of points with margin at least "gamma" that are misclassified.
# Return error of predictions that lie in intervals [0, 0.5 - gamma) and (0.5 + gamma, 1]

def margin_errors(clf, test_data, test_labels, gamma):
    preds = clf.predict_proba(test_data)[:,1]
    margin_inds = np.where((preds > (0.5+gamma)) | (preds < (0.5-gamma)))[0]
    num_errors = np.sum((preds[margin_inds] > 0.5) != (test_labels[margin_inds] > 0.0))
    return float(num_errors)/len(margin_inds)
gammas = np.arange(0, 0.5,0.01)
f = np.vectorize(lambda g:margin_errors(clf, test_data, test_labels, g))
# Plot the result
plt.plot(gammas, f(gammas), linewidth=2)
plt.ylabel("Error rate",fontsize=14)
plt.xlabel('Margin',fontsize=14)
plt.show()

# Convert vocabulary into a list:
vocab = np.array([z[0] for z in sorted(vectorizer.vocabulary_.items(),key=lambda x:x[1])])
# Get indices of sorting w
inds = np.argsort(w)
# words with large negative values
neg_inds = inds[0:50]
print("Highly negative words: ")
print([str(x) for x in list(vocab[neg_inds])])

#word with large positve values
pos_inds = inds[-49:-1]
print("Highly positive words: ")
print([str(x) for x in list(vocab[pos_inds])])
