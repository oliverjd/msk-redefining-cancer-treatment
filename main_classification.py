TRAIN_N = 3321
TEST_N = 5668

from sklearn.metrics import log_loss

def write_results(probs, filename='pred/temp'):
	f = open(filename, 'w')
	f.write('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n')
	for i in range(len(probs)):
		f.write(str(i))
		for j in probs[i]:
			f.write(','+str(j))
		f.write('\n')
	f.close()

def evaluate(predicted, ground_truth):
	mcll = log_loss(ground_truth, predicted)
	return mcll

#%% This replicates next cell if data already saved

import pandas as pd
train = pd.read_pickle('data/pd_train_2_gz', compression='gzip')
test = pd.read_pickle('data/pd_test_2_gz', compression='gzip')

#%% Not necessary if previous cell runs

import re
import spacy
import pandas as pd
from sklearn.datasets import load_files
from nltk.corpus import stopwords

nlp = spacy.load('en')
nlp.pipeline = [nlp.tagger]

train_text = load_files('data/train', encoding='utf-8')
test_text = load_files('data/test', encoding='utf-8')

train = pd.read_csv('data/training_variants', header=0, quoting=3)
train_text_array = list()
with open('data/training_text', 'r') as file:
	next(file)
	for line in file:
		line = line.rstrip('\n')
		fields = line.split('||')
		train_text_array.append(fields[1])
se = pd.Series(train_text_array)
train['text'] = se.values

test = pd.read_csv('data/test_variants', header=0, quoting=3)
test_text_array = list()
with open('data/test_text', 'r') as file:
	next(file)
	for line in file:
		line = line.rstrip('\n')
		fields = line.split('||')
		test_text_array.append(fields[1])
se = pd.Series(test_text_array)
test['text'] = se.values

def preprocess(raw_text):
	no_punc = re.sub("[^a-zA-Z0-9]", " ", raw_text)
	words = no_punc.lower().split()
	stops = set(stopwords.words("english"))
	meaningful_words = " ".join([w for w in words if w not in stops])
	doc = nlp(meaningful_words)
	lemmatized_words = [w.lemma_ for w in doc]
	return(" ".join(lemmatized_words))

N_TRAIN = train["text"].size
clean_train_text = []
for i in range(0, N_TRAIN):
	if(i%10==0):
		print(i,'/',N_TRAIN)
	clean_train_text.append(preprocess(train["text"][i]))
se = pd.Series(clean_train_text)
train['clean_text'] = se.values

N_TEST = test["text"].size
clean_test_text = []
for i in range(0, N_TEST):
	if(i%10==0):
		print(i,'/',N_TEST)
	clean_test_text.append(preprocess(test["text"][i]))
se = pd.Series(clean_test_text)
test['clean_text'] = se.values

gene_var_list = [x.lower() for x in list(train['Gene'].unique()) +
				 list(train['Variation'].unique())]
for x in gene_var_list:
	train['count_'+x] = train['clean_text'].map(lambda y: y.count(x))

gene_var_list = [x.lower() for x in list(test['Gene'].unique()) +
				 list(test['Variation'].unique())]
for x in gene_var_list:
	test['count_'+x] = test['clean_text'].map(lambda y: y.count(x))

train['text_len'] = train['text'].map(lambda x: len(x))
test['text_len'] = test['text'].map(lambda x: len(x))

#%% Relative frequency baseline

import numpy as np

freq = np.zeros(9)
for i in range(TRAIN_N):
	freq[int(train[i]['class'])-1] += 1
for i in range(9):
	freq[i] /= TRAIN_N

pred = np.zeros((TRAIN_N, 9))
for i in range(TRAIN_N):
	pred[i] = freq
ll = evaluate(pred)

pred = np.zeros((TEST_N, 9))
for i in range(TEST_N):
	pred[i] = freq
write_results(pred, 'pred/baseline_frequency')

#%% Feature union

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

train['Class'] = train['Class'] - 1
train_no_label = train.drop(['Class'],axis=1)

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]

class CustomCounts(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(
				   ['Gene', 'Variation','ID','text','clean_text','combined_text'],
				   axis=1).values
        return x

class TextLen(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x['text_len'].values
        return x

transformer_list = [
		('text', Pipeline([
				('selector1', ItemSelector(key='clean_text')),
				('vectorizer1', CountVectorizer(analyzer="word",
					max_features=10000, ngram_range=(1, 2))),
				('tfidf1', TfidfTransformer()),
				('tsvd1', TruncatedSVD(n_components=50, n_iter=25, random_state=12))
				])),
		('gene', Pipeline([
				('selector2', ItemSelector(key='Gene')),
				('vectorizer1', CountVectorizer(analyzer='char', ngram_range=(1,8))),
				('tvsd2', TruncatedSVD(n_components=50, n_iter=25, random_state=12))
				])),
		('variation', Pipeline([
				('selector3', ItemSelector(key='Variation')),
				('vectorizer3', CountVectorizer(analyzer='char', ngram_range=(1,8))),
				('tvsd3', TruncatedSVD(n_components=50, n_iter=25, random_state=12))
				])),
		('counts', Pipeline([
				('custom_counts4', CustomCounts()),
				('tvsd3', TruncatedSVD(n_components=50, n_iter=25, random_state=12))
				]))
	]

combined_features = FeatureUnion(transformer_list)
train_union = combined_features.fit_transform(train_no_label)
test_union = combined_features.transform(test)

#%% Grid search - not used for best results

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

clf = SGDClassifier(loss='log', penalty='l2', alpha=5e-5, random_state=42, n_jobs=-1)
parameters = {'alpha': (1.3e-4, 1.25e-4, 1.2e-4, 1.15e-4, 1.1e-4, 1.0e-4),
			  'n_iter': (5, 6, 7, 8, 9), }

gs_clf = GridSearchCV(clf, parameters, cv=10, n_jobs=-1)
gs_clf = gs_clf.fit(train_union, train['Class'])
train_predicted = gs_clf.predict_proba(train_union)
test_predicted = gs_clf.predict_proba(test_union)
print(gs_clf.best_params_)

SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
	eta0=0.0, fit_intercept=True, l1_ratio=0.15,
	learning_rate='optimal', loss='log', n_iter=8, n_jobs=-1,
	penalty='l2', power_t=0.5, random_state=42, shuffle=True, verbose=0,
	warm_start=False)

clf = clf.fit(train_union, train['Class'])
train_predicted = clf.predict_proba(train_union)
test_predicted = clf.predict_proba(test_union)

#%% XGBoost

from sklearn.model_selection import train_test_split
import xgboost as xgb

param = {'objective': 'multi:softprob', 'eta': 0.03333, 'max_depth': 4, 'silent': 1,
		 'num_class': 9, 'eval_metric': 'mlogloss', 'nthread': 4}
num_rounds = 1000
folds = 5
for i in range(folds):
	print(i)
	param['seed'] = i
	train_f, valid_f, train_l, valid_l = train_test_split(
			train_union, train['Class'], test_size=0.18, random_state=i)
	watchlist = [(xgb.DMatrix(train_f, label=train_l), 'train'),
			  (xgb.DMatrix(valid_f, label=valid_l), 'valid')]
	xgtrain = xgb.DMatrix(train_f, label=train_l)
	clf = xgb.train(param, xgtrain, num_rounds, watchlist,
				 verbose_eval=50, early_stopping_rounds=60)
	test_preds = clf.predict(xgb.DMatrix(test_union))
	if i == 0:
		preds = test_preds.copy()
	else:
		preds += test_preds
preds /= folds

#%%

write_results(preds, 'pred/predictions.txt')
