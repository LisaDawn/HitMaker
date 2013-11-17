import numpy as np
import pandas as pd
import sqlite3
from pandas.io import sql
import sys
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import random

#Connect to Billboard, Million Song and MusixMatch tables
def create_data():
	'''
	Creates dataframes from database
	Adds labels
	Breaks them into training and testing
	'''
	# get data from database
	conn_all = sqlite3.connect('datasets/all_the_data.db')
	popular_df = pd.io.sql.read_frame('''select * from pop_lyric''', 
		conn_all, index_col=None, coerce_float=True, params=None)
	unpopular_df = pd.io.sql.read_frame('''select * from unpop_lyric''', 
		conn_all, index_col=None, coerce_float=True, params=None)
	conn_all.close()

	# pivot and replace NaNs
	popular_df = popular_df.pivot(index='track_id', columns='word', values='count')
	unpopular_df = unpopular_df.pivot(index='track_id', columns='word', values='count')
	popular_df=popular_df.fillna(0)
	unpopular_df=unpopular_df.fillna(0)

	# make labels
	popular_df['!popular']=np.ones(5242)
	unpopular_df['`!popular']=np.zeros(5242)

	# merge the two datasets together
	merged_df = pd.concat([popular_df, unpopular_df])
	merged_df=merged_df.fillna(0)

	#create features and labels
	new_feature=merged_df.ix[:,2:5001].values
	label = merged_df.ix[:,0:1].values
	flat_label=label.flatten()

	# create dataframes from label/ceature numpy arrays
	label_df=pd.DataFrame(flat_label, index=merged_df.index, columns=[merged_df.columns[0]])
	feature_df=pd.DataFrame(new_feature, index=merged_df.index, columns=[merged_df.columns[1:5000]])

	# split into training and testing
	np.random.seed(2013)
	df_source = label_df
	df_source2 = feature_df
	#df_source2 = tfidf_df
	rows = np.random.binomial(1, .70, size=len(df_source)).astype('bool')
	label_train = df_source[rows]
	label_test = df_source[~rows]
	feat_train = df_source2[rows]
	feat_test = df_source2[~rows]

	# create labels for training/ testing
	label_train_col=label_train.iloc[:,0:1]
	label_test_col=label_test.iloc[:,0:1]
	
	# initialize transformer
	transformer = TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

	tfidf_sparse = transformer.fit_transform(feature_df)
	tfidf_ary=tfidf_sparse.toarray()  
	tfidf_df=pd.DataFrame(tfidf_ary, index=merged_df.index, columns=[merged_df.columns[1:5000]])

	# pickle it
	feat_train.to_pickle('feat_train.pkl')
	return feat_train, label_train_col, feat_test, label_test_col


def create_model(feat_train, label_train_col, feat_test, label_test_col):
	'''
	create model
	'''
	clf_nb = MultinomialNB()
	clf_nb.fit(feat_train, label_train_col)
	#clf.fit(lsa_features, label_train)
	cpt = clf_nb.predict_proba(feat_test)
	clf_nb.score(feat_test, label_test_col)
	nb_cpt=clf_nb.feature_log_prob_

# MAKE PICKLE FOR EXPORT TO
# from sklearn.externals import joblib
# joblib.dump(clf, 'nb_model.pkl')

