import numpy as np
import pandas as pd
import stem_lyrics as stem
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pdb

def predictor(song_lyrics):
	'''
	input text string of lyrics and predict popularity (boolean)
	'''
	#create data 
	unpopular_df = pd.read_pickle('unpopular.pkl')
	new_lyric_df = pd.DataFrame(index = [0],columns=unpopular_df.columns)
	new_lyric_df.fillna(0, inplace=True)
	# call stemmer on lyrics to create dictionary
	song_stem = stem.lyrics_to_bow(song_lyrics)
	# populate data frame with dictionary
	for k, v in song_stem.iteritems():
   		if k in new_lyric_df.columns:
   			new_lyric_df.ix[0,k]= v
	new_lyric_df = new_lyric_df.fillna(0)
	# load model
	model = joblib.load('nb_model.pkl') 
	feature_vector = new_lyric_df.as_matrix()
	my_prediction = model.predict(feature_vector)
	return my_prediction


