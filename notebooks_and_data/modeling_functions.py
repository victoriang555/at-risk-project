# General system libraries
import os
import sys
from IPython.display import Image, Markdown
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Dataframe libraries
import pandas as pd
from pandas import DataFrame, read_csv

# Number manipulation
import scipy.sparse
from scipy.ndimage.filters import generic_filter
import patsy
import numpy as np

# Plotting libaries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
output_notebook()


# Data type libaries
from datetime import datetime as dt

# File manipulation
import pickle
import pandas.io.sql as pd_sql
from sqlalchemy import create_engine
import psycopg2 as pg
from flatten_json import flatten

# NLP libraries
import wikipedia as wiki
from nltk import word_tokenize, sent_tokenize,FreqDist, pos_tag
from nltk.corpus import stopwords
import gensim as gn
from gensim import corpora, models, similarities
from collections import defaultdict
from six import iteritems
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS
import string
import emoji
import enchant
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


# Scraping libraries
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from scraping_functions.tumblr_api import get_client
import pytumblr

# Stats libaries
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import datasets, linear_model, metrics
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import svm, datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier



# Other libaries
import geopy


def vectorize_both_ways(cleaned_df, text_to_vectorize):
    stop_words = list(STOP_WORDS)
    stop_words.append('a')
    cv = CountVectorizer(stop_words=stop_words)
    tfidf = TfidfVectorizer(stop_words=stop_words)
    corpus = cleaned_df[text_to_vectorize]
    cv_fitted = cv.fit(corpus)
    tfidf_fitted = tfidf.fit(corpus)
    cv_data = cv.fit_transform(corpus)
    tfidf_data = tfidf.fit_transform(corpus)
    return cv_fitted, cv_data, tfidf_fitted, tfidf_data


def gen_vectorizer_model_combos(cv_fitted, cv_data, tfidf_fitted, tfidf_data, n_topics=5, random_state=30):
    n_topics=n_topics
    random_state=random_state
    
    
    nmf_cv = NMF(n_components=n_topics, random_state=random_state)
    nmf_cv_data = nmf_cv.fit_transform(cv_data)
    
    nmf_tfidf = NMF(n_components=n_topics, random_state=random_state)
    nmf_tfidf_data = nmf_tfidf.fit_transform(tfidf_data)
    
    lsa_cv = TruncatedSVD(n_components=n_topics, random_state=random_state)
    lsa_cv_data = lsa_cv.fit_transform(cv_data)
    
    lsa_tfidf = TruncatedSVD(n_components=n_topics, random_state=random_state)
    lsa_tfidf_data = lsa_tfidf.fit_transform(tfidf_data)
    
    lda_cv = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda_cv_data = lda_cv.fit_transform(cv_data)
    
    lda_tfidf = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda_tfidf_data = lda_tfidf.fit_transform(tfidf_data)
    
    combo_models_list = [nmf_cv, nmf_tfidf, lsa_cv, lsa_tfidf, lda_cv, lda_tfidf]
    
    return nmf_cv, nmf_cv_data, nmf_tfidf, nmf_tfidf_data, lsa_cv, lsa_cv_data, lsa_tfidf, lsa_tfidf_data, lda_cv, lda_cv_data, lda_tfidf, lda_tfidf_data, combo_models_list
    
    

def gen_topics_for_one_combo(combo_model, combo_model_name, fitted_vectorizer, num_top_words):
    feature_names = fitted_vectorizer.get_feature_names()
    combo_topics = []
    for idx, topic in enumerate(combo_model.components_):
        topic_words = " ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]])
        combo_topics.append("{}_topic{}: {}".format(combo_model_name, idx+1, topic_words))
    return combo_topics


def compile_topics_df(combo_models_cv, combo_models_tfidf, cv_fitted, tfidf_fitted, num_top_words):
    all_combos_topics = []
    combo_names_cv = ['nmf_cv', 'lsa_cv', 'lda_cv']
    combo_names_tfidf =  ['nmf_tfidf', 'lsa_tfidf', 'lda_tfidf']

    for idx, combo_model in enumerate(combo_models_cv):
        combo_topics = gen_topics_for_one_combo(combo_model, combo_names_cv[idx], cv_fitted, num_top_words)
        all_combos_topics.append(combo_topics)
    for idx, combo_model in enumerate(combo_models_tfidf):
        combo_topics = gen_topics_for_one_combo(combo_model, combo_names_tfidf[idx], tfidf_fitted, num_top_words)
        all_combos_topics.append(combo_topics)

    combo_topics_df = pd.DataFrame(all_combos_topics)
    return combo_topics_df



def plot_docs_svd(lsa_cv_data, lsa_tfidf_data, cleaned_df, text_plotting):
    data = cleaned_df[text_plotting]
    svd_cv = TruncatedSVD(n_components=2)
    documents_2d_cv = svd_cv.fit_transform(lsa_cv_data)
    df_cv = pd.DataFrame(columns=['x', 'y', 'document'])
    df_cv['x'], df_cv['y'], df_cv['document'] = documents_2d_cv[:,0], documents_2d_cv[:,1], range(len(data))
 
    source_cv = ColumnDataSource(ColumnDataSource.from_df(df_cv))

    plot = figure(plot_width=600, plot_height=600, title='SVD Count Vectorized Doc Distribution')
    plot.circle("x", "y", size=12, source=source_cv, line_color="black", fill_alpha=0.8)

    show(plot, notebook_handle=True)
    
    svd_tfidf = TruncatedSVD(n_components=2)
    documents_2d_tfidf = svd.fit_transform(lsa_tfidf_data)
    df_tfidf = pd.DataFrame(columns=['x', 'y', 'document'])
    df_tfidf['x'], df_tfidf['y'], df_tfidf['document'] = documents_2d_tfidf[:,0], documents_2d_tfidf[:,1], range(len(data))
 
    source_tfidf = ColumnDataSource(ColumnDataSource.from_df(df_tfidf))

 
    plot = figure(plot_width=600, plot_height=600, title='SVD TFIDF Vectorized Doc Distribution')
    plot.circle("x", "y", size=12, source=source_tfidf, line_color="black", fill_alpha=0.8)

    show(plot, notebook_handle=True)

    

    
def gen_df_per_combo(nmf_cv_data, nmf_tfidf_data, lsa_cv_data, lsa_tfidf_data, lda_cv_data, 
                     lda_tfidf_data, num_topics):
    
    # NMF_CV
    nmf_cv_columns = []
    for num in range(1, num_topics+1):
        nmf_cv_columns.append('nmf_cv_topic{}'.format(num))
    nmf_cv_df = pd.DataFrame(nmf_cv_data, columns=nmf_cv_columns)
    nmf_cv_df['nmf_cv_sum'] = nmf_cv_df.sum(axis=1)
    
    # NMF_TFIDF
    nmf_tfidf_columns = []
    for num in range(1, num_topics+1):
        nmf_tfidf_columns.append('nmf_tfidf_topic{}'.format(num))
    nmf_tfidf_df = pd.DataFrame(nmf_tfidf_data, columns=nmf_tfidf_columns)
    nmf_tfidf_df['nmf_tfidf_sum'] = nmf_tfidf_df.sum(axis=1)
    
    # LSA_CV
    lsa_cv_columns = []
    for num in range(1, num_topics+1):
        lsa_cv_columns.append('lsa_cv_topic{}'.format(num))    
    lsa_cv_df = pd.DataFrame(lsa_cv_data, columns=lsa_cv_columns)
    lsa_cv_df['lsa_cv_sum'] = lsa_cv_df.sum(axis=1)
    
    # LSA_TFIDF
    lsa_tfidf_columns = []
    for num in range(1, num_topics+1):
        lsa_tfidf_columns.append('lsa_tfidf_topic{}'.format(num))
    lsa_tfidf_df = pd.DataFrame(lsa_tfidf_data, columns=lsa_tfidf_columns)
    lsa_tfidf_df['lsa_tfidf_sum'] = lsa_tfidf_df.sum(axis=1)
    
    # LDA_CV
    lda_cv_columns = []
    for num in range(1, num_topics+1):
        lda_cv_columns.append('lda_cv_topic{}'.format(num))
    lda_cv_df = pd.DataFrame(lda_cv_data, columns=lda_cv_columns)
    lda_cv_df['lda_cv_sum'] = lda_cv_df.sum(axis=1)
    
    # LDA_TFIDF
    lda_tfidf_columns = []
    for num in range(1, num_topics+1):
        lda_tfidf_columns.append('lda_tfidf_topic{}'.format(num))
    lda_tfidf_df = pd.DataFrame(lda_tfidf_data, columns=lda_tfidf_columns)
    lda_tfidf_df['lda_tfidf_sum'] = lda_tfidf_df.sum(axis=1)
    
    
    return nmf_cv_df, nmf_tfidf_df, lsa_cv_df, lsa_tfidf_df, lda_cv_df, lda_tfidf_df



def map_topic_names(compiled_combo_df, max_topic_type, topics_dict):
    all_topic_names = []
    for topic in compiled_combo_df[max_topic_type]:
        topic_name = topics_dict[topic]
        all_topic_names.append(topic_name)
    return all_topic_names



def compile_combo_dfs(cleaned_df, text_used, topics_dict, nmf_cv_df, nmf_tfidf_df, lsa_cv_df, lsa_tfidf_df, lda_cv_df, lda_tfidf_df):
    non_lda_max_model_df = pd.concat([nmf_cv_df['nmf_cv_sum'],nmf_tfidf_df['nmf_tfidf_sum'], lsa_cv_df['lsa_cv_sum'], lsa_tfidf_df['lsa_tfidf_sum']], axis=1)
    non_lda_max_model_df['non_lda_max_value'] = non_lda_max_model_df.max(axis=1)
    non_lda_max_model_df['non_lda_max_model'] = non_lda_max_model_df.idxmax(axis=1)
    
    non_lda_max_topic_df = pd.concat([nmf_cv_df, nmf_tfidf_df, lsa_cv_df, lsa_tfidf_df], axis=1)
    non_lda_max_topic_df.drop(columns=['nmf_cv_sum', 'nmf_tfidf_sum', 'lsa_cv_sum', 'lsa_tfidf_sum'], inplace=True)
    non_lda_max_topic_df['non_lda_max_topic']  = non_lda_max_topic_df.idxmax(axis=1)
        
    lda_max_topic_df = pd.concat([lda_cv_df, lda_tfidf_df], axis=1)
    lda_max_topic_df.drop(columns=['lda_cv_sum','lda_tfidf_sum'], axis=1, inplace=True)
    lda_max_topic_df['lda_max_topic'] = lda_max_topic_df.idxmax(axis=1)
    
    final_df = pd.DataFrame()
    final_df[text_used] = cleaned_df[text_used]
    final_df['username'] = cleaned_df['username']
    final_df['non_lda_max_topic'] = non_lda_max_topic_df['non_lda_max_topic']
    final_df['lda_max_topic'] = lda_max_topic_df['lda_max_topic']
    final_df['non_lda_topic_name'] = map_topic_names(final_df, 'non_lda_max_topic', topics_dict)
    final_df['lda_topic_name'] = map_topic_names(final_df, 'lda_max_topic', topics_dict)
        
    final_df['non_lda_max_model'] = non_lda_max_model_df['non_lda_max_model']
    final_df['non_lda_max_value'] = non_lda_max_model_df['non_lda_max_value']
    
    return final_df


def random_sample(final_df, criterion1=None, value1=None, criterion2=None, value2=None, use_one_criterion=False, use_two_criteria=False, sample_size=.3, random_state=30):
    sample_size = sample_size
    random_state = random_state
    
    if use_two_criteria == True:
        new_df = final_df[(final_df[criterion1] == value1) & (final_df[criterion2] == value2)]
    elif use_one_criterion == True:
        new_df = final_df[final_df[criterion1] == value1]
    else:
        new_df = final_df.copy()
    
    return new_df.sample(frac=sample_size)


def get_precision_score(sample_df, fp_list):
    FP = len(fp_list)
    TP = len(sample_df) - FP
    
    precision_score = TP/(TP + FP)
    return precision_score


def remove_unrelated(final_df, criterion1, value1, criterion2=None, value2=None, use_two_criteria=False):
    
    if use_two_criteria == True:
        new_df = final_df[(final_df[criterion1] != value1) & (final_df[criterion2] != value2)]
    else:
        new_df = final_df[final_df[criterion1] != value1]
        new_df.reset_index(inplace=True)
        new_df.drop(column=['index'], inplace=True)
    return new_df



