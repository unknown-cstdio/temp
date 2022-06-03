import nltk
import json
import firebase_admin
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from scipy.optimize import minimize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from firebase_admin import db
import matplotlib.pyplot as plt


#Set up firebase
cred_obj = firebase_admin.credentials.Certificate("D:\Desktop\hackrice\hackriceprojecteating-firebase-adminsdk-xk4ej-dc1bd932cc.json")
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://hackriceprojecteating-default-rtdb.firebaseio.com"})
food_data_test_ref = db.reference("/foodItems")
food_data_test = food_data_test_ref.get()
food_data_test_df = pd.DataFrame.from_dict(food_data_test)
print(food_data_test_df)
ref = db.reference("/Books")
with open("random.json", "r") as f:
	file_contents = json.load(f)
ref.set(file_contents)
ref = db.reference("/Books")
ref.set({})

'''
#read data and split into train and test
data_set = pd.read_csv('D:/Desktop/hackrice/test_data/ratings_small.csv').head(5000)
data_train, data_test = train_test_split(data_set, stratify=data_set['userId'], test_size=0.2, random_state=42)
keyword_set = pd.read_csv("D:/Desktop/hackrice/test_data/keywords.csv")
data_set = data_set.set_index('movieId').join(keyword_set.set_index('movieId')).reset_index()
data_set = data_set[data_set['rating'].notna()]
data_set = data_set[data_set['keywords'].notna()]

#Indexing based on userId
data_set_indexed = data_set.set_index('userId')
data_train_indexed = data_train.set_index('userId')
data_test_indexed = data_test.set_index('userId')

#Get the user's data
def get_items_interacted(user_id, data_input):
    rated_items = data_input.loc[user_id]['movieId']
    return set(rated_items if type(rated_items) == pd.Series else [rated_items])


#Top-N accuracy
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:

    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, data_set_indexed)
        all_items = set(data_set['movieId'])
        non_interacted_items = all_items - interacted_items
        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        interacted_values_testset = data_test_indexed.loc[person_id]
        if type(interacted_values_testset['movieId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['movieId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['movieId'])])
        interacted_items_count_testset = len(person_interacted_items_testset) 
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id,data_train_indexed), topn=50000)
        hits_at_5_count = 0
        hits_at_10_count = 0
        for item_id in person_interacted_items_testset:
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, seed=item_id%(2**32))
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))
            valid_recs_df = person_recs_df[person_recs_df['movieId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['movieId'].values
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(data_test_indexed.index.unique().values)):
            if idx % 100 == 0 and idx > 0:
                print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}
        return global_metrics, detailed_results_df
model_evaluator = ModelEvaluator() 


#popularity model
item_popularity_set = data_set.groupby('movieId')['rating'].sum().sort_values(ascending=False).reset_index()


class PopularityRecommender:

    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
    
    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        recommendation_df = self.popularity_df[~self.popularity_df['movieId'].isin(items_to_ignore)].sort_values('rating', ascending = False).head(topn)
        recommendation_df['rating'] = (recommendation_df['rating']-recommendation_df['rating'].min())/(recommendation_df['rating'].max()-recommendation_df['rating'].min())
        return recommendation_df

popularity_model = PopularityRecommender(item_popularity_set)

#Content_based
stopwords_list = stopwords.words('english')
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = data_set['movieId'].tolist()
tfidf_matrix = vectorizer.fit_transform(data_set['keywords'])
tfidf_feature_name = vectorizer.get_feature_names()

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, data_set_indexed_loc):
    data_person_df = data_set_indexed_loc.loc[person_id]
    user_item_profiles = get_item_profiles(data_person_df['movieId'])
    user_item_strengths = np.array(data_person_df['rating']).reshape(-1,1)
    user_item_strenghs_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0)/np.sum(user_item_strengths)
    user_profile_norm=sklearn.preprocessing.normalize(user_item_strenghs_weighted_avg)
    return user_profile_norm

def build_users_profiles():
    data_set_indexed_loc = data_train[data_train['movieId'].isin(data_set['movieId'])].set_index('userId')
    user_profiles = {}
    for person_id in data_set_indexed_loc.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, data_set_indexed_loc)
    return user_profiles

user_profiles = build_users_profiles()
#print(pd.DataFrame(sorted(zip(tfidf_feature_name, user_profiles[1].flatten().tolist()),key=lambda x: -x[1])[:20],columns=['token','relevance']))

class ContentBasedRecommender:

    MODEL_NAME = 'Content-Based'

    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, topn=100):
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['movieId', 'rating']).head(topn)
        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(data_set)

#Collaborative
users_items_pivot_matrix_df = data_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
users_ids = list(users_items_pivot_matrix_df.index)
users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

NUMBER_OF_FACTORS_MF = 15
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
         sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'rating'})
         recommendations_df = sorted_user_predictions[~sorted_user_predictions['movieId'].isin(items_to_ignore)].sort_values('rating', ascending = False).head(topn)
         return recommendations_df


cf_recommender_model = CFRecommender(cf_preds_df, data_set)


class HybridRecommender:

    MODEL_NAME = 'Hybrid'

    def __init__(self, pop_rec_model, cb_rec_model, cf_rec_model, items_df, pop_weight=1.0, cb_weight=0.0, cf_weight=1.0):
        self.pop_rec_model = pop_rec_model
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.pop_weight = pop_weight
        self.cb_weight = cb_weight
        self.cd_weight = cf_weight
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        pop_recs_df = self.pop_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, topn=1000)
        pop_recs_df.columns = ['movieId','recStrengthPOP']
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, topn=1000)
        cb_recs_df.columns = ['movieId','recStrengthCB']
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, topn=1000)
        cf_recs_df.columns = ['movieId','recStrengthCF']
        recs_df = cb_recs_df.merge(cf_recs_df, how ='outer', left_on = 'movieId', right_on='movieId').fillna(0.0)
        recs_df = recs_df.merge(pop_recs_df, how ='outer', left_on = 'movieId', right_on='movieId').fillna(0.0)
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthPOP']*self.pop_weight) + (recs_df['recStrengthCB']*self.cb_weight) + (recs_df['recStrengthCF']*self.cd_weight)
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)
        return recommendations_df

hybrid_recommender_model = HybridRecommender(popularity_model, content_based_recommender_model, cf_recommender_model, data_set,pop_weight=0.5,cb_weight=0, cf_weight=100.0)
'''

#Optimize weights
'''
def func(x):
    hybrid_recommender_model_loc = HybridRecommender(popularity_model, content_based_recommender_model, cf_recommender_model, data_set,pop_weight=x[0],cb_weight=x[1], cf_weight=x[2])
    hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model_loc)
    evaluate = 1-hybrid_global_metrics['recall@10']
    print(x,evaluate, hybrid_global_metrics['recall@10'])
    return evaluate

max_eval = 0.0
current = [0.0, 0.0, 0.0]
for x1 in range(0, 10):
    for x2 in range(0, 10):
        for x3 in range(0, 10):
            now = func([x1/10,x2/10,x3/10])
            if now > max_eval:
                max_eval = now
                current = [x1/10,x2/10,x3/10]

print(current, max_eval)
'''

'''
print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)

print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
print(hybrid_detailed_results_df.head(10))
'''