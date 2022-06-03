import re
import nltk
import json
import firebase_admin
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
import calendar
from datetime import date
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

#Read menu
ref = db.reference('/Menu')
menu_js = ref.get()
d = {'date':[], 'isLunch':[], 'servery':[], 'foodId':[]}
menu_data = pd.DataFrame(d)
for key1 in menu_js.keys():
    for key2 in menu_js[key1].keys():
        for key3 in menu_js[key1][key2].keys():
            for item in menu_js[key1][key2][key3]:
                d2 = {'date':key1, 'isLunch':key2, 'servery':key3, 'foodId':item}
                menu_data = menu_data.append(d2, ignore_index=True)


#Read food data
ref = db.reference('/foodItems')
food_items_js = ref.get()
food_items_js.pop(0)
d = {'foodId':[], 'foodName':[], 'ingredient':[], 'flavor':[], 'allergies':[], 'Soup':[]}
food_data = pd.DataFrame(d)
for idx in range(len(food_items_js)):
    food_idx = idx
    food_list = list(food_items_js[idx].keys())
    nested = food_items_js[idx][food_list[0]]
    ingre = []
    if 'Ingredient' in nested.keys():
        ingre = nested['Ingredient']
    fla = []
    if 'Flavor' in nested.keys():
        fla = nested['Flavor']
    alle = []
    if 'Allergies' in nested.keys():
        alle = nested['Allergies']
    ifSoup = nested['Soup']
    d2 = {'foodId':food_idx + 1, 'foodName':food_list, 'ingredient':ingre, 'flavor':fla, 'allergies':alle, 'Soup':ifSoup}
    food_data = food_data.append(d2, ignore_index=True)

food_data['keywords'] = food_data['ingredient'] + food_data['flavor']
food_data['Soup'] = np.where(food_data['Soup'] == 1.0, 'Soup', '')
food_data['keywords'] = [', '.join(map(str, l)) for l in food_data['keywords']]

#default rating
d = {'userId':[], 'foodId':[], 'rating':[]}
user_rating = pd.DataFrame(d)
for id in food_data['foodId']:
    d2 = {'userId':-1, 'foodId':id, 'rating':2.5}
    user_rating = user_rating.append(d2, ignore_index=True)
for id in food_data['foodId']:
    d2 = {'userId':-2, 'foodId':id, 'rating':2.5}
    user_rating = user_rating.append(d2, ignore_index=True)
for id in food_data['foodId']:
    d2 = {'userId':-3, 'foodId':id, 'rating':2.5}
    user_rating = user_rating.append(d2, ignore_index=True)


#User information and date
current_userId = 0
user_prefer_like = ['Corn', 'Tofu', 'Carrot', 'Chicken']
user_prefer_dislike = ['Seafood', 'Mushroom']
user_alle = ['Peanut']
my_date = date.today()
current_date = calendar.day_name[my_date.weekday()]

#Initialize user prefer
for idx in food_data.index:
    temp_list = food_data['keywords'][idx].split(", ")
    rate = 3
    for item in user_prefer_like:
        if item in temp_list and rate < 5:
            rate +=1
    for item in user_prefer_dislike:
        if item in temp_list and rate > 0:
            rate -=1
    if not rate == 3:
        d2 = {'userId':current_userId, 'foodId':food_data['foodId'][idx], 'rating': rate}
        user_rating = user_rating.append(d2, ignore_index=True)

pop_on = False
if current_userId not in user_rating['userId'].unique():
    pop_on = True

#Initialize ignore
ignore_list = []
for idx in food_data.index:
    #Allergies
    if (set(food_data['allergies'][idx]) & set(user_alle)):
        print('check')
        ignore_list.append(food_data['foodId'][idx])
#date
ignore_list1 = ignore_list.copy()
ignore_list2 = ignore_list.copy()
#lunch
allowed = []
for idx in menu_data.index:
    if (menu_data['date'][idx] == current_date) and (menu_data['isLunch'][idx] == 'lunch'):
        allowed.append(menu_data['foodId'][idx])
disallowed = set(food_data['foodId']) - set(allowed)
ignore_list1.extend(list(disallowed))
#dinner
allowed = []
for idx in menu_data.index:
    if (menu_data['date'][idx] == current_date) and (menu_data['isLunch'][idx] == 'dinner'):
        allowed.append(menu_data['foodId'][idx])
disallowed = set(food_data['foodId']) - set(allowed)
ignore_list2.extend(list(disallowed))


#Merge a data set
data_set = pd.merge(user_rating, food_data, on='foodId')


#popularity model
item_popularity_set = data_set.groupby('foodId')['rating'].sum().sort_values(ascending=False).reset_index()

def recommand_result(rec_data, current_time, key):
    food_info = food_data.loc[food_data['foodId'].isin(rec_data['foodId'])]
    candidates = menu_js[current_date][current_time]
    total = len(candidates)
    d = {'Servery': list(candidates.keys()), "score": [0]*total}
    recommand_serv_df = pd.DataFrame(d)
    for idx1 in rec_data.index:
        for idx2 in recommand_serv_df.index:
            serv = recommand_serv_df['Servery'][idx2]
            if rec_data['foodId'][idx1] in candidates[serv]:
                recommand_serv_df.at[idx2, 'score'] += rec_data[key][idx1]

    recommand_serv_df = recommand_serv_df.sort_values(by='score',ascending=False).reset_index()
    recommand_serv_df['foodIncluded'] = [None]*len(candidates)

    for idx1 in recommand_serv_df.index:
        rec_food_list = []
        for idx2 in food_info.index:
            if food_info['foodId'][idx2] in candidates[recommand_serv_df['Servery'][idx1]]:
                rec_food_list = rec_food_list + list(food_info["foodName"][idx2])
                recommand_serv_df.at[idx1, 'foodIncluded'] = rec_food_list
    result_df = recommand_serv_df.copy()
    result_df = result_df.drop(['index', 'score'],axis=1)
    result = result_df.to_json(orient="index")
    json_obj = json.loads(result)
    ref_name = '/' + current_time
    ref = db.reference(ref_name)
    ref.set(json_obj)
    #print(current_time+':')
    #print(recommand_serv_df.head(3))

class PopularityRecommender:

    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
    
    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        recommendation_df = self.popularity_df[~self.popularity_df['foodId'].isin(items_to_ignore)].sort_values('rating', ascending = False).head(topn)
        recommendation_df['rating'] = (recommendation_df['rating']-recommendation_df['rating'].min())/(recommendation_df['rating'].max()-recommendation_df['rating'].min()+0.0001)
        return recommendation_df

popularity_model = PopularityRecommender(item_popularity_set)


if pop_on:
    recommand_data1 = popularity_model.recommend_items(current_userId, ignore_list1, 10).drop_duplicates(subset=['foodId'])
    recommand_result(recommand_data1, 'lunch', 'rating')
    recommand_data2 = popularity_model.recommend_items(current_userId, ignore_list2, 10).drop_duplicates(subset=['foodId'])
    recommand_result(recommand_data2, 'dinner', 'rating')

else:
    #Content_based: Usign TF-IDF and cosine similarity

    #Building tfidf matrix
    stopwords_list = stopwords.words('english')
    vectorizer = TfidfVectorizer(analyzer='word',
                        ngram_range=(1, 2),
                        min_df=0.003,
                        max_df=0.5,
                        max_features=5000,
                        stop_words=stopwords_list)

    item_ids = data_set['foodId'].tolist()
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

    #data about single person
    def build_users_profile(person_id, data_set_indexed_loc):
        data_person_df = data_set_indexed_loc.loc[person_id]
        user_item_profiles = get_item_profiles(data_person_df['foodId'])
        user_item_strengths = np.array(data_person_df['rating']).reshape(-1,1)
        user_item_strenghs_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0)/np.sum(user_item_strengths)
        user_profile_norm=sklearn.preprocessing.normalize(user_item_strenghs_weighted_avg)
        return user_profile_norm

    #the collection of data
    def build_users_profiles():
        data_set_indexed_loc = data_set.set_index('userId')
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
            recommendations_df = pd.DataFrame(similar_items_filtered, columns=['foodId', 'rating']).head(topn)
            return recommendations_df
        
    content_based_recommender_model = ContentBasedRecommender(data_set)

    #Collaborative: Using SVD method

    #Matrix Factorization
    users_items_pivot_matrix_df = data_set.pivot(index='userId', columns='foodId', values='rating').fillna(0)
    users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
    users_ids = list(users_items_pivot_matrix_df.index)
    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

    NUMBER_OF_FACTORS_MF = 2
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
            recommendations_df = sorted_user_predictions[~sorted_user_predictions['foodId'].isin(items_to_ignore)].sort_values('rating', ascending = False).head(topn)
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
            pop_recs_df.columns = ['foodId','recStrengthPOP']
            cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, topn=1000)
            cb_recs_df.columns = ['foodId','recStrengthCB']
            cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, topn=1000)
            cf_recs_df.columns = ['foodId','recStrengthCF']
            recs_df = cb_recs_df.merge(cf_recs_df, how ='outer', left_on = 'foodId', right_on='foodId').fillna(0.0)
            recs_df = recs_df.merge(pop_recs_df, how ='outer', left_on = 'foodId', right_on='foodId').fillna(0.0)
            recs_df['recStrengthHybrid'] = (recs_df['recStrengthPOP']*self.pop_weight) + (recs_df['recStrengthCB']*self.cb_weight) + (recs_df['recStrengthCF']*self.cd_weight)
            recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).drop_duplicates(subset=['foodId']).head(topn)
            return recommendations_df


    #Create Model
    hybrid_recommender_model = HybridRecommender(popularity_model, content_based_recommender_model, cf_recommender_model, data_set,pop_weight=1,cb_weight=1, cf_weight=1)

    #recommand food
    recommand_data1 = hybrid_recommender_model.recommend_items(current_userId, ignore_list1, 5)
    recommand_result(recommand_data1, 'lunch', 'recStrengthHybrid')
    recommand_data2 = hybrid_recommender_model.recommend_items(current_userId, ignore_list2, 5)
    recommand_result(recommand_data2, 'dinner', 'recStrengthHybrid')