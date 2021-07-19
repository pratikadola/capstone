# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:56.971885Z","iopub.execute_input":"2021-07-17T06:03:56.972433Z","iopub.status.idle":"2021-07-17T06:03:58.635237Z","shell.execute_reply.started":"2021-07-17T06:03:56.972339Z","shell.execute_reply":"2021-07-17T06:03:58.634133Z"}}
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:58.637157Z","iopub.execute_input":"2021-07-17T06:03:58.637583Z","iopub.status.idle":"2021-07-17T06:03:59.247882Z","shell.execute_reply.started":"2021-07-17T06:03:58.637530Z","shell.execute_reply":"2021-07-17T06:03:59.246673Z"}}
df = pd.read_csv('input/sample30.csv')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.250325Z","iopub.execute_input":"2021-07-17T06:03:59.250806Z","iopub.status.idle":"2021-07-17T06:03:59.319280Z","shell.execute_reply.started":"2021-07-17T06:03:59.250759Z","shell.execute_reply":"2021-07-17T06:03:59.318419Z"}}
# print(round(100*(df.isnull().sum()/len(df.index)), 2))

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.320681Z","iopub.execute_input":"2021-07-17T06:03:59.321118Z","iopub.status.idle":"2021-07-17T06:03:59.340790Z","shell.execute_reply.started":"2021-07-17T06:03:59.321075Z","shell.execute_reply":"2021-07-17T06:03:59.339633Z"}}
df = df.drop(['reviews_userProvince','reviews_userCity'], axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.342878Z","iopub.execute_input":"2021-07-17T06:03:59.343446Z","iopub.status.idle":"2021-07-17T06:03:59.368589Z","shell.execute_reply.started":"2021-07-17T06:03:59.343315Z","shell.execute_reply":"2021-07-17T06:03:59.367082Z"}}
df.reviews_doRecommend = df.reviews_doRecommend.fillna(value=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.370357Z","iopub.execute_input":"2021-07-17T06:03:59.370852Z","iopub.status.idle":"2021-07-17T06:03:59.382749Z","shell.execute_reply.started":"2021-07-17T06:03:59.370788Z","shell.execute_reply":"2021-07-17T06:03:59.381542Z"}}
df.reviews_title = df.reviews_title.fillna('')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.384399Z","iopub.execute_input":"2021-07-17T06:03:59.385063Z","iopub.status.idle":"2021-07-17T06:03:59.404208Z","shell.execute_reply.started":"2021-07-17T06:03:59.385018Z","shell.execute_reply":"2021-07-17T06:03:59.403022Z"}}
df = df.drop(['manufacturer','reviews_date'],axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.406925Z","iopub.execute_input":"2021-07-17T06:03:59.407449Z","iopub.status.idle":"2021-07-17T06:03:59.429242Z","shell.execute_reply.started":"2021-07-17T06:03:59.407400Z","shell.execute_reply":"2021-07-17T06:03:59.428051Z"}}
df = df[~(df.user_sentiment.isnull())]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.431087Z","iopub.execute_input":"2021-07-17T06:03:59.431389Z","iopub.status.idle":"2021-07-17T06:03:59.461889Z","shell.execute_reply.started":"2021-07-17T06:03:59.431362Z","shell.execute_reply":"2021-07-17T06:03:59.460777Z"}}
df.user_sentiment = df.user_sentiment.apply(lambda x: 1 if x=='Positive' else 0)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.463487Z","iopub.execute_input":"2021-07-17T06:03:59.463941Z","iopub.status.idle":"2021-07-17T06:03:59.486337Z","shell.execute_reply.started":"2021-07-17T06:03:59.463897Z","shell.execute_reply":"2021-07-17T06:03:59.485163Z"}}
df['review_length'] = df['reviews_text'].apply(len)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.487771Z","iopub.execute_input":"2021-07-17T06:03:59.488243Z","iopub.status.idle":"2021-07-17T06:03:59.500512Z","shell.execute_reply.started":"2021-07-17T06:03:59.488209Z","shell.execute_reply":"2021-07-17T06:03:59.499639Z"}}
df = df.reset_index(drop=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.501788Z","iopub.execute_input":"2021-07-17T06:03:59.502285Z","iopub.status.idle":"2021-07-17T06:03:59.523037Z","shell.execute_reply.started":"2021-07-17T06:03:59.502245Z","shell.execute_reply":"2021-07-17T06:03:59.522003Z"}}
# Number of unique users
len(df.reviews_username.unique())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.523937Z","iopub.execute_input":"2021-07-17T06:03:59.524208Z","iopub.status.idle":"2021-07-17T06:03:59.536035Z","shell.execute_reply.started":"2021-07-17T06:03:59.524182Z","shell.execute_reply":"2021-07-17T06:03:59.534750Z"}}
# Number of unique items
len(df.id.unique())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.537841Z","iopub.execute_input":"2021-07-17T06:03:59.538343Z","iopub.status.idle":"2021-07-17T06:03:59.543070Z","shell.execute_reply.started":"2021-07-17T06:03:59.538293Z","shell.execute_reply":"2021-07-17T06:03:59.542344Z"}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.544631Z","iopub.execute_input":"2021-07-17T06:03:59.545035Z","iopub.status.idle":"2021-07-17T06:03:59.569404Z","shell.execute_reply.started":"2021-07-17T06:03:59.545005Z","shell.execute_reply":"2021-07-17T06:03:59.567578Z"}}
# Test and Train split of the dataset.

train, test = train_test_split(df, test_size=0.30, random_state=31)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.571384Z","iopub.execute_input":"2021-07-17T06:03:59.572141Z","iopub.status.idle":"2021-07-17T06:03:59.596836Z","shell.execute_reply.started":"2021-07-17T06:03:59.572079Z","shell.execute_reply":"2021-07-17T06:03:59.595752Z"}}
train.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.597891Z","iopub.execute_input":"2021-07-17T06:03:59.598149Z","iopub.status.idle":"2021-07-17T06:03:59.786486Z","shell.execute_reply.started":"2021-07-17T06:03:59.598123Z","shell.execute_reply":"2021-07-17T06:03:59.785490Z"}}
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating',
    aggfunc=np.min,
).T

df_pivot.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.788002Z","iopub.execute_input":"2021-07-17T06:03:59.788328Z","iopub.status.idle":"2021-07-17T06:03:59.833943Z","shell.execute_reply.started":"2021-07-17T06:03:59.788298Z","shell.execute_reply":"2021-07-17T06:03:59.832859Z"}}
mean = np.nanmean(df_pivot, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.835053Z","iopub.execute_input":"2021-07-17T06:03:59.835340Z","iopub.status.idle":"2021-07-17T06:03:59.842290Z","shell.execute_reply.started":"2021-07-17T06:03:59.835312Z","shell.execute_reply":"2021-07-17T06:03:59.841300Z"}}
mean[:5]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.843618Z","iopub.execute_input":"2021-07-17T06:03:59.843948Z","iopub.status.idle":"2021-07-17T06:03:59.867834Z","shell.execute_reply.started":"2021-07-17T06:03:59.843918Z","shell.execute_reply":"2021-07-17T06:03:59.866440Z"}}
df_subtracted = (df_pivot.T-mean).T

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.871221Z","iopub.execute_input":"2021-07-17T06:03:59.871734Z","iopub.status.idle":"2021-07-17T06:03:59.903382Z","shell.execute_reply.started":"2021-07-17T06:03:59.871672Z","shell.execute_reply":"2021-07-17T06:03:59.902243Z"}}
df_subtracted.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.904662Z","iopub.execute_input":"2021-07-17T06:03:59.904950Z","iopub.status.idle":"2021-07-17T06:03:59.909785Z","shell.execute_reply.started":"2021-07-17T06:03:59.904921Z","shell.execute_reply":"2021-07-17T06:03:59.908084Z"}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.916293Z","iopub.execute_input":"2021-07-17T06:03:59.916650Z","iopub.status.idle":"2021-07-17T06:03:59.922263Z","shell.execute_reply.started":"2021-07-17T06:03:59.916618Z","shell.execute_reply":"2021-07-17T06:03:59.921192Z"}}
# X = np.array([[2, 3], [3, 5], [5, 8]])

# pairwise_distances(X,  metric='cosine')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.925357Z","iopub.execute_input":"2021-07-17T06:03:59.926063Z","iopub.status.idle":"2021-07-17T06:03:59.933830Z","shell.execute_reply.started":"2021-07-17T06:03:59.926008Z","shell.execute_reply":"2021-07-17T06:03:59.932718Z"}}
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_columns', 10)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# df_subtracted.head().sum()
# (pairwise_distances(df_subtracted.fillna(0), metric='cosine'))[2]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:03:59.935286Z","iopub.execute_input":"2021-07-17T06:03:59.935671Z","iopub.status.idle":"2021-07-17T06:04:00.103718Z","shell.execute_reply.started":"2021-07-17T06:03:59.935634Z","shell.execute_reply":"2021-07-17T06:04:00.101129Z"}}
# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.105655Z","iopub.execute_input":"2021-07-17T06:04:00.106148Z","iopub.status.idle":"2021-07-17T06:04:00.118819Z","shell.execute_reply.started":"2021-07-17T06:04:00.106102Z","shell.execute_reply":"2021-07-17T06:04:00.117440Z"}}
item_correlation[item_correlation<0]=0
item_correlation

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.120787Z","iopub.execute_input":"2021-07-17T06:04:00.121287Z","iopub.status.idle":"2021-07-17T06:04:00.240950Z","shell.execute_reply.started":"2021-07-17T06:04:00.121240Z","shell.execute_reply":"2021-07-17T06:04:00.239778Z"}}
item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.246262Z","iopub.execute_input":"2021-07-17T06:04:00.249278Z","iopub.status.idle":"2021-07-17T06:04:00.259498Z","shell.execute_reply.started":"2021-07-17T06:04:00.249080Z","shell.execute_reply":"2021-07-17T06:04:00.258323Z"}}
dummy_train = train.copy()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.264725Z","iopub.execute_input":"2021-07-17T06:04:00.267869Z","iopub.status.idle":"2021-07-17T06:04:00.301246Z","shell.execute_reply.started":"2021-07-17T06:04:00.267791Z","shell.execute_reply":"2021-07-17T06:04:00.300074Z"}}
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.306536Z","iopub.execute_input":"2021-07-17T06:04:00.309517Z","iopub.status.idle":"2021-07-17T06:04:00.475978Z","shell.execute_reply.started":"2021-07-17T06:04:00.309437Z","shell.execute_reply":"2021-07-17T06:04:00.474952Z"}}
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating',
    aggfunc=np.min,
)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.477094Z","iopub.execute_input":"2021-07-17T06:04:00.477525Z","iopub.status.idle":"2021-07-17T06:04:00.534755Z","shell.execute_reply.started":"2021-07-17T06:04:00.477494Z","shell.execute_reply":"2021-07-17T06:04:00.533915Z"}}
dummy_train = dummy_train.fillna(1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.535735Z","iopub.execute_input":"2021-07-17T06:04:00.536120Z","iopub.status.idle":"2021-07-17T06:04:00.570120Z","shell.execute_reply.started":"2021-07-17T06:04:00.536091Z","shell.execute_reply":"2021-07-17T06:04:00.569454Z"}}
dummy_train.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.571134Z","iopub.execute_input":"2021-07-17T06:04:00.571515Z","iopub.status.idle":"2021-07-17T06:04:00.649675Z","shell.execute_reply.started":"2021-07-17T06:04:00.571486Z","shell.execute_reply":"2021-07-17T06:04:00.640857Z"}}
item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:00.651114Z","iopub.execute_input":"2021-07-17T06:04:00.651630Z","iopub.status.idle":"2021-07-17T06:04:39.702526Z","shell.execute_reply.started":"2021-07-17T06:04:00.651565Z","shell.execute_reply":"2021-07-17T06:04:39.701738Z"}}
# Take the user ID as input
# user_input = input("Enter your user name")
# print(user_input)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:39.703571Z","iopub.execute_input":"2021-07-17T06:04:39.703987Z","iopub.status.idle":"2021-07-17T06:04:39.716014Z","shell.execute_reply.started":"2021-07-17T06:04:39.703956Z","shell.execute_reply":"2021-07-17T06:04:39.714870Z"}}
# Recommending the Top 5 products to the user.
# recom = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
# recom

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:39.717411Z","iopub.execute_input":"2021-07-17T06:04:39.717873Z","iopub.status.idle":"2021-07-17T06:04:39.750856Z","shell.execute_reply.started":"2021-07-17T06:04:39.717829Z","shell.execute_reply":"2021-07-17T06:04:39.750051Z"}}
# pd.set_option('display.max_columns', None)
# df[df.id == 'AVpe_5U_ilAPnD_xSrxG'].head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:39.751875Z","iopub.execute_input":"2021-07-17T06:04:39.752250Z","iopub.status.idle":"2021-07-17T06:04:39.755820Z","shell.execute_reply.started":"2021-07-17T06:04:39.752221Z","shell.execute_reply":"2021-07-17T06:04:39.754650Z"}}
# recom_ids = np.array(recom.index)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:04:39.757457Z","iopub.execute_input":"2021-07-17T06:04:39.757897Z","iopub.status.idle":"2021-07-17T06:07:27.894307Z","shell.execute_reply.started":"2021-07-17T06:04:39.757854Z","shell.execute_reply":"2021-07-17T06:07:27.893396Z"}}
# corpus=[]
# for i in range(0,29999):
#     review = re.sub('[^a-zA-Z]', ' ', df['reviews_text'][i])
#     review=review.lower()
#     review=review.split()
#     ps=PorterStemmer()
#     review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review=' '.join(review)
#     corpus.append(review)

# from sklearn.feature_extraction.text import CountVectorizer
# cv=CountVectorizer(max_features=1500)
# cv.fit_transform(corpus)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:27.895509Z","iopub.execute_input":"2021-07-17T06:07:27.895837Z","iopub.status.idle":"2021-07-17T06:07:27.901983Z","shell.execute_reply.started":"2021-07-17T06:07:27.895808Z","shell.execute_reply":"2021-07-17T06:07:27.900636Z"}}
# len(cv.vocabulary_)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:27.903329Z","iopub.execute_input":"2021-07-17T06:07:27.903655Z","iopub.status.idle":"2021-07-17T06:07:27.920494Z","shell.execute_reply.started":"2021-07-17T06:07:27.903623Z","shell.execute_reply":"2021-07-17T06:07:27.919087Z"}}
# df_similar = df[df.id.isin(recom_ids)]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:27.923809Z","iopub.execute_input":"2021-07-17T06:07:27.924262Z","iopub.status.idle":"2021-07-17T06:07:27.941890Z","shell.execute_reply.started":"2021-07-17T06:07:27.924218Z","shell.execute_reply":"2021-07-17T06:07:27.940627Z"}}
# id_rating = df_similar.groupby(['id'], sort=False)['reviews_rating'].max()
# k = df_similar.groupby(['id'], sort=False).agg(max_review=pd.NamedAgg(column="reviews_rating", aggfunc="max"))
# id_rating

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:27.943757Z","iopub.execute_input":"2021-07-17T06:07:27.944395Z","iopub.status.idle":"2021-07-17T06:07:27.962075Z","shell.execute_reply.started":"2021-07-17T06:07:27.944344Z","shell.execute_reply":"2021-07-17T06:07:27.960773Z"}}
# df_similar.groupby("id").count()
# df_similar.head()
# df_similar = df_similar.sort_values(["id","reviews_rating"], ascending=False).drop_duplicates(subset="id")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:27.963425Z","iopub.execute_input":"2021-07-17T06:07:27.963764Z","iopub.status.idle":"2021-07-17T06:07:27.967998Z","shell.execute_reply.started":"2021-07-17T06:07:27.963728Z","shell.execute_reply":"2021-07-17T06:07:27.967090Z"}}
# df_similar = df_similar.reset_index(drop=True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:27.969267Z","iopub.execute_input":"2021-07-17T06:07:27.969728Z","iopub.status.idle":"2021-07-17T06:07:28.109044Z","shell.execute_reply.started":"2021-07-17T06:07:27.969696Z","shell.execute_reply":"2021-07-17T06:07:28.108193Z"}}
# corpus=[]
# for i in range(0,20):
#     review = re.sub('[^a-zA-Z]', ' ', df_similar['reviews_text'][i])
#     review=review.lower()
#     review=review.split()
#     ps=PorterStemmer()
#     review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review=' '.join(review)
#     corpus.append(review)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:28.110239Z","iopub.execute_input":"2021-07-17T06:07:28.110739Z","iopub.status.idle":"2021-07-17T06:07:28.115517Z","shell.execute_reply.started":"2021-07-17T06:07:28.110694Z","shell.execute_reply":"2021-07-17T06:07:28.114663Z"}}

# X=cv.transform(corpus).toarray()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:28.116697Z","iopub.execute_input":"2021-07-17T06:07:28.117135Z","iopub.status.idle":"2021-07-17T06:07:29.076631Z","shell.execute_reply.started":"2021-07-17T06:07:28.117099Z","shell.execute_reply":"2021-07-17T06:07:29.075358Z"}}
# classifier = pickle.load(open('model.pkl','rb'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:29.078057Z","iopub.execute_input":"2021-07-17T06:07:29.078359Z","iopub.status.idle":"2021-07-17T06:07:29.097102Z","shell.execute_reply.started":"2021-07-17T06:07:29.078331Z","shell.execute_reply":"2021-07-17T06:07:29.096007Z"}}
# prediction = classifier.predict(X)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:29.098519Z","iopub.execute_input":"2021-07-17T06:07:29.098857Z","iopub.status.idle":"2021-07-17T06:07:29.104951Z","shell.execute_reply.started":"2021-07-17T06:07:29.098828Z","shell.execute_reply":"2021-07-17T06:07:29.104184Z"}}
# prediction

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-17T06:07:29.105945Z","iopub.execute_input":"2021-07-17T06:07:29.106335Z","iopub.status.idle":"2021-07-17T06:07:29.118955Z","shell.execute_reply.started":"2021-07-17T06:07:29.106307Z","shell.execute_reply":"2021-07-17T06:07:29.117812Z"}}
# print(np.array(df_similar['user_sentiment']))
# print(prediction)

# %% [code] {"jupyter":{"outputs_hidden":false}}
df = pd.read_csv('input/sample30.csv')
classifier = pickle.load(open('model.pkl','rb'))
def get_recomm(username):
    recom = item_final_rating.loc[username].sort_values(ascending=False)[0:20]
    recom_ids = np.array(recom.index)
    
    df_similar = df[df.id.isin(recom_ids)]
    df_similar = df_similar.sort_values(["id","reviews_rating"], ascending=False).drop_duplicates(subset="id")
    df_similar = df_similar.reset_index(drop=True)
    
    transformer = pickle.load(open('tranform.pkl','rb'))
    X=transformer.transform(df_similar.reviews_text)
    prediction = classifier.predict(X)
    count = 0
    returnlist = []
    for idx, pred in enumerate(prediction):
        if(pred == 1 and count<5):
            returnlist.append(df_similar.iloc[idx]['name'])
            count = count+1

    return returnlist