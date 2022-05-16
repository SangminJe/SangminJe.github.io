---  
layout: post  
title: "추천시스템의 기본 및 컨텐츠 기반 필터링 실습"
subtitle: ""  
categories: DATA
tags: DATA python recommend contents-filter 컨텐츠기반 필터링 추천시스템입문
comments: true  
---  

# 잠재요인 최근접 이웃 이해하기


```python
import numpy as np

# 원본 행렬 R 생성
R = np.array([[4, np.NaN, np.NaN, 2, np.NaN ],
              [np.NaN, 5, np.NaN, 3, 1 ],
              [np.NaN, np.NaN, 3, 4, 4 ],
              [5, 2, 1, 2, np.NaN ]])
num_users, num_items = R.shape
K=3

# P,Q 행렬의 크기를 지정하고 정규분포를 가진 임의의 값 임력
np.random.seed(1)
P = np.random.normal(scale=1./K, size=(num_users, K))
Q = np.random.normal(scale=1./K, size=(num_items, K))
```


```python
P
```




    array([[ 0.54144845, -0.2039188 , -0.17605725],
           [-0.35765621,  0.28846921, -0.76717957],
           [ 0.58160392, -0.25373563,  0.10634637],
           [-0.08312346,  0.48736931, -0.68671357]])




```python
Q
```




    array([[-0.1074724 , -0.12801812,  0.37792315],
           [-0.36663042, -0.05747607, -0.29261947],
           [ 0.01407125,  0.19427174, -0.36687306],
           [ 0.38157457,  0.30053024,  0.16749811],
           [ 0.30028532, -0.22790929, -0.04096341]])




```python
num_users
```




    4




```python
np.dot(P,Q.T)
```




    array([[-9.86215743e-02, -1.35273244e-01,  3.25938576e-02,
             1.15829937e-01,  2.16275915e-01],
           [-2.88426030e-01,  3.39039250e-01,  3.32466259e-01,
            -1.78279922e-01, -1.41717429e-01],
           [ 1.01671414e-02, -2.29768982e-01, -8.01253853e-02,
             1.63482851e-01,  2.28119515e-01],
           [-3.12983578e-01,  2.03409279e-01,  3.45449141e-01,
            -2.71808518e-04, -1.07906618e-01]])




```python
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)
    
    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
      
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
    
    return rmse
```


```python
non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ]

steps=1000
learning_rate=0.01
r_lambda=0.01

# SGD 기법으로 P와 Q 매트릭스를 계속 업데이트. 
for step in range(steps):
    for i, j, r in non_zeros:
        # 실제 값과 예측 값의 차이인 오류 값 구함
        eij = r - np.dot(P[i, :], Q[j, :].T)
        # Regularization을 반영한 SGD 업데이트 공식 적용
        P[i,:] = P[i,:] + learning_rate*(eij * Q[j, :] - r_lambda*P[i,:])
        Q[j,:] = Q[j,:] + learning_rate*(eij * P[i, :] - r_lambda*Q[j,:])

    rmse = get_rmse(R, P, Q, non_zeros)
    if (step % 50) == 0 :
        print("### iteration step : ", step," rmse : ", rmse)
```

    ### iteration step :  0  rmse :  3.2388050277987723
    ### iteration step :  50  rmse :  0.48767231013696477
    ### iteration step :  100  rmse :  0.1564340384819248
    ### iteration step :  150  rmse :  0.07455141311978032
    ### iteration step :  200  rmse :  0.0432522679857931
    ### iteration step :  250  rmse :  0.029248328780879088
    ### iteration step :  300  rmse :  0.022621116143829344
    ### iteration step :  350  rmse :  0.01949363619652533
    ### iteration step :  400  rmse :  0.018022719092132503
    ### iteration step :  450  rmse :  0.01731968595344277
    ### iteration step :  500  rmse :  0.01697365788757103
    ### iteration step :  550  rmse :  0.016796804595895533
    ### iteration step :  600  rmse :  0.01670132290188455
    ### iteration step :  650  rmse :  0.016644736912476654
    ### iteration step :  700  rmse :  0.016605910068210192
    ### iteration step :  750  rmse :  0.01657420047570466
    ### iteration step :  800  rmse :  0.016544315829215932
    ### iteration step :  850  rmse :  0.01651375177473506
    ### iteration step :  900  rmse :  0.016481465738195183
    ### iteration step :  950  rmse :  0.016447171683479173
    


```python
pred_matrix = np.dot(P, Q.T)
print(pred_matrix)
```

    [[3.99062329 0.89653623 1.30649077 2.00210666 1.66340846]
     [6.69571106 4.97792757 0.97850229 2.98066034 1.0028451 ]
     [6.67689303 0.39076095 2.98728588 3.9769208  3.98610743]
     [4.96790858 2.00517956 1.00634763 2.01691675 1.14044567]]
    

# TMDB 데이터 - 컨텐츠 기반 필터링


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

movies = pd.read_csv("./data/tmdb/tmdb_5000_movies.csv")
print(movies.shape)
movies.head(3)
```

    (4803, 20)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>[{"iso_3166_1": "GB", "name": "United Kingdom"...</td>
      <td>2015-10-26</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_df = movies[['id','title', 'genres', 'vote_average', 'vote_count', 'popularity', 'keywords', 'overview']]
```


```python
pd.set_option("max_colwidth", 100)
movies_df[["genres", "keywords"]][:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {...</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id": 2964, "name": "future"}, {"id": 3386, "name": "sp...</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(movies_df["genres"][0])
```




    str



- 현재 `genre` 컬럼의 리스트 내의 딕셔너리 형태의 데이터가 있음
- 하지만 데이터 타입을 보면 **string** 형태임
- `literal_eval` 함수를 사용해서 이를 객체로 인식시킬 수 있음
- 즉 str을 읽어서 리스트와 딕셔너리 형태로 맞게 데이터를 바꿔줌


```python
from ast import literal_eval # 리스트 내의 딕셔너리 형태의 스트링 값을 객체로 바꿔주기 위해 사용
movies_df["genres"] = movies_df["genres"].apply(literal_eval)
```


```python
movies_df["genres"]
```




    0       [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'}, {...
    1            [{'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'}, {'id': 28, 'name': 'Action'}]
    2              [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}, {'id': 80, 'name': 'Crime'}]
    3       [{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}, {'id': 18, 'name': 'Drama'}, {'id': ...
    4       [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}, {'id': 878, 'name': 'Science Fic...
                                                           ...                                                 
    4798            [{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}, {'id': 53, 'name': 'Thriller'}]
    4799                                       [{'id': 35, 'name': 'Comedy'}, {'id': 10749, 'name': 'Romance'}]
    4800    [{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}, {'...
    4801                                                                                                     []
    4802                                                                    [{'id': 99, 'name': 'Documentary'}]
    Name: genres, Length: 4803, dtype: object




```python
movies_df["genres"][0][0]["name"]
```




    'Action'



- 데이터가 잘 읽힘


```python
movies_df["keywords"] = movies_df["keywords"].apply(literal_eval)
```


```python
# name 키에 해당하는 값만 추출
movies_df["genres"] = movies_df["genres"].apply(lambda x : [y["name"] for y in x])
movies_df["keywords"] = movies_df["keywords"].apply(lambda x : [y["name"] for y in x])
movies_df[["genres", "keywords"]][:1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genres</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Action, Adventure, Fantasy, Science Fiction]</td>
      <td>[culture clash, future, space war, space colony, society, space travel, futuristic, romance, spa...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_extraction.text import CountVectorizer

# Count Vectorizer를 적용하기 위해 공백문자로 word 단위로 구분되는 문자열로 반환
movies_df["genres_literal"] = movies_df["genres"].apply(lambda x : (' ').join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(movies_df["genres_literal"])
print(genre_mat.shape)
```

    (4803, 276)
    


```python
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:1])
```

    (4803, 4803)
    [[1.         0.59628479 0.4472136  ... 0.         0.         0.        ]]
    


```python
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1] # [::-1]은 내림차순 정렬
print(genre_sim_sorted_ind[:1])
```

    [[   0 3494  813 ... 3038 3037 2401]]
    

- 가장 높은 유사도를 가진 것은 0번 레코드 다음 ,3494 레코드
- 가장 낮은 유사도는 2401번 레코드라는 의미로 해석함


```python
# 장르 유사도에 따라 영화를 추천하는 함수 생성
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    # 인자로 입력된 title_name인 df 추출
    title_movie = df[df["title"] == title_name]
    
    title_index = title_movie.index.values
    # top_n개의 index 추출
    similar_indexes = sorted_ind[title_index, :(top_n)]
    
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)
    
    return df.iloc[similar_indexes]
```


```python
similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, "The Godfather", 10)
similar_movies[["title", "vote_average"]]
```

    [[2731 1243 3636 1946 2640 4065 1847 4217  883 3866]]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>Mean Streets</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>3636</th>
      <td>Light Sleeper</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>The Bad Lieutenant: Port of Call - New Orleans</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>Things to Do in Denver When You're Dead</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>4065</th>
      <td>Mi America</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>GoodFellas</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>4217</th>
      <td>Kids</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>883</th>
      <td>Catch Me If You Can</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>City of God</td>
      <td>8.1</td>
    </tr>
  </tbody>
</table>
</div>



- Mi America는 평점이 0점
- 낯선 영화도 많음
- 개선이 필요함


```python
movies_df[["title", "vote_average", "vote_count"]].sort_values("vote_average", ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3519</th>
      <td>Stiff Upper Lips</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4247</th>
      <td>Me You and Five Bucks</td>
      <td>10.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4045</th>
      <td>Dancer, Texas Pop. 81</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4662</th>
      <td>Little Big Top</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3992</th>
      <td>Sardaarji</td>
      <td>9.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2386</th>
      <td>One Man's Hero</td>
      <td>9.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2970</th>
      <td>There Goes My Baby</td>
      <td>8.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>The Shawshank Redemption</td>
      <td>8.5</td>
      <td>8205</td>
    </tr>
    <tr>
      <th>2796</th>
      <td>The Prisoner of Zenda</td>
      <td>8.4</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3337</th>
      <td>The Godfather</td>
      <td>8.4</td>
      <td>5893</td>
    </tr>
  </tbody>
</table>
</div>



- `vote_average`순으로 내림차순 정렬해보면 `vote_count`가 1~2개인데 높은 점수를 준 경우가 있음
- 이는 왜곡된 평점데이터라고 가정하고 가중치가 부여된 평점으로 개선
- 가중평점 = (v/(v+m)) \* R + (m/(v+m)) \* C
    - v : 개별영화 평점을 투표한 횟수
    - m : 평점을 부여하기 위한 최소 투표횟수
    - R : 개별 영화에 대한 평균 평점
    - C : 전체 영화에 대한 평균 평점


```python
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.6)
print("C:", round(C, 3), "m:", round(m, 3))
```

    C: 6.092 m: 370.2
    


```python
percentile = 0.6
m = movies_df["vote_count"].quantile(percentile)
C = movies_df["vote_average"].mean()

def weighted_vote_average(record):
    v = record["vote_count"]
    R = record["vote_average"]
    
    return ((v/(v+m))*R)+ ((m/(m+v))*C)

movies_df["weighted_vote"] = movies.apply(weighted_vote_average, axis=1)
```


```python
movies_df[["title", "vote_average", "weighted_vote", "vote_count"]].sort_values("weighted_vote", ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
      <th>weighted_vote</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1881</th>
      <td>The Shawshank Redemption</td>
      <td>8.5</td>
      <td>8.396052</td>
      <td>8205</td>
    </tr>
    <tr>
      <th>3337</th>
      <td>The Godfather</td>
      <td>8.4</td>
      <td>8.263591</td>
      <td>5893</td>
    </tr>
    <tr>
      <th>662</th>
      <td>Fight Club</td>
      <td>8.3</td>
      <td>8.216455</td>
      <td>9413</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>Pulp Fiction</td>
      <td>8.3</td>
      <td>8.207102</td>
      <td>8428</td>
    </tr>
    <tr>
      <th>65</th>
      <td>The Dark Knight</td>
      <td>8.2</td>
      <td>8.136930</td>
      <td>12002</td>
    </tr>
    <tr>
      <th>1818</th>
      <td>Schindler's List</td>
      <td>8.3</td>
      <td>8.126069</td>
      <td>4329</td>
    </tr>
    <tr>
      <th>3865</th>
      <td>Whiplash</td>
      <td>8.3</td>
      <td>8.123248</td>
      <td>4254</td>
    </tr>
    <tr>
      <th>809</th>
      <td>Forrest Gump</td>
      <td>8.2</td>
      <td>8.105954</td>
      <td>7927</td>
    </tr>
    <tr>
      <th>2294</th>
      <td>Spirited Away</td>
      <td>8.3</td>
      <td>8.105867</td>
      <td>3840</td>
    </tr>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
      <td>8.079586</td>
      <td>3338</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 장르 유사도에 따라 영화를 추천하는 함수 생성
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    # 인자로 입력된 title_name인 df 추출
    title_movie = df[df["title"] == title_name]
    title_index = title_movie.index.values
    
    # top_n X 2개의 index 추출
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)
    
    # 기준영화 인덱스 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    return df.iloc[similar_indexes].sort_values("weighted_vote", ascending=False)[:top_n]

similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, "The Godfather", 10)
similar_movies[["title", "vote_average"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>GoodFellas</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>City of God</td>
      <td>8.1</td>
    </tr>
    <tr>
      <th>1663</th>
      <td>Once Upon a Time in America</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>883</th>
      <td>Catch Me If You Can</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>281</th>
      <td>American Gangster</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>This Is England</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>American Hustle</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>Mean Streets</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>2839</th>
      <td>Rounders</td>
      <td>6.9</td>
    </tr>
  </tbody>
</table>
</div>


