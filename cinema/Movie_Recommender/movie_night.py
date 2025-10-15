import streamlit as st
import numpy as np
import pandas as pd
import model_keras.recommender_model as rm
import keras
from threading import RLock
import joblib

# Cache functions
@st.cache_data
def load_data():
    '''
    This function loads all necessary tabular data to use with the recommender model

    Returns:
    movies_encoded_by_genre - All movies one hot encoded by genre
    links_df - Identifiers that can be used to link to other sources of movie data
    '''
    movies_encoded_by_genre = pd.read_csv('cinema/Movie_Recommender/csv_files/movies_encoded_by_genre_big.csv')
    links_df = pd.read_csv('cinema/Movie_Recommender/csv_files/links.csv')
    return movies_encoded_by_genre, links_df

@st.cache_data
def load_constants(movies_encoded_by_genre, quantile=0.8):
    '''
    This function loads the necessary constants

    Returns:
    MIN_NUM_RATINGS - minimum number of user ratings for the movie to be considered for IMDB's rating
    USER_COLS - columns considered for user input vector
    '''

    MIN_NUM_RATINGS = movies_encoded_by_genre['num_ratings'].quantile(quantile)
    USER_COLS = ['[userId]', '[avg_user_rating]', 'Action', 'Adventure', 'Animation',
       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
       'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
       'Western']
    
    return MIN_NUM_RATINGS, USER_COLS

@st.cache_data
def load_scalers():
    '''
    This function loads the necessary scalers to use the model

    Returns:
    scalerUser - user vector standard scaler
    scalerMovies - movies vector standard scaler
    scalerTarget - rating predictions vector min/max scaler
    '''
    scalerUser = joblib.load('cinema/Movie_Recommender/scalers/scalerUser.gz')
    scalerMovies = joblib.load('cinema/Movie_Recommender/scalers/scalerMovies.gz')
    scalerTarget = joblib.load('cinema/Movie_Recommender/scalers/scalerTarget.gz')

    return scalerUser,scalerMovies,scalerTarget

@st.cache_resource
def load_recommender():
    '''
    This function loads the movies recommender model

    Returns:
    model - recommender model deserealized
    '''
    model = keras.models.load_model('cinema/Movie_Recommender/model_keras/content_based_recommender.keras', custom_objects={'NormLayer' : rm.NormLayer})

    return model

# Execution time functions
def imdb_weighted_rating(v, r, c, min_num_ratings):
    '''
    This function calculates the IMDB's weighted rating for a movie
    Parameters:
    v - number of votes for the movie
    r - average rating of the movie
    c - mean vote across the whole report
    '''
    m = min_num_ratings

    return (r*v/(v+m))+(c*m/(v+m))

def gen_user_vec(user_dict):
    '''
    This function generates a user vector with all features to be transformed to an input vector
    Parameters:
    user_dict - dictionary with all genres values
    
    Returns:
    user_vec - User vector with all features to be transformed to an input vector
    '''
    user_vec = np.array([user_dict["[userId]"], user_dict['[avg_user_rating]'],
                        user_dict['Action'], user_dict['Adventure'], user_dict['Animation'], user_dict['Children'],
                        user_dict['Comedy'], user_dict['Crime'], user_dict['Documentary'],
                        user_dict['Drama'], user_dict['Fantasy'], user_dict['Horror'],
                        user_dict['Musical'], user_dict['Mystery'],
                        user_dict['Romance'], user_dict['Sci-Fi'], user_dict['Thriller'],
                        user_dict['War'], user_dict['Western']])

    return user_vec

def gen_movie_vecs(movies_df, user_vec, min_num_ratings, year_filter=2000,top_movies=50):
    '''
    This function generates a movies vector to be an input for the recommender model
    Parameters:
    movies_df - Dataframe with movies one-hot-encoded
    user_vec - User vector with all features to be transformed to an input vector
    year_filter - Parameter for filtering movies dataframe by year
    top_movies - Number of movies to be considered for each genre

    Returns:
    movies_vecs - Movies vector to be an input for the recommender model
    '''
    movies_vecs = movies_df.drop(columns=['title'])
    movies_vecs.rename(columns={'movieId' : '[movieId]', 'num_ratings':'[num_ratings]', 'avg_movie_rating':'[avg_movie_rating]'}, inplace=True)

    # Adding IMDB's wheighted rating
    c = movies_vecs['[avg_movie_rating]'].mean()
    movies_vecs['imdb_rating'] = imdb_weighted_rating(movies_vecs['[num_ratings]'], movies_vecs['[avg_movie_rating]'], c, min_num_ratings).round(2)

    # Filtering by movie year
    year_filter = year_filter
    movies_vecs = movies_vecs[movies_vecs['year'] >= year_filter]

    # Filtering by favorite genres
    movies_vecs.sort_values('imdb_rating', ascending=False, inplace=True)
    aux = pd.DataFrame()
    for i in range(len(user_vec[2:])):
        genre_rate = user_vec[i+2]
        offset = 4
        column = movies_vecs.columns[offset+i]
        if genre_rate >= 3:
            subset = movies_vecs[movies_vecs[column] == 1].head(top_movies)
            aux = pd.concat([aux, subset], axis=0)

    aux = pd.concat([aux, movies_vecs.head(100)], axis=0)

    movies_vecs = aux.drop_duplicates().drop(columns=['imdb_rating'])


    return movies_vecs

def gen_user_vecs(user_vec, movies_vecs, user_cols):
    '''
    This function generetes an user input for the recommender model
    Parameters:
    user_vec - User vector with all features to be transformed to an input vector
    movies_vecs - Movies vector to be an input for the recommender model
    user_cols - User features names

    Returns:
    user_vecs - user input for the recommender model
    '''
    user_columns = user_cols
    user_vecs = pd.DataFrame([user_vec]*len(movies_vecs), columns = user_columns)
    return user_vecs

def make_recommendations(movies_encoded_by_genre, links_df, user_vec, min_num_ratings,user_cols, scalerUser, scalerMovies, scalerTarget, year_filter=2000, top_movies=50):
    # generate and replicate the user vector to match the number movies in the data set.
    movies_vecs = gen_movie_vecs(movies_encoded_by_genre, user_vec,min_num_ratings, year_filter=year_filter, top_movies=top_movies)
    user_vecs = gen_user_vecs(user_vec, movies_vecs, user_cols)

    # scale our user and item vectors
    suser_vecs = scalerUser.transform(user_vecs)
    smovies_vecs = scalerMovies.transform(movies_vecs)

    # make a prediction
    y_p = model.predict([suser_vecs[:, 2:], smovies_vecs[:, 3:]])

    # unscale y prediction 
    y_pu = scalerTarget.inverse_transform(y_p)

    recommendations = pd.merge(movies_vecs['[movieId]'], movies_encoded_by_genre[['movieId','title', 'avg_movie_rating']], left_on='[movieId]', right_on='movieId')
    recommendations = pd.concat([recommendations,pd.DataFrame(y_pu.round(2), columns=['y_pu'])],axis=1)

    output = recommendations[['movieId','title', 'avg_movie_rating', 'y_pu']].sort_values('y_pu',ascending=False).reset_index(drop=True)

    links_df['imdb_url'] = links_df['imdbId'].apply(lambda x: "https://www.imdb.com/title/tt"+ str(x) + "/")
    output = pd.merge(output, links_df[['movieId','imdb_url']], on='movieId')

    return output[['movieId','title','avg_movie_rating','y_pu','imdb_url']]

# Loading data, scalers, constants, and model
movies_encoded_by_genre, links_df = load_data()
MIN_NUM_RATINGS, USER_COLS = load_constants(movies_encoded_by_genre)
MAX_NUM_GENRES = len(USER_COLS[2:])
model = load_recommender()
scalerUser,scalerMovies,scalerTarget = load_scalers()

output_samples = 10

# Initializing session_state attributes and button functions
if 'num_genres' not in st.session_state:
    st.session_state.num_genres = 5

def add_genre(max):
    if st.session_state.num_genres < max:
        st.session_state.num_genres += 1
    else:
        st.session_state.num_genres = max

def clear_genre():
    st.session_state.num_genres = 0

# Initializing user_dict to capture user inputs and generate user vectors
user_dict = {x:0.0 for x in USER_COLS}

# Setting up home page
st.set_page_config(page_title='Movie Night', layout='wide')
st.logo('cinema/Movie_Recommender/img/movie_night_logo.png', size='large')

# Title
with st.container(horizontal=True, horizontal_alignment='center'):
    with st.container(horizontal=True, horizontal_alignment='center',width=1700):
        st.title('_Movie Night !_')



with st.container(horizontal=True, horizontal_alignment='center'):
    with st.container(width=800, horizontal_alignment='center'):
        # Menu Container
        st.header('Menu', divider='gray')
        menu = st.container(horizontal= True, horizontal_alignment='center') 
        with menu:
            filter_container = st.container(horizontal= True, horizontal_alignment='center', width=500)
            with filter_container:
                year_filter = st.slider('Release Year', 1874, 2025, 2000, 1)
                output_samples = st.selectbox('How many movies you want?', [5,10,15,20,30,50],index=None, placeholder="Choose...")

        buttons_container = st.container(horizontal=True)
        with buttons_container:
            add_genre_button = st.button('Add Genre', on_click=add_genre, args=[MAX_NUM_GENRES])
            clear_button = st.button('Clear', on_click=clear_genre)
                

        # Features Selection Container
        features_selec = st.container(height=450)
        with features_selec:
            genre_names_col, rating_col = st.columns(2)
            feature_input = ['']*MAX_NUM_GENRES
            rating_input = np.zeros(MAX_NUM_GENRES)
            if st.session_state.num_genres <= MAX_NUM_GENRES:
                for i in range(st.session_state.num_genres):
                    with genre_names_col:
                        if 'feature_'+str(i) in st.session_state:
                            feature_input[i] = st.session_state['feature_'+str(i)]
                            del st.session_state['feature_'+str(i)]
                            st.selectbox('Genre', USER_COLS[2:],index=None, placeholder="Choose genre...", key='feature_'+str(i)) 
                        else:
                            feature_input[i] = st.selectbox('Genre', USER_COLS[2:],index=None, placeholder="Choose genre...", key='feature_'+str(i))
                    with rating_col:
                        if 'rating_'+str(i) in st.session_state:
                            rating_input[i] = st.session_state['rating_'+str(i)]
                            del st.session_state['rating_'+str(i)]
                            st.slider('How much do you like?', 0.0, 5.0, 0.0, 0.5, key='rating_'+str(i))
                        else:
                            rating_input[i] = st.slider('How much do you like?', 0.0, 5.0, 0.0, 0.5, key='rating_'+str(i))

        # "GIVE ME MOVIE RECOMMENDATIONS !" Button
        st.caption('Obs.: More information you give, more personalized the recommendations will be.')
        recommend_b_container = st.container(horizontal= True, horizontal_alignment='center')
        with recommend_b_container:
            recommend_button = st.button('GIVE ME MOVIE RECOMMENDATIONS !', type='primary')


    with st.container(width=900, horizontal_alignment='center'):
        # Results Container
        results = st.container()
        with results:
            st.header('Selected For You:', divider='grey')
            if recommend_button:
                for i in range(st.session_state.num_genres):
                    user_dict[feature_input[i]] = rating_input[i]
                        
                user_vec = gen_user_vec(user_dict)
                output = make_recommendations(movies_encoded_by_genre, links_df, user_vec, MIN_NUM_RATINGS,USER_COLS, scalerUser, scalerMovies, scalerTarget, year_filter=year_filter)

                if not output_samples:
                    output_samples = 10

                with st.container(horizontal= True,horizontal_alignment='center'):
                    with st.container(horizontal= True,horizontal_alignment='center'):
                        st.dataframe(
                            output.head(output_samples).drop(columns=['movieId']),
                            column_config={
                                'title' : ' Movie Title',
                                'avg_movie_rating': st.column_config.NumberColumn('Movie Night AVG Rate', format='%.2f'),
                                'y_pu' : st.column_config.NumberColumn('Recommender Pred.', format='%.2f'),
                                'imdb_url' : st.column_config.LinkColumn('URL', display_text='IMDB Link')
                            },
                            hide_index=True,
                            width='content'
                        )
                
                with st.container():
                    st.caption("Movie Night AVG Rate - Average rate calculated by movie night recommender model.")
                    st.caption("Recommender Pred. - Predicted rate you would give based on menu selected options.")
                    