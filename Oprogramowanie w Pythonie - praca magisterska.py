
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PyCharm w wersji 2019.2.6


Najważniejsze informacje dotyczące bibliotek:

Keras - wersja 2.4.0
TensorFlow - wersja 2.4.0
Pozostałe biblioteki - najnowsze wersje na dzień 09.08.2021r.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""






"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import bibliotek
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import ast
import datetime
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfkit
import seaborn as sns
import scikit_posthocs as sp
import scipy.stats as stats
import shap
import tensorflow as tf

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from math import trunc
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS














"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Zdefiniowanie funkcji
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def convert_to_int(x):
    """
    Wczytuje wartość, a następnie:
        1) jeśli to możliwe, zwraca tę samą wartość przerobioną na typ 'int',
        2) jeśli nie jest to możliwe, zwraca wartość np.nan.
    """
    try:
        return int(x)
    except:
        return np.nan


def wez_miesiac(x):
    """
    Wczytuje wartość daty, a następnie:
        1) jeśli to możliwe, rozdziela datę na listę separatorem '-' i wybiera drugi element tej listy (nr miesiąca),
        2) jeśli nie jest to możliwe, zwraca wartość np.nan.
    """
    try:
        return int(str(x).split('-')[1])
    except:
        return np.nan


def wez_nazwe_miesiaca(x):
    """
    Wczytuje wartość daty, a następnie:
        1) jeśli to możliwe, rozdziela datę na listę separatorem '-' i wybiera nazwę miesiąca odpowiadającą drugiemu elementowi tej listy,
        2) jeśli nie jest to możliwe, zwraca wartość np.nan.
    """
    try:
        return miesiace[
            int(str(x).split('-')[1]) - 1
        ]
    except:
        return np.nan


def wez_dzien(x):
    """
    Wczytuje wartość daty, a następnie:
        1) jeśli to możliwe, rozdziela datę na listę separatorem '-' i wybiera trzeci element tej listy (dzień),
        2) jeśli nie jest to możliwe, zwraca wartość np.nan.
    """
    try:
        year, month, day = (
            int(i) for i in x.split('-')
        )
        return day
    except:
        return np.nan


def wez_dzien_tyg(x):
    """
    Wczytuje wartość daty, a następnie:
        1) jeśli to możliwe, rozdziela datę na listę separatorem '-' i wybiera nazwę dnia tygodnia odpowiadającą trzeciemu elementowi tej listy,
        2) jeśli nie jest to możliwe, zwraca wartość np.nan.
    """
    try:
        year, month, day = (
            int(i) for i in x.split('-')
        )
        answer = datetime.date(
            year,
            month,
            day).weekday()
        return dni[answer]
    except:
        return np.nan


def wez_nr_dnia_tyg(x):
    """
    Wczytuje wartość daty, a następnie:
        1) jeśli to możliwe, rozdziela datę na listę separatorem '-' i wybiera nr dnia tygodnia odpowiadający trzeciemu elementowi tej listy,
        2) jeśli nie jest to możliwe, zwraca wartość np.nan.
    """
    try:
        year, month, day = (
            int(i) for i in x.split('-')
        )
        answer = datetime.date(
            year,
            month,
            day).weekday()
        return dni_nr[answer]
    except:
        return np.nan


def list_el_in_oth_list (list1, list2):
    """
    Funkcja sprawdzająca, czy przynajmniej jeden element z jeden listy zawiera się w elementach innej listy.
    """
    return any([element in list2 for element in list1])


def get_job(x):
    """
    Wczytuje listę słowników a następnie zwraca listę wszystkich zawodów filmowych.
    """
    y=[]
    for i in x:
        y.append(i['job'])
    return y


def get_producer(x):
    """
    Wczytuje listę słowników a następnie zwraca listę wszystkich producentów danego filmu.
    """
    y=[]
    for i in x:
        if i['job'] == 'Producer':
            y.append(i['name'])
        elif i['job'] == 'Executive Producer':
            y.append(i['name'])
    return y


def get_director(x):
    """
    Wczytuje listę słowników a następnie zwraca listę wszystkich reżyserów danego filmu.
    """
    y=[]
    for i in x:
        if i['job'] == 'Director':
            y.append(i['name'])
    return y


def get_screenplay(x):
    """
    Wczytuje listę słowników a następnie zwraca listę wszystkich scenarzystów danego filmu.
    """
    y=[]
    for i in x:
        if i['job'] == 'Screenplay':
            y.append(i['name'])
    return y



def actModel(model, activation):
    """
    Funkcja pomocnicza, która pozwoli wybierać po imieniu zaawansowane funkcje aktywacji
    """
    if activation=='prelu':
        model.add(keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero"))
    else:
        model.add(Activation(activation))
    return





def create_network(n=3, nu1=10, nu2=5, nu3=5,
                   activation='sigmoid',
                   dropout=Dropout,
                   dropout_rate=0,
                   regu=0,
                   kernel_initializer='lecun_normal',
                   optimizer='SGD',
                   num_classes=1,
                   inputShape=10, dropout_all_layers=False, debug=False, *args, **kwargs):

    """
    Wydzielona funkcja, która odpowiada wyłącznie za budowę sieci.
    """

    nu = [nu1, nu2, nu3] # 3 warstwy ukryte
    #nu = [nu1, nu2] # 2 warstwy ukryte

    # Jeżeli włączony jest tryb debug wyświetlą się parametry funkcji
    if debug:
        print(locals())

    # Inicjacja podstawowego modelu keras w trybie sekwencyjnym
    # W ten sposób najłatwiej zbutować w pełni połączone sieci feed forward.
    model = Sequential()

    # Definicja inputów do sieci oraz pierwszej warstwy
    # nu[0] odpowiada za liczbę neuronów w pierwszej warstwie
    # dodatkowo ustala się jak inicjalizować parametry: raz czy stosować regularyzację.
    model.add(Dense(nu[0], input_shape=(inputShape,),
                    kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(regu)))

    # Dodanie funkcji aktywacji do pierwszej warstwy.

    actModel(model, activation)

    # Opcjonalne włączenie mechanizmu dropout

    if dropout_rate > 0.01:
        model.add(dropout(dropout_rate))

    # Kolejne warstwy budowane są w ten sam sposób, co pozwala budować je w pętli
    # Pierwsza warstwa jako jedyna musiała mieć zdefiniowany input
    # W pozostałych warstwach model automatycznie keras połączy warstwy,
    # Input do kolejnej warstwy będzie outputem z wcześniejszych warstw.

    for i in range(1, n):

        # Inicjalizacja warstwy przez podanie liczby neuronów oraz sposobu inicjacji
        model.add(Dense(nu[i], kernel_initializer=kernel_initializer))
        # Definicja funkcji aktywacji
        actModel(model, activation)

        # Opcjonalne dodanie dropout
        if dropout_rate > 0.01 and dropout_all_layers:
            model.add(dropout(dropout_rate))

    # Aby "zakończyć" sieć niezbędne jest przygotowanie odpowiedniej liczby neuronów
    # Wystarczy jeden neuron dla regresji
    # Dla klasyfikacji binarnej oraz wieloklasowej potrzeba tyle neuronów co klas.
    # Dla binarne będą to dwa neurony.
    # W przypadku klasyfikacji wieloklasowej target musi być również podany w postaci one-hot encoding.
    model.add(Dense(num_classes))

    # Dla klasyfikacji binarnej lub wieloklasowej jako funkcję aktywacji stosuje się softmax
    # Będzie on odpowiadał transformacji logistycznej.
    # W problemie regresji można po prostu wykorzystać funkcję liniową.
    # Na chwilę obecną buduje się wrapper dla klasyfikacji binarnej
    # Poniżej pozostawiony jest softmax, z funkcją straty categorical_crossentropy oraz accuracy jak bazową metryką.
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model





def net(train, test, features, target, df, epochs=30, batchSize=100, debug=False, *args, **kwargs):
    """
    Funkcja net jest wrapperem do budowania siecie, trenowania, oraz zbierania wyników/statystyk.
    """

    # Przygotowanie zmiennych
    y_train = tf.keras.utils.to_categorical(train[target].values)
    y_test = tf.keras.utils.to_categorical(test[target].values)
    x_train = train[features].values
    x_test = test[features].values
    df_sv = df[features].values

    num_classes = y_train.shape[1]

    # Wyświetlenie informacji opisowych jeśli włączony jest tryb debugowy
    if debug:
        print('Loading data...')
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        print(num_classes, 'classes')
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('\nBuilding network 1...')

    # Stworzenie modelu z użyciem funkcji create_network.
    model = create_network(num_classes=num_classes, inputShape=x_train.shape[1], debug=debug, *args, **kwargs)

    # Przygotowanie zmiennych do przechowywania statystyk dotyczących szkolenia
    history_model = dict()
    loss = []
    valLoss = []
    auc = []
    valAuc = []

    bestTrainProba = []
    bestTestProba = []
    bestAuc = 0

    # Iteracyjne uruchomienie szkolenia.
    # UWAGA! W "profesjonalnym" wykorzystaniu Kerasa stosuje się tzw. callbacki,
    # callbacki to funkcje do uruchamiania pomiędzy iteracjami.
    # W tej dydaktycznej implementacji należy samodzielnie iterować szkolenie w pętli po jednej iteracji

    for z in range(epochs):
        # Wykonanie jednej ideacji szkolenia
        histModel = model.fit(x_train,
                              y_train,
                              batch_size=batchSize,
                              epochs=1,
                              verbose=debug,
                              validation_split=0.0,
                              validation_data=(x_test, y_test))
        # Utworzenie prognozy na zbiorze treningowym
        probaTrain = model.predict(x_train,
                                   batch_size=y_train.shape[0],
                                   verbose=debug)
        # Utworzenie prognozy na zbiorze testowym
        probaTest = model.predict(x_test,
                                  batch_size=y_test.shape[0],
                                  verbose=debug)
        # Obliczenie statystyk AUC
        # We wcześniejszych wersjach Keras nie miał wbudowanego liczenia auc pomiędzy iteracjami
        aucTrain = roc_auc_score(train[target], probaTrain[:, 1])
        aucTest = roc_auc_score(test[target], probaTest[:, 1])

        if debug:
            print(aucTrain, aucTest)

        # Ręczna implementacja zapisywania prognoz z najlepszej iteracji
        if aucTest > bestAuc:
            bestAuc = aucTest
            bestTrainProba = probaTrain[:, 1].tolist()
            bestTestProba = probaTest[:, 1].tolist()

        # Zapisanie wyników w tej iteracji
        loss.append(histModel.history['loss'][0])
        valLoss.append(histModel.history['val_loss'][0])
        auc.append(aucTrain)
        valAuc.append(aucTest)

    # Zapisanie wyników ze wszystkich iteracji do słownika
    history_model['loss'] = loss
    history_model['valLoss'] = valLoss
    history_model['auc'] = auc
    history_model['valAuc'] = valAuc

    # Zapisanie istotności parametrów (analiza wrażliwości)
    background = x_train[np.random.choice(x_train.shape[0], 3000, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(df_sv)

    return max(valAuc), bestTrainProba, bestTestProba, history_model, shap_values




































"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Rozdział 2: Przygotowanie danych
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

##############################################################################################################################
# KOD 1 - Wczytanie danych
##############################################################################################################################

filmy_raw = pd.read_csv(r"data\movies_metadata.csv",
                        low_memory=False)
print(filmy_raw.shape)


##############################################################################################################################
# KOD 2 - kilka pierwszych obserwacji
##############################################################################################################################

df = filmy_raw.head(3).transpose()
f = open('exp.html','w')
a = df.to_html()
f.write(a)
f.close()
pdfkit.from_file('exp.html', 'first_obs.pdf')


##############################################################################################################################
#KOD 3 - Informacje o zmiennych
##############################################################################################################################

filmy_raw.info()


##############################################################################################################################
# Kod 4 - deduplikacja i dołączenie obsady i ekipy filmowej
##############################################################################################################################

filmy0 = filmy_raw.drop_duplicates(subset=['id'])
print('Filmy0:')
print(filmy0.shape)
filmy0['id'] = filmy0['id'].apply(convert_to_int)
ludzie_filmu = pd.read_csv(r'data\credits.csv')
ludzie_filmu.id = ludzie_filmu.id.astype('int')
ludzie_filmu = ludzie_filmu.drop_duplicates(subset=['id'])
print('Ludzie filmu:')
print(ludzie_filmu.shape)
print("\n")
filmy1 = filmy0[filmy0['id'].notnull()].merge(ludzie_filmu[ludzie_filmu['id'].notnull()], on='id')
print("Filmy1 (Filmy0 z dołączonymi ludźmi filmu):")
print(filmy1.shape)


##############################################################################################################################
# Kod 5 - Uporanie się z najbardziej skomplikowanymi zmiennymi - przemiana list słowników
#         w listy pojedynczych wartości i utworzenie tej podstawie zmiennych ilościowych
##############################################################################################################################

######################
# Genres
######################
filmy1['genres'] = filmy1['genres'].fillna('[]')
filmy1['genres'] = filmy1['genres'].apply(ast.literal_eval)
filmy1['genres'] = filmy1['genres'].apply(
    lambda x: [i['name'] for i in x]
    if isinstance(x, list)
    else []
)
filmy1['genres_count'] = filmy1['genres'].apply(
    lambda x: len(x)
)

######################
# Production companies
######################
filmy1['production_companies'] = filmy1['production_companies'].fillna('[]')
filmy1['production_companies'] = filmy1['production_companies'].apply(ast.literal_eval)
filmy1['production_companies'] = filmy1['production_companies'].apply(
    lambda x: [i['name'] for i in x]
    if isinstance(x, list)
    else [])
filmy1['prod_comp_count'] = filmy1['production_companies'].apply(
    lambda x: len(x)
)

######################
# Production countries
######################
filmy1['production_countries'] = filmy1['production_countries'].fillna('[]')
filmy1['production_countries'] = filmy1['production_countries'].apply(ast.literal_eval)
filmy1['production_countries'] = filmy1['production_countries'].apply(
    lambda x: [i['name'] for i in x]
    if isinstance(x, list)
    else [])
filmy1['prod_countr_count'] = filmy1['production_countries'].apply(
    lambda x: len(x)
)

######################
# Spoken languages
######################
filmy1['spoken_languages'] = filmy1['spoken_languages'].fillna('[]')
filmy1['spoken_languages'] = filmy1['spoken_languages'].apply(ast.literal_eval)
filmy1['spoken_languages'] = filmy1['spoken_languages'].apply(
    lambda x: [i['name'] for i in x]
    if isinstance(x, list)
    else [])
filmy1['spok_lang_count'] = filmy1['spoken_languages'].apply(
    lambda x: len(x)
)

######################
# Cast
######################
filmy1['cast'] = filmy1['cast'].apply(ast.literal_eval)
filmy1['cast_size'] = filmy1['cast'].apply(lambda x: len(x))
filmy1['cast'] = filmy1['cast'].apply(
    lambda x: [i['name'] for i in x]
    if isinstance(x, list)
    else []
)

######################
# Crew
######################
filmy1['crew'] = filmy1['crew'].apply(ast.literal_eval)
filmy1['crew_size'] = filmy1['crew'].apply(lambda x: len(x))



##############################################################################################################################
# KOD 6.1 - zmienne ilościowe (wyczyszczenie - m. in. usunięcie outlierów)
##############################################################################################################################

######################
# budget
######################
filmy1.budget = pd.to_numeric(filmy1.budget, errors='coerce')
filmy1.budget = filmy1.budget.replace(0, np.nan)

######################
# revenue
######################
filmy1.revenue = pd.to_numeric(filmy1.revenue, errors='coerce')
filmy1.revenue = filmy1.revenue.replace(0, np.nan)
filmy1[filmy1.revenue.isnull()].shape

######################
# runtime
######################
filmy1.runtime = pd.to_numeric(filmy1.runtime, errors='coerce')
filmy1.runtime = filmy1.runtime.replace(0, np.nan)
filmy1[filmy1.runtime.isnull()].shape

######################
# vote_average
######################
filmy1.vote_average = pd.to_numeric(filmy1.vote_average, errors='coerce')
filmy1.vote_average = filmy1.vote_average.replace(0, np.nan)
filmy1[filmy1.vote_average.isnull()].shape

######################
# vote_count
######################
filmy1.vote_count = pd.to_numeric(filmy1.vote_count, errors='coerce')
filmy1.vote_count = filmy1.vote_count.replace(0, np.nan)
filmy1[filmy1.vote_count.isnull()].shape

######################
# popularity
######################
filmy1.popularity = pd.to_numeric(filmy1.popularity, errors='coerce')
filmy1.popularity = filmy1.popularity.replace(0, np.nan)
filmy1[filmy1.popularity.isnull()].shape

######################
# genres_count
######################
filmy1.genres_count = pd.to_numeric(filmy1.genres_count, errors='coerce')
filmy1.genres_count = filmy1.genres_count.replace(0, np.nan)
filmy1[filmy1.genres_count.isnull()].shape

######################
# prod_comp_count
######################
filmy1.prod_comp_count = pd.to_numeric(filmy1.prod_comp_count, errors='coerce')
filmy1.prod_comp_count = filmy1.prod_comp_count.replace(0, np.nan)
filmy1[filmy1.prod_comp_count.isnull()].shape

######################
# prod_countr_count
######################
filmy1.prod_countr_count = pd.to_numeric(filmy1.prod_countr_count, errors='coerce')
filmy1.prod_countr_count = filmy1.prod_countr_count.replace(0, np.nan)
filmy1[filmy1.prod_countr_count.isnull()].shape

######################
# spok_lang_count
######################
filmy1.spok_lang_count = pd.to_numeric(filmy1.spok_lang_count, errors='coerce')
filmy1.spok_lang_count = filmy1.spok_lang_count.replace(0, np.nan)
filmy1[filmy1.spok_lang_count.isnull()].shape

######################
# cast_size
######################
filmy1.cast_size = pd.to_numeric(filmy1.cast_size, errors='coerce')
filmy1.cast_size = filmy1.cast_size.replace(0, np.nan)
filmy1[filmy1.cast_size.isnull()].shape

######################
# crew_size
######################
filmy1.crew_size = pd.to_numeric(filmy1.crew_size, errors='coerce')
filmy1.crew_size = filmy1.crew_size.replace(0, np.nan)
filmy1[filmy1.crew_size.isnull()].shape



##############################################################################################################################
# Kod 6.2 - Wyciągnięcie tylko pełnych danych
##############################################################################################################################

filmy11 = filmy1[ filmy1.budget.notnull() & filmy1.revenue.notnull() & filmy1.runtime.notnull() & filmy1.vote_average.notnull() &
                  filmy1.vote_count.notnull() & filmy1.popularity.notnull() & filmy1.genres_count.notnull() & filmy1.prod_comp_count.notnull() &
                  filmy1.prod_countr_count.notnull() & filmy1.spok_lang_count.notnull() & filmy1.cast_size.notnull() & filmy1.crew_size.notnull()]
print(filmy11.shape)


##############################################################################################################################
# Kod 6.3 - Likwidacja outlierów - CAP
##############################################################################################################################

######################
# budget
######################
print(round(filmy11['budget'].describe()))
Q1_budget = filmy11['budget'].describe()['25%']
Q3_budget = filmy11['budget'].describe()['75%']
IQR_budget = Q3_budget - Q1_budget
filmy11['budget_cap'] = filmy11['budget'].apply(
    lambda x: Q3_budget + (1.5 * IQR_budget)
    if x > Q3_budget + (1.5 * IQR_budget)
    else x)
round(filmy11['budget_cap'].describe())

######################
# revenue
######################
round(filmy11['revenue'].describe())
Q1_revenue = round(filmy11['revenue'].describe())['25%']
Q3_revenue = round(filmy11['revenue'].describe())['75%']
IQR_revenue = Q3_revenue - Q1_revenue
filmy11['revenue_cap'] = filmy11['revenue'].apply(
    lambda x: Q3_revenue + (1.5 * IQR_revenue)
    if x > Q3_revenue + (1.5 * IQR_revenue)
    else x)
round(filmy11['revenue_cap'].describe())

######################
# runtime
######################
print(round(filmy11['runtime'].describe()))
Q1_runtime = filmy11['runtime'].describe()['25%']
Q3_runtime = filmy11['runtime'].describe()['75%']
IQR_runtime = Q3_runtime - Q1_runtime
runtime_cap1 = filmy11['runtime'].apply(
    lambda x: Q3_runtime + (1.5 * IQR_runtime)
    if x > Q3_runtime + (1.5 * IQR_runtime)
    else x)
runtime_cap2 = runtime_cap1.apply(
    lambda x: Q1_runtime - (1.5 * IQR_runtime)
    if x < Q1_runtime - (1.5 * IQR_runtime)
    else x)
filmy11['runtime_cap'] = runtime_cap2
round(filmy11['runtime_cap'].describe())

######################
# vote_average
######################
print(round(filmy11['vote_average'].describe(), 1))
Q1_vote_average = filmy11['vote_average'].describe()['25%']
Q3_vote_average = filmy11['vote_average'].describe()['75%']
IQR_vote_average = Q3_vote_average - Q1_vote_average
vote_average_cap1 = filmy11['vote_average'].apply(
    lambda x: Q3_vote_average + (1.5 * IQR_vote_average)
    if x > Q3_vote_average + (1.5 * IQR_vote_average)
    else x
)
vote_average_cap2 = vote_average_cap1.apply(
    lambda x: Q1_vote_average - (1.5 * IQR_vote_average)
    if x < Q1_vote_average - (1.5 * IQR_vote_average)
    else x
)
filmy11['vote_average_cap'] = vote_average_cap2
round(filmy11['vote_average_cap'].describe(), 1)

######################
# vote_count
######################
print(filmy11['vote_count'].describe())
Q1_vote_count = filmy11['vote_count'].describe()['25%']
Q3_vote_count = filmy11['vote_count'].describe()['75%']
IQR_vote_count = Q3_vote_count - Q1_vote_count
filmy11['vote_count_cap'] = filmy11['vote_count'].apply(
    lambda x: Q3_vote_count + (1.5 * IQR_vote_count)
    if x > Q3_vote_count + (1.5 * IQR_vote_count)
    else x
)
filmy11['vote_count_cap'].describe()

######################
# popularity
######################
print(filmy11['popularity'].describe())
Q1_popularity = filmy11['popularity'].describe()['25%']
Q3_popularity = filmy11['popularity'].describe()['75%']
IQR_popularity = Q3_popularity - Q1_popularity
filmy11['popularity_cap'] = filmy11['popularity'].apply(
    lambda x: Q3_popularity + (1.5 * IQR_popularity)
    if x > Q3_popularity + (1.5 * IQR_popularity)
    else x
)
print(filmy11['popularity_cap'].describe())

######################
# genres_count
######################
filmy11.genres_count = filmy11.genres_count.replace(0, np.nan)
print(round(filmy11['genres_count'].describe(), 2))
Q1_genres_count = filmy11['genres_count'].describe()['25%']
Q3_genres_count = filmy11['genres_count'].describe()['75%']
IQR_genres_count = Q3_genres_count - Q1_genres_count
filmy11['genres_count_cap'] = filmy11['genres_count'].apply(
    lambda x: round(Q3_genres_count + (1.5 * IQR_genres_count))
    if x > Q3_genres_count + (1.5 * IQR_genres_count)
    else x
)
print(filmy11['genres_count_cap'].describe())

######################
# prod_comp_count
######################
filmy11.prod_comp_count = filmy11.prod_comp_count.replace(0, np.nan)
print(round(filmy11['prod_comp_count'].describe()))
Q1_prod_comp_count = filmy11['prod_comp_count'].describe()['25%']
Q3_prod_comp_count = filmy11['prod_comp_count'].describe()['75%']
IQR_prod_comp_count = Q3_prod_comp_count - Q1_prod_comp_count
filmy11['prod_comp_count_cap'] = filmy11['prod_comp_count'].apply(
    lambda x: Q3_prod_comp_count + (1.5 * IQR_prod_comp_count)
    if x > Q3_prod_comp_count + (1.5 * IQR_prod_comp_count)
    else x
)
print(filmy11['prod_comp_count_cap'].describe())

######################
# prod_comp_count
######################
filmy11.prod_countr_count = filmy11.prod_countr_count.replace(0, np.nan)
print(round(filmy11['prod_countr_count'].describe()))
Q1_prod_countr_count = filmy11['prod_countr_count'].describe()['25%']
Q3_prod_countr_count = filmy11['prod_countr_count'].describe()['75%']
IQR_prod_countr_count = Q3_prod_countr_count - Q1_prod_countr_count
filmy11['prod_countr_count_cap'] = filmy11['prod_countr_count'].apply(
    lambda x: round(Q3_prod_countr_count + (1.5 * IQR_prod_countr_count))
    if x > Q3_prod_countr_count + (1.5 * IQR_prod_countr_count)
    else x
)
print(filmy11['prod_countr_count_cap'].describe())

######################
# spok_lang_count
######################
filmy11.spok_lang_count = filmy11.spok_lang_count.replace(0, np.nan)
print(round(filmy11['spok_lang_count'].describe()))
Q1_spok_lang_count = filmy11['spok_lang_count'].describe()['25%']
Q3_spok_lang_count = filmy11['spok_lang_count'].describe()['75%']
IQR_spok_lang_count = Q3_spok_lang_count - Q1_spok_lang_count
filmy11['spok_lang_count_cap'] = filmy11['spok_lang_count'].apply(
    lambda x: round(Q3_spok_lang_count + (1.5 * IQR_spok_lang_count))
    if x > Q3_spok_lang_count + (1.5 * IQR_spok_lang_count)
    else x
)
print(filmy11['spok_lang_count_cap'].describe())

######################
# cast_size
######################
filmy11.cast_size = filmy11.cast_size.replace(0, np.nan)
round(filmy11['cast_size'].describe())
Q1_cast_size = filmy11['cast_size'].describe()['25%']
Q3_cast_size = filmy11['cast_size'].describe()['75%']
IQR_cast_size = Q3_cast_size - Q1_cast_size
filmy11['cast_size_cap'] = filmy11['cast_size'].apply(
    lambda x: round(Q3_cast_size + (1.5 * IQR_cast_size))
    if x > Q3_cast_size + (1.5 * IQR_cast_size)
    else x
)
filmy11['cast_size_cap'].describe()

######################
# crew_size
######################
filmy11.crew_size = filmy11.crew_size.replace(0, np.nan)
round(filmy11['crew_size'].describe())
Q1_crew_size = filmy11['crew_size'].describe()['25%']
Q3_crew_size = filmy11['crew_size'].describe()['75%']
IQR_crew_size = Q3_crew_size - Q1_crew_size
filmy11['crew_size_cap'] = filmy11['crew_size'].apply(
    lambda x: round(Q3_crew_size + (1.5 * IQR_crew_size))
    if x > Q3_crew_size + (1.5 * IQR_crew_size)
    else x
)
filmy11['crew_size_cap'].describe()


######################
# Ostateczna baza (bo pojawiło się trochę nulli na zmiennych dotyczących m. in. liczby gatunków, rozmiaru ekipy i obsady itp.)
######################
filmy2 = filmy11[filmy11.budget.notnull() & filmy11.revenue.notnull() & filmy11.runtime.notnull() & filmy11.vote_average.notnull() &
                 filmy11.vote_count.notnull() & filmy11.popularity.notnull() & filmy11.genres_count.notnull() & filmy11.prod_comp_count.notnull() &
                 filmy11.prod_countr_count.notnull() & filmy11.spok_lang_count.notnull() & filmy11.cast_size.notnull() & filmy11.crew_size.notnull()]
print(filmy2.shape)


##############################################################################################################################
# KOD 7 - Statystyki pozycyjne i opisowe
##############################################################################################################################
df_stat = round(filmy2[['budget', 'budget_cap', 'revenue', 'revenue_cap', 'runtime', 'runtime_cap', 'vote_average',
                        'vote_average_cap', 'vote_count', 'vote_count_cap', 'popularity', 'popularity_cap']].describe(),
                1)
df_stat_cap = round(filmy2[['genres_count', 'genres_count_cap', 'prod_comp_count', 'prod_comp_count_cap',
                            'prod_countr_count', 'prod_countr_count_cap', 'spok_lang_count', 'spok_lang_count_cap',
                            'cast_size', 'cast_size_cap', 'crew_size', 'crew_size_cap']].describe(), 1)

df_stat.to_csv(r'data\df_stat.csv', index = False, sep = ';', decimal= ',')
df_stat_cap.to_csv(r'data\df_stat_cap.csv', index = False, sep = ';', decimal= ',')


##############################################################################################################################
# KOD 8 - zmienne kategoryczne
##############################################################################################################################

######################
# Data premiery
######################
miesiace = [
    'Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze',
    'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru'
]
dni = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Nie']
dni_nr = [1, 2, 3, 4, 5, 6, 7]
filmy2['rd_year'] = pd.to_datetime(
    filmy2['release_date'],
    errors='coerce').apply(
    lambda x:
    str(x).split('-')[0]
    if x != np.nan
    else np.nan
)
filmy2['rd_day'] = filmy2['release_date'].apply(wez_dzien)
filmy2['rd_weekday'] = filmy2['release_date'].apply(wez_dzien_tyg)
filmy2['rd_weekday_num'] = filmy2['release_date'].apply(wez_nr_dnia_tyg)
filmy2['rd_month_num'] = filmy2['release_date'].apply(wez_miesiac)
filmy2['rd_month'] = filmy2['release_date'].apply(wez_nazwe_miesiaca)

######################
# Original language
######################
print(filmy2['original_language'].astype('str').apply(
    lambda x: len(x)).value_counts()
      )

##############################################################################################################################
# KOD 9 - Zmienne flagowe na podstawie braków danych
##############################################################################################################################

#######################
# belongs_to_collection
#######################
filmy2['belongs_to_collection'] = filmy2['belongs_to_collection'].fillna("{'name': 'None'}")
filmy2['belongs_to_collection'] = filmy2['belongs_to_collection'].apply(ast.literal_eval)
filmy2['belongs_to_collection'] = filmy2['belongs_to_collection'].apply(
    lambda x: x['name']
    if isinstance(x, dict)
    else np.nan
)
filmy2['if_franchise'] = filmy2['belongs_to_collection'].apply(
    lambda x: 0
    if x == 'None'
    else 1
)
filmy3 = filmy2.drop('belongs_to_collection', axis=1)
filmy3['if_franchise'].value_counts()

#######################
# homepage
#######################
filmy3['homepage'] = filmy3['homepage'].fillna(0)
filmy3['if_homepage'] = filmy3['homepage'].apply(
    lambda x: 0
    if x == 0
    else 1
)
filmy4 = filmy3.drop('homepage', axis=1)
filmy4['if_homepage'].value_counts()



##############################################################################################################################
# KOD 10 - Zmienne flagowe na podstawie gatunków filmu oraz utworzenie pomocniczej tabeli z rozbiciem na pojedyncze gatunki
##############################################################################################################################
s_genres = filmy4.apply(lambda x: pd.Series(x['genres']),
                        axis=1).stack().reset_index(level=1,
                                                    drop=True)
s_genres.name = 'genre'
df_gatunki = filmy4.drop('genres', axis=1).join(s_genres)
gatunki_hist = pd.DataFrame(df_gatunki['genre'].value_counts()).reset_index()
gatunki_hist.columns = ['genre', 'movies']
gatunki_hist = gatunki_hist.merge(df_gatunki.groupby('genre')['budget'].sum().sort_values(ascending=False), on='genre')
gatunki_hist = gatunki_hist.merge(df_gatunki.groupby('genre')['budget'].mean().sort_values(ascending=False), on='genre')
gatunki_hist = gatunki_hist.merge(df_gatunki.groupby('genre')['revenue'].sum().sort_values(ascending=False), on='genre')
gatunki_hist = gatunki_hist.merge(df_gatunki.groupby('genre')['revenue'].mean().sort_values(ascending=False),
                                  on='genre')
gatunki_hist.columns = ['genre', 'movies', 'budget_sum', 'budget_mean', 'revenue_sum', 'revenue_mean']
print('filmy4:')
print(filmy4.shape)
print('df_gatunki:')
print(df_gatunki.shape)
print('gatunki_hist:')
print(gatunki_hist.shape)

plt.figure(figsize=(18, 8))
g = sns.barplot(x='genre', y='movies', data=gatunki_hist.head(9))
for index, row in gatunki_hist.head(9).iterrows():
    g.text(row.name, row.movies + 20, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Genre', fontsize=30)
plt.ylabel('Movies', fontsize=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("gatunki_hist.pdf", bbox_inches="tight")
plt.show()

gatunki = gatunki_hist['genre'].head(10)
for gatunek in gatunki:
    filmy4['is_' + str(gatunek)] = filmy4['genres'].apply(
        lambda x:
        1 if gatunek in x
        else 0)

filmy5 = filmy4.drop('genres', axis=1)


##############################################################################################################################
# KOD 11 - Zmienne flagowe na podstawie daty premiery filmu
##############################################################################################################################

###############################
# is_(friday_thursday_wednesday)
###############################
rd_weekday_hist = pd.DataFrame(filmy5['rd_weekday'].value_counts()).reset_index()
rd_weekday_hist.columns = ['rd_weekday', 'movies']
rd_weekday_hist = rd_weekday_hist.merge(filmy5.groupby('rd_weekday')['budget'].sum().sort_values(ascending=False),
                                        on='rd_weekday')
rd_weekday_hist = rd_weekday_hist.merge(filmy5.groupby('rd_weekday')['budget'].mean().sort_values(ascending=False),
                                        on='rd_weekday')
rd_weekday_hist = rd_weekday_hist.merge(filmy5.groupby('rd_weekday')['revenue'].sum().sort_values(ascending=False),
                                        on='rd_weekday')
rd_weekday_hist = rd_weekday_hist.merge(filmy5.groupby('rd_weekday')['revenue'].mean().sort_values(ascending=False),
                                        on='rd_weekday')
rd_weekday_hist.columns = ['weekday', 'movies', 'budget_sum', 'budget_mean', 'revenue_sum', 'revenue_mean']

plt.figure(figsize=(18, 8))
g = sns.barplot(x='weekday', y='movies', data=rd_weekday_hist)
for index, row in rd_weekday_hist.iterrows():
    g.text(row.name, row.movies + 20, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Weekday', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("weekday_hist.pdf", bbox_inches="tight")
plt.show()

rd_weekday_hist['budget_sum'] = rd_weekday_hist['budget_sum'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))
rd_weekday_hist['budget_mean'] = rd_weekday_hist['budget_mean'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))
rd_weekday_hist['revenue_sum'] = rd_weekday_hist['revenue_sum'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))
rd_weekday_hist['revenue_mean'] = rd_weekday_hist['revenue_mean'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))

filmy6 = filmy5
filmy6['is_friday'] = filmy6['rd_weekday_num'].apply(
    lambda x: 1
    if x == 5.0
    else 0
)
print(filmy6['is_friday'].value_counts())
filmy6['is_thursday'] = filmy6['rd_weekday_num'].apply(
    lambda x: 1
    if x == 4.0
    else 0
)
print(filmy6['is_thursday'].value_counts())
filmy6['is_wednesday'] = filmy6['rd_weekday_num'].apply(
    lambda x: 1
    if x == 3.0
    else 0
)
print(filmy6['is_wednesday'].value_counts())

###############################
# is_dmp_month
###############################
rd_month_hist = pd.DataFrame(filmy6['rd_month'].value_counts()).reset_index()
rd_month_hist.columns = ['rd_month', 'movies']
rd_month_hist = rd_month_hist.merge(filmy6.groupby('rd_month')['budget'].sum().sort_values(ascending=False),
                                    on='rd_month')
rd_month_hist = rd_month_hist.merge(filmy6.groupby('rd_month')['budget'].mean().sort_values(ascending=False),
                                    on='rd_month')
rd_month_hist = rd_month_hist.merge(filmy6.groupby('rd_month')['revenue'].sum().sort_values(ascending=False),
                                    on='rd_month')
rd_month_hist = rd_month_hist.merge(filmy6.groupby('rd_month')['revenue'].mean().sort_values(ascending=False),
                                    on='rd_month')
rd_month_hist.columns = ['rd_month', 'movies', 'budget_sum', 'budget_mean', 'revenue_sum', 'revenue_mean']

plt.figure(figsize=(18, 8))
g = sns.barplot(x='rd_month', y='movies', data=rd_month_hist)
for index, row in rd_month_hist.iterrows():
    g.text(row.name, row.movies + 5, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Month', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("month_hist.pdf", bbox_inches="tight")
plt.show()

rd_month_hist['budget_sum'] = rd_month_hist['budget_sum'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))
rd_month_hist['budget_mean'] = rd_month_hist['budget_mean'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))
rd_month_hist['revenue_sum'] = rd_month_hist['revenue_sum'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))
rd_month_hist['revenue_mean'] = rd_month_hist['revenue_mean'].apply(lambda x: '{:,}'.format(trunc(x)).replace(',', ' '))

print(filmy6['rd_month_num'].value_counts())

filmy7 = filmy6
filmy7['is_dmp_month'] = filmy7['rd_month_num'].apply(
    lambda x: 1
    if (x == 1.0 or x == 9.0)
    else 0
)
filmy7['is_dmp_month'].value_counts()


##############################################################################################################################
# KOD 12 - Zmienne flagowe na podstawie języków filmu oraz na podstawie krajów produkcji
##############################################################################################################################

###############################
# is_english_spoken
###############################
s_jezyki_obce = filmy7.apply(
    lambda x: pd.Series(x['spoken_languages']),
    axis=1
).stack().reset_index(
    level=1,
    drop=True
)
s_jezyki_obce.name = 'spoken_language'
df_spok_lang = filmy7.drop(
    'spoken_languages',
    axis=1
).join(s_jezyki_obce)
jezyki_obce_hist = pd.DataFrame(
    df_spok_lang['spoken_language'].value_counts()
).reset_index()
jezyki_obce_hist.columns = ['spoken_language', 'movies']
jezyki_obce_hist = jezyki_obce_hist.merge(
    df_spok_lang.groupby('spoken_language')['budget'].sum().sort_values(ascending=False), on='spoken_language')
jezyki_obce_hist = jezyki_obce_hist.merge(
    df_spok_lang.groupby('spoken_language')['budget'].mean().sort_values(ascending=False), on='spoken_language')
jezyki_obce_hist.columns = ['spoken_language', 'movies', 'budget_sum', 'budget_mean']

print('filmy7:')
print(filmy7.shape)

print('df_spok_lang:')
print(df_spok_lang.shape)

print('jezyki_obce_hist:')
print(jezyki_obce_hist.shape)

plt.figure(figsize=(18, 8))
g = sns.barplot(x='spoken_language',
                y='movies',
                data=jezyki_obce_hist.head(6))
for index, row in jezyki_obce_hist.head(6).iterrows():
    g.text(row.name, row.movies + 50, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Spoken Language', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("spok_lang_hist.pdf", bbox_inches="tight")
plt.show()

filmy7['is_english_spoken'] = filmy7['spoken_languages'].apply(
    lambda x:
    1 if 'English' in x
    else 0
)

filmy7 = filmy7.drop('spoken_languages', axis=1)
print(filmy7['is_english_spoken'].value_counts())

###############################
# Is English original
###############################

#################
# Dane do wykresu
#################
org_lang_hist = pd.DataFrame(
    filmy7['original_language'].value_counts()
).reset_index()
org_lang_hist.columns = ['original_language', 'movies']
X = org_lang_hist['movies'].head(10)
XX = org_lang_hist['original_language'].head(10)
XX2 = pd.Series(X.values,
                index=XX
                )

###############################################################
# Przygotowanie wykresu (dwie części: górna - ax1 i dolna - ax2)
###############################################################
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                               figsize=(10, 5))

ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x', which='both', bottom=False)

ax2.spines['top'].set_visible(False)
ax2.set_ylim(0, 100)

ax1.set_ylim(4600, 4645)
ax1.set_yticks(np.arange(4600, 4651, 20))

g1 = XX2.plot(ax=ax1, kind='bar')
g2 = XX2.plot(ax=ax2, kind='bar')

#################################
# Podpisy nad słupkami histogramu
#################################
for index, row in org_lang_hist.head(1).iterrows():
    g1.text(row.name, row.movies + 1, row.movies,
            color='black', ha="center", fontsize=8
            )

org_lang_list = org_lang_hist['original_language']
loop_list = org_lang_hist[org_lang_list != 'en'].head(9).iterrows()
for index, row in loop_list:
    g2.text(row.name, row.movies + 1, row.movies,
            color='black', ha="center", fontsize=8
            )

########################
# Obrót etykiet na osi X
########################
for tick in ax2.get_xticklabels():
    tick.set_rotation(0)

#######################################
# Kreski uwydatniające przerwę na osi Y
#######################################
d1 = .007
d2 = .025
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((- d1, + d1), (- d1, + d1), **kwargs)
ax1.plot((0.05 - d2, 0.05 + d2), (- d2, + d2), **kwargs)
ax1.plot((1 - d1, 1 + d1), (-d1, +d1), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((- d1, + d1), (1 - d1, 1 + d1), **kwargs)
ax2.plot((0.05 - d2, 0.05 + d2), (1 - d2, 1 + d2), **kwargs)
ax2.plot((1 - d1, 1 + d1), (1 - d1, 1 + d1), **kwargs)

#######################################
# Nazwanie osi X i wyświetlenie wykresu
#######################################
plt.xlabel('Original Language', fontsize=15)
plt.tight_layout()
plt.savefig("org_language_hist_uciete.pdf", bbox_inches="tight")
plt.show()

##################################
# Utworzenie zmiennej wskaźnikowej
##################################
filmy7['is_english_original'] = filmy7['original_language'].apply(
    lambda x: 1
    if x == 'en'
    else 0
)
filmy7['is_english_original'].value_counts()

###############################
# is_usa_prod
###############################
s_kraje_prod = filmy7.apply(
    lambda x: pd.Series(x['production_countries']),
    axis=1
).stack().reset_index(
    level=1,
    drop=True
)
s_kraje_prod.name = 'production_country'
df_kraje_prod = filmy7.drop(
    'production_countries',
    axis=1
).join(s_kraje_prod)
kraje_prod_hist = pd.DataFrame(
    df_kraje_prod['production_country'].value_counts()
).reset_index()
kraje_prod_hist.columns = ['production_country', 'movies']
kraje_prod_hist = kraje_prod_hist.merge(
    df_kraje_prod.groupby('production_country')['budget'].sum().sort_values(ascending=False), on='production_country')
kraje_prod_hist = kraje_prod_hist.merge(
    df_kraje_prod.groupby('production_country')['budget'].mean().sort_values(ascending=False), on='production_country')
kraje_prod_hist.columns = ['production_country', 'movies', 'budget_sum', 'budget_mean']

print('filmy7:')
print(filmy7.shape)

print('df_kraje_prod:')
print(df_kraje_prod.shape)

print('kraje_prod_hist:')
print(kraje_prod_hist.shape)

plt.figure(figsize=(18, 8))
g = sns.barplot(x='production_country',
                y='movies',
                data=kraje_prod_hist.head(6))
for index, row in kraje_prod_hist.head(6).iterrows():
    g.text(row.name, row.movies + 50, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Production Country', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=15)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("prod_countr_hist.pdf", bbox_inches="tight")
plt.show()

kraje_prod = kraje_prod_hist['production_country'].head(1)

for kraj in kraje_prod:
    filmy7['is_' + str(kraj)] = filmy7['production_countries'].apply(
        lambda x:
        1 if kraj in x
        else 0)

filmy7['is_usa_prod'] = filmy7['is_United States of America']
filmy7 = filmy7.drop(['is_United States of America'], axis=1)
filmy8 = filmy7.drop('production_countries', axis=1)
print(filmy8['is_usa_prod'].value_counts(), '\n')

##############################################################################################################################
# KOD 13 - Zmienne flagowe na podstawie obsady filmu oraz utworzenie pomocniczej tabeli z rozbiciem na pojedynczych aktorów
##############################################################################################################################

###############################
# Cast
###############################
s_cast = filmy8.apply(lambda x: pd.Series(x['cast']), axis=1).stack().reset_index(level=1, drop=True)
s_cast.name = 'actor'
cast_df = filmy8.drop('cast', axis=1).join(s_cast)
obsada_hist = pd.DataFrame(cast_df['actor'].value_counts()).reset_index()
obsada_hist.columns = ['actor', 'movies']
obsada_hist = obsada_hist.merge(cast_df.groupby('actor')['budget'].sum().sort_values(ascending=False), on='actor')
obsada_hist = obsada_hist.merge(cast_df.groupby('actor')['budget'].mean().sort_values(ascending=False), on='actor')
obsada_hist.columns = ['actor', 'movies', 'budget_sum', 'budget_mean']

print('filmy8:')
print(filmy8.shape)

print('cast_df:')
print(cast_df.shape)

print('obsada_hist:')
print(obsada_hist.shape)

plt.figure(figsize=(18, 8))
g = sns.barplot(x='actor', y='movies', data=obsada_hist.head(7))
for index, row in obsada_hist.head(7).iterrows():
    g.text(row.name, row.movies + 0.5, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Actor', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=15)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("actors_hist.pdf", bbox_inches="tight")
plt.show()

aktorzy = list(
    obsada_hist[
        obsada_hist['movies'] > np.percentile(obsada_hist.movies, 99)
        ]['actor']
)
filmy8['actor_top_freq'] = filmy8['cast'].apply(
    lambda x: list_el_in_oth_list(
        aktorzy,
        x
    )
)
filmy8['actor_top_freq'] = filmy8['actor_top_freq'].astype('int')
print(filmy8['actor_top_freq'].value_counts())

##############################################################################################################################
# KOD 14 - Sprawdzenie liczności poszczególnych zawodów w ekipach filmowych
##############################################################################################################################

filmy8['crew_job'] = filmy8['crew'].apply(get_job)
print('filmy8:')
print(filmy8.shape)

s_crew_job = filmy8.apply(lambda x: pd.Series(x['crew_job']), axis=1).stack().reset_index(level=1, drop=True)
s_crew_job.name = 'crew_job'
df_crew_job = filmy8.drop('crew_job', axis=1).join(s_crew_job)
crew_job_hist = pd.DataFrame(df_crew_job['crew_job'].value_counts()).reset_index()
crew_job_hist.columns = ['crew_job', 'movies']
crew_job_hist = crew_job_hist.merge(df_crew_job.groupby('crew_job')['budget'].sum().sort_values(ascending=False),
                                    on='crew_job')
crew_job_hist = crew_job_hist.merge(df_crew_job.groupby('crew_job')['budget'].mean().sort_values(ascending=False),
                                    on='crew_job')
crew_job_hist = crew_job_hist.merge(df_crew_job.groupby('crew_job')['revenue'].sum().sort_values(ascending=False),
                                    on='crew_job')
crew_job_hist = crew_job_hist.merge(df_crew_job.groupby('crew_job')['revenue'].mean().sort_values(ascending=False),
                                    on='crew_job')
crew_job_hist.columns = ['crew_job', 'movies', 'budget_sum', 'budget_mean', 'revenue_sum', 'revenue_mean']

print('df_crew_job:')
print(df_crew_job.shape)

print('crew_job_hist:')
print(crew_job_hist.shape)

plt.figure(figsize=(18, 8))
g = sns.barplot(x='crew_job', y='movies', data=crew_job_hist.head(7))
for index, row in crew_job_hist.head(7).iterrows():
    g.text(row.name, row.movies + 50, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Crew job', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=18)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("crew_job_hist.pdf", bbox_inches="tight")
plt.show()

##############################################################################################################################
# KOD 15 - Zmienne flagowe na podstawie ekipy filmowej (a dokładniej: na podstawie producentów, reżyserów i scenarzystów) oraz
#          utworzenie pomocniczych tabel z rozbiciem na pojedynczych producentów, reżyserów oraz scenarzystów
##############################################################################################################################

##################################################################################
# Wyciągnięcie konkretnych zawodów filmowych (4 - producent, reżyser, scenarzysta)
##################################################################################
filmy8['producer'] = filmy8['crew'].apply(get_producer)
filmy8['director'] = filmy8['crew'].apply(get_director)
filmy8['screenplay'] = filmy8['crew'].apply(get_screenplay)

s_producer = filmy8.apply(lambda x: pd.Series(x['producer']), axis=1).stack().reset_index(level=1, drop=True)
s_producer.name = 'producer'
df_producer = filmy8.drop('producer', axis=1).join(s_producer)
producer_hist = pd.DataFrame(df_producer['producer'].value_counts()).reset_index()
producer_hist.columns = ['producer', 'movies']
producer_hist = producer_hist.merge(df_producer.groupby('producer')['budget'].sum().sort_values(ascending=False),
                                    on='producer')
producer_hist = producer_hist.merge(df_producer.groupby('producer')['budget'].mean().sort_values(ascending=False),
                                    on='producer')
producer_hist.columns = ['producer', 'movies', 'budget_sum', 'budget_mean']

print('df_producer:')
print(df_producer.shape)

print('producer_hist:')
print(producer_hist.shape)

s_director = filmy8.apply(lambda x: pd.Series(x['director']), axis=1).stack().reset_index(level=1, drop=True)
s_director.name = 'director'
df_director = filmy8.drop('director', axis=1).join(s_director)
director_hist = pd.DataFrame(df_director['director'].value_counts()).reset_index()
director_hist.columns = ['director', 'movies']
director_hist = director_hist.merge(df_director.groupby('director')['budget'].sum().sort_values(ascending=False),
                                    on='director')
director_hist = director_hist.merge(df_director.groupby('director')['budget'].mean().sort_values(ascending=False),
                                    on='director')
director_hist.columns = ['director', 'movies', 'budget_sum', 'budget_mean']

print('df_director:')
print(df_director.shape)

print('director_hist:')
print(director_hist.shape)

s_screenplay = filmy8.apply(lambda x: pd.Series(x['screenplay']), axis=1).stack().reset_index(level=1, drop=True)
s_screenplay.name = 'screenplay'
df_screenplay = filmy8.drop('screenplay', axis=1).join(s_screenplay)
screen_hist = pd.DataFrame(df_screenplay['screenplay'].value_counts()).reset_index()
screen_hist.columns = ['screenplay', 'movies']
screen_hist = screen_hist.merge(df_screenplay.groupby('screenplay')['budget'].sum().sort_values(ascending=False),
                                on='screenplay')
screen_hist = screen_hist.merge(df_screenplay.groupby('screenplay')['budget'].mean().sort_values(ascending=False),
                                on='screenplay')
screen_hist.columns = ['screenplay', 'movies', 'budget_sum', 'budget_mean']

print('df_screenplay:')
print(df_screenplay.shape)

print('screen_hist:')
print(screen_hist.shape)

plt.figure(figsize=(18, 8))
g = sns.barplot(x='producer', y='movies', data=producer_hist.head(7))
for index, row in producer_hist.head(7).iterrows():
    g.text(row.name, row.movies + 0.5, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Producer', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=17)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("producer_hist.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(18, 8))
g = sns.barplot(x='director', y='movies', data=director_hist.head(7))
for index, row in director_hist.head(7).iterrows():
    g.text(row.name, row.movies + 0.2, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Director', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=17)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("director_hist.pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(18, 8))
g = sns.barplot(x='screenplay', y='movies', data=screen_hist.head(7))
for index, row in screen_hist.head(7).iterrows():
    g.text(row.name, row.movies + 0.15, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Screenplay', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=17)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("screenplay_hist.pdf", bbox_inches="tight")
plt.show()

###########################################################################################
# Zmienne flagowe na podstawie liczności producentów, reżyserów, scenarzystów i montażystów
###########################################################################################
producenci = list(
    producer_hist[
        producer_hist['movies'] > np.percentile(producer_hist.movies, 99)
        ]['producer']
)
filmy8['producer_top_freq'] = filmy8['producer'].apply(
    lambda x: list_el_in_oth_list(
        producenci,
        x
    )
)
filmy8['producer_top_freq'] = filmy8['producer_top_freq'].astype('int')
reżyserowie = list(
    director_hist[
        director_hist['movies'] > np.percentile(director_hist.movies, 95)
        ]['director']
)
filmy8['director_top_freq'] = filmy8['director'].apply(
    lambda x: list_el_in_oth_list(
        reżyserowie,
        x
    )
)
filmy8['director_top_freq'] = filmy8['director_top_freq'].astype('int')
scenarzysci = list(
    screen_hist[
        screen_hist['movies'] > np.percentile(screen_hist.movies, 95)
        ]['screenplay']
)
filmy8['screenplay_top_freq'] = filmy8['screenplay'].apply(
    lambda x: list_el_in_oth_list(
        scenarzysci,
        x
    )
)
filmy8['screenplay_top_freq'] = filmy8['screenplay_top_freq'].astype('int')

print(filmy8.actor_top_freq.value_counts(), '\n')
print(filmy8.producer_top_freq.value_counts(), '\n')
print(filmy8.director_top_freq.value_counts(), '\n')
print(filmy8.screenplay_top_freq.value_counts())

##############################################################################################################################
# KOD 16 - Dodanie zmiennych liczbowych dotyczących ekipy filmowej - liczba reżyserów, liczba scenarzystów, liczba producentów
#          oraz zastosowanie do nich "cap"
##############################################################################################################################

filmy8['producer_count'] = filmy8['producer'].apply(lambda x: len(x))
filmy8['director_count'] = filmy8['director'].apply(lambda x: len(x))
filmy8['screenplay_count'] = filmy8['screenplay'].apply(lambda x: len(x))
filmy9 = filmy8.drop(['cast', 'crew', 'crew_job', 'producer', 'director', 'screenplay'], axis=1)

# CAP
round(filmy9['producer_count'].describe())
Q1_producer_count = 1.0
Q3_producer_count = 5.0
IQR_producer_count = Q3_producer_count - Q1_producer_count
print(Q1_producer_count - (1.5 * IQR_producer_count))
print(Q3_producer_count + (1.5 * IQR_producer_count))
filmy9['producer_count_cap'] = filmy9['producer_count'].apply(
    lambda x: Q3_producer_count + (1.5 * IQR_producer_count)
    if x > Q3_producer_count + (1.5 * IQR_producer_count)
    else x
)
filmy9['producer_count_cap'].describe()

round(filmy9['director_count'].describe())
Q1_director_count = 1.0
Q3_director_count = 1.0
IQR_director_count = Q3_director_count - Q1_director_count
print(Q1_director_count - (1.5 * IQR_director_count))
print(Q3_director_count + (1.5 * IQR_director_count))
filmy9['director_count_cap'] = filmy9['director_count'].apply(
    lambda x: Q3_director_count + (1.5 * IQR_director_count)
    if x > Q3_director_count + (1.5 * IQR_director_count)
    else x
)
filmy9['director_count_cap'].describe()

round(filmy9['screenplay_count'].describe())
Q1_screenplay_count = 0.0
Q3_screenplay_count = 2.0
IQR_screenplay_count = Q3_screenplay_count - Q1_screenplay_count
print(Q1_screenplay_count - (1.5 * IQR_screenplay_count))
print(Q3_screenplay_count + (1.5 * IQR_screenplay_count))
filmy9['screenplay_count_cap'] = filmy9['screenplay_count'].apply(
    lambda x: Q3_screenplay_count + (1.5 * IQR_screenplay_count)
    if x > Q3_screenplay_count + (1.5 * IQR_screenplay_count)
    else x
)
filmy9['screenplay_count_cap'].describe()

##############################################################################################################################
# KOD 17 - Zmienne flagowe na podstawie koncernów filmowych oraz utworzenie pomocniczej tabeli z rozbiciem na pojedyncze firmy
##############################################################################################################################

s_prod_comp = filmy9.apply(
    lambda x: pd.Series(x['production_companies']),
    axis=1
).stack().reset_index(
    level=1,
    drop=True
)
s_prod_comp.name = 'production_company'
df_producenci = filmy9.drop(
    'production_companies',
    axis=1
).join(s_prod_comp)
producenci_hist = pd.DataFrame(
    df_producenci['production_company'].value_counts()
).reset_index()
producenci_hist.columns = ['production_company', 'movies']
producenci_hist = producenci_hist.merge(
    df_producenci.groupby('production_company')['budget'].sum().sort_values(ascending=False), on='production_company')
producenci_hist = producenci_hist.merge(
    df_producenci.groupby('production_company')['budget'].mean().sort_values(ascending=False), on='production_company')
producenci_hist = producenci_hist.merge(
    df_producenci.groupby('production_company')['revenue'].sum().sort_values(ascending=False), on='production_company')
producenci_hist = producenci_hist.merge(
    df_producenci.groupby('production_company')['revenue'].mean().sort_values(ascending=False), on='production_company')
producenci_hist.columns = ['production_company', 'movies', 'budget_sum', 'budget_mean', 'revenue_sum', 'revenue_mean']

print('filmy9:')
print(filmy9.shape)

print('df_producenci:')
print(df_producenci.shape)

print('producenci_hist:')
print(producenci_hist.shape)

producenci_hist['production_company'] = producenci_hist['production_company'].apply(
    lambda x: 'Twentieth Century Fox' if x == 'Twentieth Century Fox Film Corporation' else x)

plt.figure(figsize=(18, 8))
g = sns.barplot(x='production_company',
                y='movies',
                data=producenci_hist.head(5))
for index, row in producenci_hist.head(5).iterrows():
    g.text(row.name, row.movies + 5, row.movies, color='black', ha="center", fontsize=15)
plt.xlabel('Production Company', fontsize=25)
plt.ylabel('Movies', fontsize=25)
plt.xticks(size=17)
plt.yticks(size=20)
plt.tight_layout()
plt.savefig("prod_comp_hist.pdf", bbox_inches="tight")
plt.show()

# producenci_hist[['production_company', 'movies']].head(10).to_csv(r'data\producenci_hist.csv', index = False, sep = ';')

prod_comp_top = ['Warner Bros.', 'Universal Pictures', 'Paramount Pictures', 'Walt Disney Pictures',
                 'Columbia Pictures']

filmy9['is_prod_comp_top'] = filmy9['production_companies'].apply(
    lambda x: list_el_in_oth_list(
        prod_comp_top,
        x
    )
)

filmy9['is_prod_comp_top'] = filmy9['is_prod_comp_top'].astype('int')
filmy10 = filmy9.drop('production_companies', axis=1)
filmy10['is_prod_comp_top'].value_counts()

##############################################################################################################################
# KOD 18.1 - Chmury słów
##############################################################################################################################

###############################
# Title
###############################
title_corpus = ' '.join(filmy10.title)
overview_corpus = ' '.join(filmy10['overview'].astype(str))
tagline_corpus = ' '.join(filmy10[filmy10.tagline.notnull()]['tagline'].astype(str))
STOPWORDS.add('Movie')
STOPWORDS.add('one')
title_wordcloud = WordCloud(stopwords=STOPWORDS,
                            background_color='black',
                            height=2000,
                            width=4000,
                            prefer_horizontal=0.9,
                            mask=None,
                            min_font_size=4,
                            font_step=1,
                            max_words=200,
                            max_font_size=None,
                            normalize_plurals=True,
                            include_numbers=False,
                            min_word_length=3
                            ).generate(title_corpus)
plt.figure(figsize=(15,10))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.tight_layout()
plt.savefig ("title_wordcloud.pdf",bbox_inches="tight")
plt.show()

###############################
# Overview
###############################
overview_wordcloud = WordCloud(stopwords=STOPWORDS,
                            background_color='black',
                            height=2000,
                            width=4000,
                            prefer_horizontal=0.9,
                            mask=None,
                            min_font_size=4,
                            font_step=1,
                            max_words=200,
                            max_font_size=None,
                            normalize_plurals=True,
                            include_numbers=False,
                            min_word_length=3
                               ).generate(overview_corpus)
plt.figure(figsize=(16,8))
plt.imshow(overview_wordcloud)
plt.axis('off')
plt.tight_layout()
plt.savefig ("overview_wordcloud.pdf",bbox_inches="tight")
plt.show()

###############################
# Tagline
###############################
tagline_wordcloud = WordCloud(stopwords=STOPWORDS,
                            background_color='black',
                            height=2000,
                            width=4000,
                            prefer_horizontal=0.9,
                            mask=None,
                            min_font_size=4,
                            font_step=1,
                            max_words=200,
                            max_font_size=None,
                            normalize_plurals=True,
                            include_numbers=False,
                            min_word_length=3
                            ).generate(tagline_corpus)
plt.figure(figsize=(15,10))
plt.imshow(tagline_wordcloud)
plt.axis('off')
plt.tight_layout()
plt.savefig ("tagline_wordcloud.pdf",bbox_inches="tight")
plt.show()

##############################################################################################################################
# KOD 18.2 - Zmienne flagowe na podstawie częstości pojawiania się słów w tytułach i opisach filmu
##############################################################################################################################

###############################
# Title
###############################
filmy10['if_love_in_title'] = filmy10['title'].apply(
    lambda x: x.upper().find("LOVE")
    if isinstance(x, str)
    else np.nan
)
filmy10['if_love_in_title'] = filmy10['if_love_in_title'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy10['if_love_in_title'].value_counts())
filmy10['if_man_in_title'] = filmy10['title'].apply(
    lambda x: x.upper().find("MAN")
    if isinstance(x, str)
    else np.nan
)
filmy10['if_man_in_title'] = filmy10['if_man_in_title'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy10['if_man_in_title'].value_counts())
filmy10['if_day_in_title'] = filmy10['title'].apply(
    lambda x: x.upper().find("DAY")
    if isinstance(x, str)
    else np.nan
)
filmy10['if_day_in_title'] = filmy10['if_day_in_title'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy10['if_day_in_title'].value_counts())
filmy10['if_nigth_in_title'] = filmy10['title'].apply(
    lambda x: x.upper().find("NIGHT")
    if isinstance(x, str)
    else np.nan
)
filmy10['if_nigth_in_title'] = filmy10['if_nigth_in_title'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy10['if_nigth_in_title'].value_counts())
filmy10['ti_words_top_freq'] = (filmy10['if_love_in_title']
                                + filmy10['if_man_in_title']
                                + filmy10['if_nigth_in_title']
                                + filmy10['if_day_in_title']
                                )
filmy10['ti_words_top_freq'] = filmy10['ti_words_top_freq'].apply(
    lambda x: 1
    if x > 0
    else 0
)
filmy11 = filmy10.drop(['if_man_in_title',
                        'if_love_in_title',
                        'if_nigth_in_title',
                        'if_day_in_title'
                        ], axis=1
                       )
print(filmy11['ti_words_top_freq'].value_counts())

###############################
# Overview
###############################
filmy11['if_life_in_overview'] = filmy11['overview'].apply(
    lambda x: x.upper().find("LIFE")
    if isinstance(x, str)
    else np.nan
)
filmy11['if_life_in_overview'] = filmy11['if_life_in_overview'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy11['if_life_in_overview'].value_counts())

filmy11['if_find_in_overview'] = filmy11['overview'].apply(
    lambda x: x.upper().find("FIND")
    if isinstance(x, str)
    else np.nan
)
filmy11['if_find_in_overview'] = filmy11['if_find_in_overview'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy11['if_find_in_overview'].value_counts())

filmy11['if_world_in_overview'] = filmy11['overview'].apply(
    lambda x: x.upper().find("WORLD")
    if isinstance(x, str)
    else np.nan
)
filmy11['if_world_in_overview'] = filmy11['if_world_in_overview'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy11['if_world_in_overview'].value_counts())

filmy11['if_love_in_overview'] = filmy11['overview'].apply(
    lambda x: x.upper().find("LOVE")
    if isinstance(x, str)
    else np.nan
)
filmy11['if_love_in_overview'] = filmy11['if_love_in_overview'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy11['if_love_in_overview'].value_counts())

filmy11['ov_words_top_freq'] = (filmy11['if_life_in_overview']
                                + filmy11['if_find_in_overview']
                                + filmy11['if_world_in_overview']
                                + filmy11['if_love_in_overview']
                                )

filmy11['ov_words_top_freq'] = filmy11['ov_words_top_freq'].apply(
    lambda x: 1
    if x > 0
    else 0
)

filmy12 = filmy11.drop(['overview'
                           , 'if_life_in_overview'
                           , 'if_find_in_overview'
                           , 'if_world_in_overview'
                           , 'if_love_in_overview'
                        ]
                       , axis=1)

print(filmy12['ov_words_top_freq'].value_counts())

###############################
# Tagline
###############################
filmy12['tagline'] = filmy12['tagline'].fillna('No tagline')
filmy12['if_love_in_tagline'] = filmy12['tagline'].apply(
    lambda x: x.upper().find("LOVE")
    if isinstance(x, str)
    else np.nan
)
filmy12['if_love_in_tagline'] = filmy12['if_love_in_tagline'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy12['if_love_in_tagline'].value_counts())

filmy12['if_life_in_tagline'] = filmy12['tagline'].apply(
    lambda x: x.upper().find("LIFE")
    if isinstance(x, str)
    else np.nan
)
filmy12['if_life_in_tagline'] = filmy12['if_life_in_tagline'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy12['if_life_in_tagline'].value_counts())

filmy12['if_find_in_tagline'] = filmy12['tagline'].apply(
    lambda x: x.upper().find("FIND")
    if isinstance(x, str)
    else np.nan
)
filmy12['if_find_in_tagline'] = filmy12['if_find_in_tagline'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy12['if_find_in_tagline'].value_counts())

filmy12['if_world_in_tagline'] = filmy12['tagline'].apply(
    lambda x: x.upper().find("WORLD")
    if isinstance(x, str)
    else np.nan
)
filmy12['if_world_in_tagline'] = filmy12['if_world_in_tagline'].apply(
    lambda x: 1
    if x >= 0.0
    else 0
)
print(filmy12['if_world_in_tagline'].value_counts())

filmy12['ta_words_top_freq'] = (filmy12['if_love_in_tagline']
                                + filmy12['if_life_in_tagline']
                                + filmy12['if_find_in_tagline']
                                + filmy12['if_world_in_tagline']
                                )

filmy12['ta_words_top_freq'] = filmy12['ta_words_top_freq'].apply(
    lambda x: 1
    if x > 0
    else 0
)

filmy13 = filmy12.drop(['tagline'
                           , 'if_love_in_tagline'
                           , 'if_life_in_tagline'
                           , 'if_find_in_tagline'
                           , 'if_world_in_tagline'
                        ]
                       , axis=1)

print(filmy13['ta_words_top_freq'].value_counts())

###############################
# Overview and tagline
###############################
print(filmy13['ov_words_top_freq'].value_counts())
print(filmy13['ta_words_top_freq'].value_counts())
filmy13['ov_ta_words_top_freq'] = (filmy13['ov_words_top_freq'] +
                                   filmy13['ta_words_top_freq'])
filmy13['ov_ta_words_top_freq'] = filmy13['ov_ta_words_top_freq'].apply(
    lambda x: 1
    if x > 0
    else 0
)
filmy14 = filmy13.drop(['ov_words_top_freq', 'ta_words_top_freq'], axis=1)
print(filmy14['ov_ta_words_top_freq'].value_counts())


##############################################################################################################################
# Kod 19 - Usuwanie niepotrzebnych zmiennych
##############################################################################################################################
filmy14['adult'].value_counts()
filmy14['video'].value_counts()
filmy14['poster_path'].isnull().value_counts()
filmy14['status'].value_counts()
filmy14[filmy14.original_title != filmy14.title][['title', 'original_title']].head()
filmy15 = filmy14.drop(['adult', 'original_title', 'imdb_id', 'video', 'poster_path', 'status'], axis=1)


##############################################################################################################################
# KOD 20 - utworzenie zmiennej ROI dotyczących sukcesu filmu
##############################################################################################################################

###############################
# ROI
###############################
filmy15['ROI'] = (filmy15['revenue'] - filmy15['budget']) / filmy15['budget']
filmy15['ROI_cap'] = (filmy15['revenue_cap'] - filmy15['budget_cap']) / filmy15['budget_cap']
filmy15[filmy15['ROI'].isnull()].shape
filmy15[filmy15['ROI_cap'].isnull()].shape


##############################################################################################################################
# KOD 21 - uporządkowanie bazy
##############################################################################################################################
filmy15.info()
filmy16 = filmy15[
    ['id', 'title', 'original_language', 'release_date', 'rd_year', 'rd_month', 'rd_month_num', 'rd_day', 'rd_weekday',
     'rd_weekday_num', 'runtime', 'runtime_cap', 'ROI', 'ROI_cap', 'budget', 'budget_cap', 'revenue', 'revenue_cap',
     'vote_average', 'vote_average_cap', 'vote_count', 'vote_count_cap', 'popularity', 'popularity_cap', 'genres_count',
     'genres_count_cap', 'prod_comp_count', 'prod_comp_count_cap', 'prod_countr_count', 'prod_countr_count_cap',
     'spok_lang_count', 'spok_lang_count_cap', 'cast_size', 'cast_size_cap', 'crew_size', 'crew_size_cap',
     'producer_count', 'producer_count_cap', 'director_count', 'director_count_cap', 'screenplay_count',
     'screenplay_count_cap', 'if_franchise', 'if_homepage', 'is_dmp_month', 'is_friday', 'is_friday', 'is_thursday',
     'is_wednesday', 'is_english_spoken', 'is_Drama', 'is_Comedy', 'is_Thriller', 'is_Action', 'is_Romance',
     'is_Adventure', 'is_Crime', 'is_Science Fiction', 'is_Horror', 'is_Family', 'is_usa_prod', 'actor_top_freq',
     'producer_top_freq', 'director_top_freq', 'screenplay_top_freq', 'is_prod_comp_top', 'ti_words_top_freq',
     'ov_ta_words_top_freq']]


##############################################################################################################################
# KOD 22 - Zapisanie najważniejszych tabel
##############################################################################################################################
filmy16.to_csv(r'data\filmy_clean.csv', index = False)
# df_screenplay.to_csv(r'data\df_screenplay.csv', index = False)
cast_df.to_csv(r'data\cast_df.csv', index = False)
# df_director.to_csv(r'data\df_director.csv', index = False)
# df_kraje_prod.to_csv(r'data\df_kraje_prod.csv', index = False)
# df_spok_lang.to_csv(r'data\df_spok_lang.csv', index = False)
df_gatunki.to_csv(r'data\df_gatunki.csv', index = False)
# df_producer.to_csv(r'data\df_producer.csv', index = False)




































"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Rozdział 3: Analiza statystyczna
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

##############################################################################################################################
# KOD 23: Wczytanie danych
##############################################################################################################################
filmy_clean = pd.read_csv(r"data\filmy_clean.csv",
                          low_memory=False)
print(filmy_clean.shape)
df_gatunki = pd.read_csv(r"data\df_gatunki.csv",
                         low_memory=False)
print(df_gatunki.shape)
cast_df = pd.read_csv(r"data\cast_df.csv",
                      low_memory=False)
print(cast_df.shape)


##############################################################################################################################
# KOD 24: Testy normalności
##############################################################################################################################
numFeatures = ['runtime_cap', 'revenue_cap',
               'budget_cap', 'ROI_cap', 'vote_count_cap', 'vote_average_cap']
filmy_clean_ln = filmy_clean[numFeatures].apply(lambda x: np.log(x))
filmy_clean_sqrt = filmy_clean[numFeatures].apply(lambda x: np.sqrt(x))
filmy_clean_1_sqrt = filmy_clean[numFeatures].apply(lambda x: 1 / (np.sqrt(x)))
filmy_clean_stand = filmy_clean[numFeatures].apply(lambda x: (x - x.mean()) / x.std())
filmy_clean_ln_stand = filmy_clean_ln[numFeatures].apply(lambda x: (x - x.mean()) / x.std())
filmy_clean_sqrt_stand = filmy_clean_sqrt[numFeatures].apply(lambda x: (x - x.mean()) / x.std())
filmy_clean_1_sqrt_stand = filmy_clean_1_sqrt[numFeatures].apply(lambda x: (x - x.mean()) / x.std())

############################################
# Testy normalności bez transformacji danych
############################################
for feature in numFeatures:
    stat_sw, p_sw = stats.shapiro(filmy_clean[feature])
    stat_ks, p_ks = stats.kstest(filmy_clean[feature],
                                 'norm')
    print(feature)
    print('S-W stat: ', round(stat_sw, 2), 'p-value', round(p_sw, 2))
    print('K-S stat: ', round(stat_ks, 2), 'p-value', round(p_ks, 2))

############################################
# Testy normalności z transformacją ln(x)
############################################
for feature in numFeatures:
    stat_sw, p_sw = stats.shapiro(filmy_clean_ln[feature])
    stat_ks, p_ks = stats.kstest(filmy_clean_ln[feature],
                                 'norm')
    print(feature)
    print('S-W stat: ', round(stat_sw, 2), 'p-value', round(p_sw, 2))
    print('K-S stat: ', round(stat_ks, 2), 'p-value', round(p_ks, 2))

############################################
# Testy normalności z transformacją sqrt(x)
############################################
for feature in numFeatures:
    stat_sw, p_sw = stats.shapiro(filmy_clean_sqrt[feature])
    stat_ks, p_ks = stats.kstest(filmy_clean_sqrt[feature],
                                 'norm')
    print(feature)
    print('S-W stat: ', round(stat_sw, 2), 'p-value', round(p_sw, 2))
    print('K-S stat: ', round(stat_ks, 2), 'p-value', round(p_ks, 2))

#################################################
# Testy normalności z transformacją (1 / sqrt(x))
#################################################
for feature in numFeatures:
    stat_sw, p_sw = stats.shapiro(filmy_clean_1_sqrt[feature])
    stat_ks, p_ks = stats.kstest(filmy_clean_1_sqrt[feature],
                                 'norm')
    print(feature)
    print('S-W stat: ', round(stat_sw, 2), 'p-value', round(p_sw, 2))
    print('K-S stat: ', round(stat_ks, 2), 'p-value', round(p_ks, 2))

baza_testy_stat = filmy_clean[['is_prod_comp_top', 'is_dmp_month']]
baza_testy_stat['revenue_cap'] = filmy_clean_stand['revenue_cap']



##############################################################################################################################
# KOD 25: Testy Manna-Whitneya - Dump Months
##############################################################################################################################

#################################################
# Test Manna-Whitneya - Dump Months
#################################################
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
sns.boxplot(x='is_dmp_month', y='revenue_cap', data=baza_testy_stat, palette="muted", ax=ax)
plt.xlabel('Is Dump Month', fontsize=21)
plt.ylabel('Revenue', fontsize=21)
plt.xticks(size=15)
plt.yticks(size=15)
plt.tight_layout()
plt.savefig("MW_dmp.pdf", bbox_inches="tight")
plt.show()

dump_months0 = baza_testy_stat[baza_testy_stat['is_dmp_month'] == 0]['revenue_cap']
dump_months1 = baza_testy_stat[baza_testy_stat['is_dmp_month'] == 1]['revenue_cap']

U_dmp, p_dmp = mannwhitneyu(dump_months0, dump_months1)
print(round(U_dmp, 2))
print(round(p_dmp, 2))


#################################################
# Test Manna-Whitneya - Top production companies
#################################################
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
sns.boxplot(x='is_prod_comp_top', y='revenue_cap', data=baza_testy_stat, palette="muted", ax=ax)
plt.xlabel('Is Prod Comp Top', fontsize=21)
plt.ylabel('Revenue', fontsize=21)
plt.xticks(size=15)
plt.yticks(size=15)
plt.tight_layout()
plt.savefig("MW_top.pdf", bbox_inches="tight")
plt.show()

prod_comp_top0 = baza_testy_stat[baza_testy_stat['is_prod_comp_top'] == 0]['revenue_cap']
prod_comp_top1 = baza_testy_stat[baza_testy_stat['is_prod_comp_top'] == 1]['revenue_cap']

U_top, p_top = mannwhitneyu(prod_comp_top0, prod_comp_top1)
print(round(U_top, 2))
print(round(p_top, 2))



##############################################################################################################################
# KOD 26: Test Kruskala-Wallisa
##############################################################################################################################

numFeatures2 = ['runtime_cap', 'revenue_cap',
                'budget_cap', 'vote_count_cap', 'vote_average_cap']
genres = ['Drama', 'Action', 'Comedy', 'Romance', 'Thriller']
df_gatunki_mod = df_gatunki[df_gatunki['genre'].apply(lambda x: x if x in genres else np.nan).notnull()]
df_gatunki_mod[numFeatures2] = df_gatunki_mod[numFeatures2].apply(lambda x: (x - x.mean()) / x.std())

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
sns.boxplot(x='genre', y='budget_cap', data=df_gatunki_mod, palette="muted", ax=ax)
plt.xlabel('Is Prod Comp Top', fontsize=21)
plt.ylabel('Budget', fontsize=21)
plt.xticks(size=15)
plt.yticks(size=15)
plt.tight_layout()
plt.savefig("KW_genre.pdf", bbox_inches="tight")
plt.show()

#################################
# Grupowanie budżetu po gatunkach
#################################
df_gatunki_KW = df_gatunki[['genre', 'budget_cap']]
groups = df_gatunki_KW.groupby('genre').groups

################################
# Wyciąganie poszczególnych grup
################################
drama = df_gatunki['budget_cap'][groups['Drama']]
action = df_gatunki['budget_cap'][groups['Action']]
comedy = df_gatunki['budget_cap'][groups['Comedy']]
romance = df_gatunki['budget_cap'][groups['Romance']]
thriller = df_gatunki['budget_cap'][groups['Thriller']]

################################
# K-W test
################################
stats.kruskal(drama, action, comedy, romance, thriller)

################################
# Dunn test (post-hoc)
################################
genres = ['Drama', 'Action', 'Comedy', 'Romance', 'Thriller']
df_gatunki_mod = df_gatunki[df_gatunki['genre'].apply(lambda x: x if x in genres else np.nan).notnull()]
sp.posthoc_dunn(df_gatunki, 'budget_cap', 'genre', 'bonferroni')
round(sp.posthoc_dunn(df_gatunki_mod, 'budget_cap', 'genre', 'bonferroni'), 2)


##############################################################################################################################
# KOD 27: Testy Chi-kwadrat na niezależność zmiennych
##############################################################################################################################

################################################################
# gatunek / aktor
################################################################

chi_genres_actor = df_gatunki[['id', 'genre']].merge(cast_df[['id', 'actor']], on='id')
print('chi_genres_actor: ', chi_genres_actor.shape)
chi_genres_actor_dedup = chi_genres_actor.drop_duplicates(subset=['id', 'genre', 'actor'])
print('chi_genres_actor_dedup: ', chi_genres_actor_dedup.shape)
df_actors_mod = pd.DataFrame(cast_df['actor'].value_counts()).reset_index()
df_actors_mod.columns = ['actor', 'freq']
list_act = list(df_actors_mod[df_actors_mod['freq'] > 32]['actor'])
chi_genres_actor_dedup['actor_mod'] = chi_genres_actor_dedup['actor'].apply(lambda x: x if x in list_act else np.nan)
df_genres_mod = pd.DataFrame(df_gatunki['genre'].value_counts()).reset_index()
df_genres_mod.columns = ['genre', 'freq']
list_genre = list(df_genres_mod['genre'].head(5))
chi_genres_actor_dedup['genre_mod'] = chi_genres_actor_dedup['genre'].apply(lambda x: x if x in list_genre else np.nan)
chi_genres_actor_dedup[chi_genres_actor_dedup['genre_mod'].notnull() & chi_genres_actor_dedup['actor_mod'].notnull()][
    'id'].value_counts()
chi_table_genres_actor = pd.crosstab(index=chi_genres_actor_dedup['actor_mod'],
                                     columns=chi_genres_actor_dedup['genre_mod'], margins=True)
gat_akt_col = chi_table_genres_actor.columns
avg_cell_count_under_5 = []
zeros_count = []

stat_genres_actor, p_genres_actor, dof_genres_actor, expected_genres_actor = stats.chi2_contingency(
    chi_table_genres_actor, correction=False)
sum(expected_genres_actor == 0)  # ok

chi_table_genres_actor_ex = pd.DataFrame(expected_genres_actor)
chi_table_genres_actor_ex.columns = chi_table_genres_actor.columns
gat_akt_col_ex = chi_table_genres_actor_ex.columns

avg_ex_cell_count_under_5 = []
zeros_count = []
for gen in gat_akt_col_ex:
    df = pd.DataFrame(chi_table_genres_actor_ex[gen].value_counts()).reset_index()
    df.columns = ['cell_count', 'cell_freq']
    avg_ex_cell_count_under_5.append(round(sum(df[df['cell_count'] < 5]['cell_freq']) / sum(df['cell_freq']) * 100))
    zeros_count.append(sum(df[df['cell_count'] == 0]['cell_freq']))

print('Procent liczności teoretycznych poniżej 5:', np.mean(avg_ex_cell_count_under_5), '%')
print('Komórki z licznością teoretyczną równą 0:', np.sum(zeros_count))

###################################
# Interpretacja statystyki testowej
###################################
prob = 0.95
critical = stats.chi2.ppf(prob, dof_genres_actor)
if abs(stat_genres_actor) >= critical:
    print('Zależne (odrzuć H0)')
else:
    print('Niezależne (nie ma podstaw do odrzucenia H0)')

###################################
# Interpretacja p-value
###################################
alpha = 1.0 - prob
if p_genres_actor <= alpha:
    print('Zależne (odrzuć H0)')
else:
    print('Niezależne (nie ma podstaw do odrzucenia H0)')


################################################################
# gatunek / oceny
################################################################
list_genre = list(df_genres_mod['genre'].head(5))
df_gatunki['genre_mod'] = df_gatunki['genre'].apply(lambda x: x if x in list_genre else np.nan)

df_gatunki['vote_average_mod'] = df_gatunki['vote_average_cap'].apply(
    lambda x: 5.0 if round(x) in [2.0, 3.0, 4.0] else 7.0 if round(x) in [9.0, 8.0] else round(x))

sum(df_gatunki[df_gatunki['genre_mod'].notnull()]['vote_average_cap'].apply(lambda x: round(x)).value_counts())
df_gatunki[df_gatunki['genre_mod'].notnull()]['vote_average_cap'].apply(lambda x: round(x)).value_counts()

chi_table_genres_vt_avg = pd.crosstab(index=round(df_gatunki['vote_average_mod']), columns=df_gatunki['genre_mod'],
                                      margins=True)

stat_genres_vt_avg, p_genres_vt_avg, dof_genres_vt_avg, expected_genres_vt_avg = stats.chi2_contingency(
    chi_table_genres_vt_avg, correction=False)
sum(expected_genres_vt_avg == 0)

chi_table_genres_vt_avg_ex = pd.DataFrame(expected_genres_vt_avg)
chi_table_genres_vt_avg_ex.columns = chi_table_genres_vt_avg.columns
gat_vt_avg_col_ex = chi_table_genres_vt_avg_ex.columns

avg_ex_cell_count_under_5 = []
zeros_count = []
for gen in gat_vt_avg_col_ex:
    df = pd.DataFrame(chi_table_genres_vt_avg_ex[gen].value_counts()).reset_index()
    df.columns = ['cell_count', 'cell_freq']
    avg_ex_cell_count_under_5.append(round(sum(df[df['cell_count'] < 5]['cell_freq']) / sum(df['cell_freq']) * 100))
    zeros_count.append(sum(df[df['cell_count'] == 0]['cell_freq']))
print('Procent liczności teoretycznych poniżej 5:', np.mean(avg_ex_cell_count_under_5), '%')
print('Komórki z licznością teoretyczną równą 0:', np.sum(zeros_count))

###################################
# Interpretacja statystyki testowej
###################################
prob = 0.95
critical = stats.chi2.ppf(prob, dof_genres_vt_avg)
if abs(stat_genres_vt_avg) >= critical:
    print('Zależne (odrzuć H0)')
else:
    print('Niezależne (nie ma podstaw do odrzucenia H0)')

###################################
# Interpretacja p-value
###################################
alpha = 1.0 - prob
if p_genres_vt_avg <= alpha:
    print('Zależne (odrzuć H0)')
else:
    print('Niezależne (nie ma podstaw do odrzucenia H0)')

################################################################
# aktorzy / oceny
################################################################
df_act_vt_avg_mod = pd.DataFrame(cast_df['actor'].value_counts()).reset_index()
df_act_vt_avg_mod.columns = ['actor', 'freq']
list_act_vt_avg = list(df_act_vt_avg_mod[df_act_vt_avg_mod['freq'] > 32]['actor'])
cast_df['actor_mod'] = cast_df['actor'].apply(lambda x: x if x in list_act_vt_avg else np.nan)

cast_df['vote_average_mod'] = cast_df['vote_average_cap'].apply(
    lambda x: 5.0 if round(x) in [2.0, 3.0, 4.0] else 7.0 if round(x) in [9.0, 8.0] else round(x))
cast_df[cast_df['actor_mod'].notnull() & cast_df['vote_average_mod'].notnull()]['id'].value_counts()
chi_table_actor_vt_avg = pd.crosstab(index=cast_df['actor_mod'],
                                     columns=cast_df[cast_df['vote_average_mod'].notnull()]['vote_average_mod'].apply(
                                         lambda x: str(round(x))), margins=True)

stat_actor_vt_avg, p_actor_vt_avg, dof_actor_vt_avg, expected_actor_vt_avg = stats.chi2_contingency(
    chi_table_actor_vt_avg, correction=True)
sum(expected_actor_vt_avg == 0)  # ok

chi_table_actor_vt_avg_ex = pd.DataFrame(expected_actor_vt_avg)
chi_table_actor_vt_avg_ex.columns = chi_table_actor_vt_avg.columns
actor_vt_avg_col_ex = chi_table_actor_vt_avg_ex.columns

avg_ex_cell_count_under_5 = []
zeros_count = []
for vt in actor_vt_avg_col_ex:
    df = pd.DataFrame(chi_table_actor_vt_avg_ex[vt].value_counts()).reset_index()
    df.columns = ['cell_count', 'cell_freq']
    avg_ex_cell_count_under_5.append(round(sum(df[df['cell_count'] < 5]['cell_freq']) / sum(df['cell_freq']) * 100))
    zeros_count.append(sum(df[df['cell_count'] == 0]['cell_freq']))

print('Procent liczności teoretycznych poniżej 5:', np.mean(avg_ex_cell_count_under_5), '%')
print('Komórki z licznością teoretyczną równą 0:', np.sum(zeros_count))

###################################
# Interpretacja statystyki testowej
###################################
prob = 0.95
critical = stats.chi2.ppf(prob, dof_actor_vt_avg)
if abs(stat_actor_vt_avg) >= critical:
    print('Zależne (odrzuć H0)')
else:
    print('Niezależne (nie ma podstaw do odrzucenia H0)')

###################################
# Interpretacja p-value
###################################
alpha = 1.0 - prob
if p_actor_vt_avg <= alpha:
    print('Zależne (odrzuć H0)')
else:
    print('Niezależne (nie ma podstaw do odrzucenia H0)')























"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Rozdział 4: Analiza wybranymi metodami uczenia maszynowego
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

##############################################################################################################################
# KOD 28: Wczytanie danych
##############################################################################################################################

filmy_clean = pd.read_csv(r"data\filmy_clean.csv",
                          low_memory=False)
print(filmy_clean.shape)

##############################################################################################################################
# KOD 29: Wykorzystanie definicji sukcesów filmów
##############################################################################################################################

########################
# Finansowo
########################
i_min_50 = np.percentile(filmy_clean['ROI_cap'], 50)
i_min_75 = np.percentile(filmy_clean['ROI_cap'], 75)
i_min_95 = np.percentile(filmy_clean['ROI_cap'], 95)
filmy_clean['FS_p50'] = filmy_clean['ROI_cap'].apply(
    lambda x: 1 if x >= i_min_50 else 0)
filmy_clean['FS_p75'] = filmy_clean['ROI_cap'].apply(
    lambda x: 1 if x >= i_min_75 else 0)
filmy_clean['FS_p95'] = filmy_clean['ROI_cap'].apply(
    lambda x: 1 if x >= i_min_95 else 0)
print(filmy_clean['FS_p50'].value_counts())
print(filmy_clean['FS_p75'].value_counts())
print(filmy_clean['FS_p95'].value_counts())

########################
# Popularnościowo
########################
v_min_50 = np.percentile(filmy_clean['vote_count_cap'], 50)
v_min_75 = np.percentile(filmy_clean['vote_count_cap'], 75)
v_min_95 = np.percentile(filmy_clean['vote_count_cap'], 95)
filmy_clean['VCS_p50'] = filmy_clean['vote_count_cap'].apply(
    lambda x: 1 if x >= v_min_50 else 0)
filmy_clean['VCS_p75'] = filmy_clean['vote_count_cap'].apply(
    lambda x: 1 if x >= v_min_75 else 0)
filmy_clean['VCS_p95'] = filmy_clean['vote_count_cap'].apply(
    lambda x: 1 if x >= v_min_95 else 0)
print(filmy_clean['VCS_p50'].value_counts())
print(filmy_clean['VCS_p75'].value_counts())
print(filmy_clean['VCS_p95'].value_counts())

########################
# Ocenowo
########################
aa_min = np.percentile(filmy_clean['vote_count_cap'], 33)
a_min_50 = np.percentile(filmy_clean['vote_average_cap'], 33)
a_min_75 = np.percentile(filmy_clean['vote_average_cap'], 66)
a_min_95 = np.percentile(filmy_clean['vote_average_cap'], 90)
filmy_clean['VAS1'] = filmy_clean['vote_average_cap'].apply(
    lambda x: 1 if x >= a_min_50 else 0)
filmy_clean['VAS2'] = filmy_clean['vote_count_cap'].apply(
    lambda x: 1 if x >= aa_min else 0)
filmy_clean['VAS3'] = filmy_clean['VAS1'] + filmy_clean['VAS2']
filmy_clean['VAS_p50'] = filmy_clean['VAS3'].apply(
    lambda x: 1 if x > 1 else 0)
filmy_clean = filmy_clean.drop(['VAS1', 'VAS2', 'VAS3'], axis=1)

filmy_clean['VAS4'] = filmy_clean['vote_average_cap'].apply(
    lambda x: 1 if x >= a_min_75 else 0)
filmy_clean['VAS5'] = filmy_clean['vote_count_cap'].apply(
    lambda x: 1 if x >= aa_min else 0)
filmy_clean['VAS6'] = filmy_clean['VAS4'] + filmy_clean['VAS5']
filmy_clean['VAS_p75'] = filmy_clean['VAS6'].apply(
    lambda x: 1 if x > 1 else 0)
filmy_clean = filmy_clean.drop(['VAS4', 'VAS5', 'VAS6'], axis=1)

filmy_clean['VAS7'] = filmy_clean['vote_average_cap'].apply(
    lambda x: 1 if x >= a_min_95 else 0)
filmy_clean['VAS8'] = filmy_clean['vote_count_cap'].apply(
    lambda x: 1 if x >= aa_min else 0)
filmy_clean['VAS9'] = filmy_clean['VAS7'] + filmy_clean['VAS8']
filmy_clean['VAS_p95'] = filmy_clean['VAS9'].apply(
    lambda x: 1 if x > 1 else 0)
filmy_clean = filmy_clean.drop(['VAS7', 'VAS8', 'VAS9'], axis=1)

print(filmy_clean['VAS_p50'].value_counts())
print(filmy_clean['VAS_p75'].value_counts())
print(filmy_clean['VAS_p95'].value_counts())

##############################################################################################################################
# Kod 30: Drzewa decyzyjne (CART)
##############################################################################################################################

####################
# Przykładowe drzewo
####################
iris = load_iris()
X = iris.data[:, 2:]  # Długość i szerokość płatka
y = iris.target

tree_clf = tree.DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(tree_clf,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)
plt.savefig("iris_drzewo.pdf", bbox_inches="tight")


################################
# Wybór zmiennych objaśniających
################################
inputs = filmy_clean.drop(
    ['FS_p50', 'FS_p75', 'FS_p95', 'VCS_p50', 'VCS_p75', 'VCS_p95', 'VAS_p50', 'VAS_p75', 'VAS_p95', 'is_friday.1',
     'screenplay_count', 'director_count', 'director_count_cap',
     'producer_count', 'crew_size', 'cast_size', 'spok_lang_count', 'prod_countr_count', 'prod_comp_count',
     'genres_count',
     'popularity', 'popularity_cap', 'vote_count_cap', 'vote_count', 'vote_average', 'vote_average_cap', 'revenue',
     'revenue_cap', 'budget', 'ROI_cap', 'ROI', 'runtime', 'rd_weekday', 'rd_day', 'rd_month', 'rd_year',
     'release_date',
     'title', 'id']
    , axis=1
)
print(inputs.shape)

################################
# Wybór zmiennych objaśnianych
################################
target_FS_p50 = filmy_clean['FS_p50']
target_FS_p75 = filmy_clean['FS_p75']
target_FS_p95 = filmy_clean['FS_p95']

target_VCS_p50 = filmy_clean['VCS_p50']
target_VCS_p75 = filmy_clean['VCS_p75']
target_VCS_p95 = filmy_clean['VCS_p95']

target_VAS_p50 = filmy_clean['VAS_p50']
target_VAS_p75 = filmy_clean['VAS_p75']
target_VAS_p95 = filmy_clean['VAS_p95']

inputs.columns


############################################
# Zamiana zmiennych kategorycznych na liczby
############################################
le_org_lang = LabelEncoder()
inputs['original_language_n'] = le_org_lang.fit_transform(inputs['original_language'])
inputs_n = inputs.drop('original_language', axis=1)




############################################
# Utworzenie modeli
############################################
model_FS_p50_score = []
model_FS_p75_score = []
model_FS_p95_score = []

model_VCS_p50_score = []
model_VCS_p75_score = []
model_VCS_p95_score = []

model_VAS_p50_score = []
model_VAS_p75_score = []
model_VAS_p95_score = []

model_FS_p50_feature_importances = np.zeros(38)
model_FS_p75_feature_importances = np.zeros(38)
model_FS_p95_feature_importances = np.zeros(38)

model_VCS_p50_feature_importances = np.zeros(38)
model_VCS_p75_feature_importances = np.zeros(38)
model_VCS_p95_feature_importances = np.zeros(38)

model_VAS_p50_feature_importances = np.zeros(38)
model_VAS_p75_feature_importances = np.zeros(38)
model_VAS_p95_feature_importances = np.zeros(38)

for i in range(0, 10): #bootstrap
    X_train_FS_p50, X_test_FS_p50, y_train_FS_p50, y_test_FS_p50 = train_test_split(inputs_n, target_FS_p50,
                                                                                    test_size=0.3)
    X_train_FS_p75, X_test_FS_p75, y_train_FS_p75, y_test_FS_p75 = train_test_split(inputs_n, target_FS_p75,
                                                                                    test_size=0.3)
    X_train_FS_p95, X_test_FS_p95, y_train_FS_p95, y_test_FS_p95 = train_test_split(inputs_n, target_FS_p95,
                                                                                    test_size=0.3)

    X_train_VCS_p50, X_test_VCS_p50, y_train_VCS_p50, y_test_VCS_p50 = train_test_split(inputs_n, target_VCS_p50,
                                                                                        test_size=0.3)
    X_train_VCS_p75, X_test_VCS_p75, y_train_VCS_p75, y_test_VCS_p75 = train_test_split(inputs_n, target_VCS_p75,
                                                                                        test_size=0.3)
    X_train_VCS_p95, X_test_VCS_p95, y_train_VCS_p95, y_test_VCS_p95 = train_test_split(inputs_n, target_VCS_p95,
                                                                                        test_size=0.3)

    X_train_VAS_p50, X_test_VAS_p50, y_train_VAS_p50, y_test_VAS_p50 = train_test_split(inputs_n, target_VAS_p50,
                                                                                        test_size=0.3)
    X_train_VAS_p75, X_test_VAS_p75, y_train_VAS_p75, y_test_VAS_p75 = train_test_split(inputs_n, target_VAS_p75,
                                                                                        test_size=0.3)
    X_train_VAS_p95, X_test_VAS_p95, y_train_VAS_p95, y_test_VAS_p95 = train_test_split(inputs_n, target_VAS_p95,
                                                                                        test_size=0.3)

    model_FS_p50 = tree.DecisionTreeClassifier()  # (criterion='entropy')
    model_FS_p75 = tree.DecisionTreeClassifier()  # (criterion='entropy')
    model_FS_p95 = tree.DecisionTreeClassifier()  # (criterion='entropy')

    model_VCS_p50 = tree.DecisionTreeClassifier()  # (criterion='entropy')
    model_VCS_p75 = tree.DecisionTreeClassifier()  # (criterion='entropy')
    model_VCS_p95 = tree.DecisionTreeClassifier()  # (criterion='entropy')

    model_VAS_p50 = tree.DecisionTreeClassifier()  # (criterion='entropy')
    model_VAS_p75 = tree.DecisionTreeClassifier()  # (criterion='entropy')
    model_VAS_p95 = tree.DecisionTreeClassifier()  # (criterion='entropy')

    model_FS_p50.fit(X_train_FS_p50, y_train_FS_p50)
    model_FS_p75.fit(X_train_FS_p75, y_train_FS_p75)
    model_FS_p95.fit(X_train_FS_p95, y_train_FS_p95)

    model_VCS_p50.fit(X_train_VCS_p50, y_train_VCS_p50)
    model_VCS_p75.fit(X_train_VCS_p75, y_train_VCS_p75)
    model_VCS_p95.fit(X_train_VCS_p95, y_train_VCS_p95)

    model_VAS_p50.fit(X_train_VAS_p50, y_train_VAS_p50)
    model_VAS_p75.fit(X_train_VAS_p75, y_train_VAS_p75)
    model_VAS_p95.fit(X_train_VAS_p95, y_train_VAS_p95)

    model_FS_p50_score.append(model_FS_p50.score(X_test_FS_p50, y_test_FS_p50))
    model_FS_p75_score.append(model_FS_p75.score(X_test_FS_p75, y_test_FS_p75))
    model_FS_p95_score.append(model_FS_p95.score(X_test_FS_p95, y_test_FS_p95))

    model_VCS_p50_score.append(model_VCS_p50.score(X_test_VCS_p50, y_test_VCS_p50))
    model_VCS_p75_score.append(model_VCS_p75.score(X_test_VCS_p75, y_test_VCS_p75))
    model_VCS_p95_score.append(model_VCS_p95.score(X_test_VCS_p95, y_test_VCS_p95))

    model_VAS_p50_score.append(model_VAS_p50.score(X_test_VAS_p50, y_test_VAS_p50))
    model_VAS_p75_score.append(model_VAS_p75.score(X_test_VAS_p75, y_test_VAS_p75))
    model_VAS_p95_score.append(model_VAS_p95.score(X_test_VAS_p95, y_test_VAS_p95))

    model_FS_p50_feature_importances = model_FS_p50_feature_importances + model_FS_p50.feature_importances_
    model_FS_p75_feature_importances = model_FS_p75_feature_importances + model_FS_p75.feature_importances_
    model_FS_p95_feature_importances = model_FS_p95_feature_importances + model_FS_p95.feature_importances_

    model_VCS_p50_feature_importances = model_VCS_p50_feature_importances + model_VCS_p50.feature_importances_
    model_VCS_p75_feature_importances = model_VCS_p75_feature_importances + model_VCS_p75.feature_importances_
    model_VCS_p95_feature_importances = model_VCS_p95_feature_importances + model_VCS_p95.feature_importances_

    model_VAS_p50_feature_importances = model_VAS_p50_feature_importances + model_VAS_p50.feature_importances_
    model_VAS_p75_feature_importances = model_VAS_p75_feature_importances + model_VAS_p75.feature_importances_
    model_VAS_p95_feature_importances = model_VAS_p95_feature_importances + model_VAS_p95.feature_importances_

model_FS_p50_feature_importances = model_FS_p50_feature_importances / 10
model_FS_p75_feature_importances = model_FS_p75_feature_importances / 10
model_FS_p95_feature_importances = model_FS_p95_feature_importances / 10

model_VCS_p50_feature_importances = model_VCS_p50_feature_importances / 10
model_VCS_p75_feature_importances = model_VCS_p75_feature_importances / 10
model_VCS_p95_feature_importances = model_VCS_p95_feature_importances / 10

model_VAS_p50_feature_importances = model_VAS_p50_feature_importances / 10
model_VAS_p75_feature_importances = model_VAS_p75_feature_importances / 10
model_VAS_p95_feature_importances = model_VAS_p95_feature_importances / 10



############################################
# Dokładność modeli
############################################
print(np.mean(model_FS_p50_score))
print(np.mean(model_FS_p75_score))
print(np.mean(model_FS_p95_score), '\n')

print(np.mean(model_VCS_p50_score))
print(np.mean(model_VCS_p75_score))
print(np.mean(model_VCS_p95_score), '\n')

print(np.mean(model_VAS_p50_score))
print(np.mean(model_VAS_p75_score))
print(np.mean(model_VAS_p95_score), '\n')



############################################
# Istotności zmiennych w modelach
############################################
importances_FS_p50 = pd.DataFrame(model_FS_p50.feature_importances_).transpose()
importances_FS_p50.columns = X_train_FS_p50.columns
importances_FS_p50 = importances_FS_p50.transpose().reset_index()
importances_FS_p50.columns = ['feature', 'importance']

importances_FS_p75 = pd.DataFrame(model_FS_p75.feature_importances_).transpose()
importances_FS_p75.columns = X_train_FS_p75.columns
importances_FS_p75 = importances_FS_p75.transpose().reset_index()
importances_FS_p75.columns = ['feature', 'importance']

importances_FS_p95 = pd.DataFrame(model_FS_p95.feature_importances_).transpose()
importances_FS_p95.columns = X_train_FS_p95.columns
importances_FS_p95 = importances_FS_p95.transpose().reset_index()
importances_FS_p95.columns = ['feature', 'importance']

importances_VCS_p50 = pd.DataFrame(model_VCS_p50.feature_importances_).transpose()
importances_VCS_p50.columns = X_train_VCS_p50.columns
importances_VCS_p50 = importances_VCS_p50.transpose().reset_index()
importances_VCS_p50.columns = ['feature', 'importance']

importances_VCS_p75 = pd.DataFrame(model_VCS_p75.feature_importances_).transpose()
importances_VCS_p75.columns = X_train_VCS_p75.columns
importances_VCS_p75 = importances_VCS_p75.transpose().reset_index()
importances_VCS_p75.columns = ['feature', 'importance']

importances_VCS_p95 = pd.DataFrame(model_VCS_p95.feature_importances_).transpose()
importances_VCS_p95.columns = X_train_VCS_p95.columns
importances_VCS_p95 = importances_VCS_p95.transpose().reset_index()
importances_VCS_p95.columns = ['feature', 'importance']

importances_VAS_p50 = pd.DataFrame(model_VAS_p50.feature_importances_).transpose()
importances_VAS_p50.columns = X_train_VAS_p50.columns
importances_VAS_p50 = importances_VAS_p50.transpose().reset_index()
importances_VAS_p50.columns = ['feature', 'importance']

importances_VAS_p75 = pd.DataFrame(model_VAS_p75.feature_importances_).transpose()
importances_VAS_p75.columns = X_train_VAS_p75.columns
importances_VAS_p75 = importances_VAS_p75.transpose().reset_index()
importances_VAS_p75.columns = ['feature', 'importance']

importances_VAS_p95 = pd.DataFrame(model_VAS_p95.feature_importances_).transpose()
importances_VAS_p95.columns = X_train_VAS_p95.columns
importances_VAS_p95 = importances_VAS_p95.transpose().reset_index()
importances_VAS_p95.columns = ['feature', 'importance']

importances = importances_FS_p50
importances.columns = ['feature', 'FS_50']
importances['FS_p75'] = importances_FS_p75['importance']
importances['FS_p95'] = importances_FS_p95['importance']

importances['VCS_p50'] = importances_VCS_p50['importance']
importances['VCS_p75'] = importances_VCS_p75['importance']
importances['VCS_p95'] = importances_VCS_p95['importance']

importances['VAS_p50'] = importances_VAS_p50['importance']
importances['VAS_p75'] = importances_VAS_p75['importance']
importances['VAS_p95'] = importances_VAS_p95['importance']


####################################################
# Drzewo decyzyjne dla najbardziej dokładnego modelu
####################################################
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(model_FS_p95,
                   filled=True)
plt.savefig("FS_95_nn.pdf", bbox_inches="tight")


##############################################################################################################################
# KOD 30: Głębokie sieci neuronowe
##############################################################################################################################

#################################
# Wybór zmiennych objaśniających
#################################
df_o = filmy_clean
df_o.columns
numFeatures = [ 'runtime_cap',
               'budget_cap', 'genres_count_cap', 'prod_comp_count_cap',
               'prod_countr_count_cap', 'spok_lang_count_cap', 'cast_size_cap',
               'crew_size_cap', 'producer_count_cap', 'screenplay_count_cap'
               ]

catFeatures = [ 'original_language',
                'rd_month_num', #'rd_weekday_num',
                'if_franchise', 'if_homepage', 'is_dmp_month', 'is_friday',
               'is_thursday', 'is_wednesday', 'is_english_spoken', 'is_Drama',
               'is_Comedy', 'is_Thriller', 'is_Action', 'is_Romance', 'is_Adventure',
               'is_Crime', 'is_Science Fiction', 'is_Horror', 'is_Family',
               'is_usa_prod', 'actor_top_freq', 'producer_top_freq',
               'director_top_freq', 'screenplay_top_freq', 'is_prod_comp_top',
               'ti_words_top_freq', 'ov_ta_words_top_freq']


#################################
# Wybór zmiennych objaśnianych
#################################
target_FS_p50 = "FS_p50"
target_FS_p75 = "FS_p75"
target_FS_p95 = "FS_p95"

target_VCS_p50 = "VCS_p50"
target_VCS_p75 = "VCS_p75"
target_VCS_p95 = "VCS_p95"

target_VAS_p50 = "VAS_p50"
target_VAS_p75 = "VAS_p75"
target_VAS_p95 = "VAS_p95"


###############################################################
# Przekodowanie zmiennych kategorycznych na zmienne wskaźnikowe
###############################################################
dummLev = pd.get_dummies(df_o[catFeatures], drop_first=True)

################################################################################
# Połączenie zmiennych numerycznych, kategorycznych (wskaźnikowych) oraz targetu
################################################################################
df_FS_p50 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_FS_p50]]], axis=1)
df_FS_p75 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_FS_p75]]], axis=1)
df_FS_p95 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_FS_p95]]], axis=1)

df_VCS_p50 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_VCS_p50]]], axis=1)
df_VCS_p75 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_VCS_p75]]], axis=1)
df_VCS_p95 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_VCS_p95]]], axis=1)

df_VAS_p50 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_VAS_p50]]], axis=1)
df_VAS_p75 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_VAS_p75]]], axis=1)
df_VAS_p95 = pd.concat([df_o[numFeatures], dummLev, df_o[[target_VAS_p95]]], axis=1)


#################################################################
# Standaryzacja (może być też normalizacja zamiast standaryzacji)
#################################################################

df_FS_p50[numFeatures] = df_FS_p50[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))
df_FS_p75[numFeatures] = df_FS_p75[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))
df_FS_p95[numFeatures] = df_FS_p95[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))

df_VCS_p50[numFeatures] = df_VCS_p50[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))
df_VCS_p75[numFeatures] = df_VCS_p75[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))
df_VCS_p95[numFeatures] = df_VCS_p95[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))

df_VAS_p50[numFeatures] = df_VAS_p50[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))
df_VAS_p75[numFeatures] = df_VAS_p75[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))
df_VAS_p95[numFeatures] = df_VAS_p95[numFeatures].apply(lambda x: (x-x.min())/(x.max()-x.min()))



#################################################################
# Utworzenie modeli
#################################################################
features_FS_p50 = df_FS_p50.columns.tolist()
features_FS_p50.remove(target_FS_p50)
features_FS_p75 = df_FS_p75.columns.tolist()
features_FS_p75.remove(target_FS_p75)
features_FS_p95 = df_FS_p95.columns.tolist()
features_FS_p95.remove(target_FS_p95)

features_VCS_p50 = df_VCS_p50.columns.tolist()
features_VCS_p50.remove(target_VCS_p50)
features_VCS_p75 = df_VCS_p75.columns.tolist()
features_VCS_p75.remove(target_VCS_p75)
features_VCS_p95 = df_VCS_p95.columns.tolist()
features_VCS_p95.remove(target_VCS_p95)

features_VAS_p50 = df_VAS_p50.columns.tolist()
features_VAS_p50.remove(target_VAS_p50)
features_VAS_p75 = df_VAS_p75.columns.tolist()
features_VAS_p75.remove(target_VAS_p75)
features_VAS_p95 = df_VAS_p95.columns.tolist()
features_VAS_p95.remove(target_VAS_p95)

model_FS_p50_score_nn = []
model_FS_p75_score_nn = []
model_FS_p95_score_nn = []

model_VCS_p50_score_nn = []
model_VCS_p75_score_nn = []
model_VCS_p95_score_nn = []

model_VAS_p50_score_nn = []
model_VAS_p75_score_nn = []
model_VAS_p95_score_nn = []

test_size=0.3
for i in range(0,10): #bootstrap
    X_train_FS_p50, X_valid_FS_p50 = train_test_split(df_FS_p50, test_size=test_size, random_state=2020, stratify=df_FS_p50[target_FS_p50].values)
    X_train_FS_p75, X_valid_FS_p75 = train_test_split(df_FS_p75, test_size=test_size, random_state=2020, stratify=df_FS_p75[target_FS_p75].values)
    X_train_FS_p95, X_valid_FS_p95 = train_test_split(df_FS_p95, test_size=test_size, random_state=2020, stratify=df_FS_p95[target_FS_p95].values)

    X_train_VCS_p50, X_valid_VCS_p50 = train_test_split(df_VCS_p50, test_size=test_size, random_state=2020, stratify=df_VCS_p50[target_VCS_p50].values)
    X_train_VCS_p75, X_valid_VCS_p75 = train_test_split(df_VCS_p75, test_size=test_size, random_state=2020, stratify=df_VCS_p75[target_VCS_p75].values)
    X_train_VCS_p95, X_valid_VCS_p95 = train_test_split(df_VCS_p95, test_size=test_size, random_state=2020, stratify=df_VCS_p95[target_VCS_p95].values)

    X_train_VAS_p50, X_valid_VAS_p50 = train_test_split(df_VAS_p50, test_size=test_size, random_state=2020, stratify=df_VAS_p50[target_VAS_p50].values)
    X_train_VAS_p75, X_valid_VAS_p75 = train_test_split(df_VAS_p75, test_size=test_size, random_state=2020, stratify=df_VAS_p75[target_VAS_p75].values)
    X_train_VAS_p95, X_valid_VAS_p95 = train_test_split(df_VAS_p95, test_size=test_size, random_state=2020, stratify=df_VAS_p95[target_VAS_p95].values)

    score_FS_p50, bestTrainProba_FS_p50, bestTestProba_FS_p50, history_FS_p50, shap_values_FS_p50 = net(X_train_FS_p50,
                                                                                                        X_valid_FS_p50,
                                                                                                        features_FS_p50,
                                                                                                        target_FS_p50,
                                                                                                        df_FS_p50,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=50,
                                                                                                        debug=True)

    score_FS_p75, bestTrainProba_FS_p75, bestTestProba_FS_p75, history_FS_p75, shap_values_FS_p75 = net(X_train_FS_p75,
                                                                                                        X_valid_FS_p75,
                                                                                                        features_FS_p75,
                                                                                                        target_FS_p75,
                                                                                                        df_FS_p75,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )

    score_FS_p95, bestTrainProba_FS_p95, bestTestProba_FS_p95, history_FS_p95, shap_values_FS_p95 = net(X_train_FS_p95,
                                                                                                        X_valid_FS_p95,
                                                                                                        features_FS_p95,
                                                                                                        target_FS_p95,
                                                                                                        df_FS_p95,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )

    score_VCS_p50, bestTrainProba_VCS_p50, bestTestProba_VCS_p50, history_VCS_p50, shap_values_VCS_p50 = net(X_train_VCS_p50,
                                                                                                        X_valid_VCS_p50,
                                                                                                        features_VCS_p50,
                                                                                                        target_VCS_p50,
                                                                                                        df_VCS_p50,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )

    score_VCS_p75, bestTrainProba_VCS_p75, bestTestProba_VCS_p75, history_VCS_p75, shap_values_VCS_p75 = net(X_train_VCS_p75,
                                                                                                        X_valid_VCS_p75,
                                                                                                        features_VCS_p75,
                                                                                                        target_VCS_p75,
                                                                                                        df_VCS_p75,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )

    score_VCS_p95, bestTrainProba_VCS_p95, bestTestProba_VCS_p95, history_VCS_p95, shap_values_VCS_p95 = net(X_train_VCS_p95,
                                                                                                        X_valid_VCS_p95,
                                                                                                        features_VCS_p95,
                                                                                                        target_VCS_p95,
                                                                                                        df_VCS_p95,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )

    score_VAS_p50, bestTrainProba_VAS_p50, bestTestProba_VAS_p50, history_VAS_p50, shap_values_VAS_p50 = net(X_train_VAS_p50,
                                                                                                        X_valid_VAS_p50,
                                                                                                        features_VAS_p50,
                                                                                                        target_VAS_p50,
                                                                                                        df_VAS_p50,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )

    score_VAS_p75, bestTrainProba_VAS_p75, bestTestProba_VAS_p75, history_VAS_p75, shap_values_VAS_p75 = net(X_train_VAS_p75,
                                                                                                        X_valid_VAS_p75,
                                                                                                        features_VAS_p75,
                                                                                                        target_VAS_p75,
                                                                                                        df_VAS_p75,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )

    score_VAS_p95, bestTrainProba_VAS_p95, bestTestProba_VAS_p95, history_VAS_p95, shap_values_VAS_p95 = net(X_train_VAS_p95,
                                                                                                        X_valid_VAS_p95,
                                                                                                        features_VAS_p95,
                                                                                                        target_VAS_p95,
                                                                                                        df_VAS_p95,
                                                                                                        n=2,
                                                                                                        nu1=400,
                                                                                                        nu2=100,
                                                                                                        epochs = 50,
                                                                                                        batchSize=100,
                                                                                                        debug=True
                                                                                                        )
    model_FS_p50_score_nn.append(score_FS_p50)
    model_FS_p75_score_nn.append(score_FS_p75)
    model_FS_p95_score_nn.append(score_FS_p95)

    model_VCS_p50_score_nn.append(score_VCS_p50)
    model_VCS_p75_score_nn.append(score_VCS_p75)
    model_VCS_p95_score_nn.append(score_VCS_p95)

    model_VAS_p50_score_nn.append(score_VAS_p50)
    model_VAS_p75_score_nn.append(score_VAS_p75)
    model_VAS_p95_score_nn.append(score_VAS_p95)


#################################################################
# Dokładność modeli
#################################################################
print('score_FS_p50: ', np.mean(model_FS_p50_score_nn))
print('score_FS_p75: ', np.mean(model_FS_p75_score_nn))
print('score_FS_p95: ', np.mean(model_FS_p95_score_nn))

print('score_VCS_p50: ', np.mean(model_VCS_p50_score_nn))
print('score_VCS_p75: ', np.mean(model_VCS_p75_score_nn))
print('score_VCS_p95: ', np.mean(model_VCS_p95_score_nn))

print('score_VAS_p50: ', np.mean(model_VAS_p50_score_nn))
print('score_VAS_p75 ', np.mean(model_VAS_p75_score_nn))
print('score_VAS_p95: ', np.mean(model_VAS_p95_score_nn))


#################################################################
# Istotności zmiennych w modelach (analiza wrażliwości)
#################################################################

#########
# FS
#########
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
shap.summary_plot(shap_values_FS_p50, df_FS_p50.drop('FS_p50', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('FS 50', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplot(1,3,2)
shap.summary_plot(shap_values_FS_p75, df_FS_p75.drop('FS_p75', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('FS 75', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplot(1,3,3)
shap.summary_plot(shap_values_FS_p95, df_FS_p95.drop('FS_p95', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('FS 95', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplots_adjust(wspace=1.2)
plt.tight_layout()
plt.savefig("sensitivity_analysis_nn_FS.pdf")
plt.show

#########
# VCS
#########
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
shap.summary_plot(shap_values_VCS_p50, df_VCS_p50.drop('VCS_p50', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('VCS 50', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplot(1,3,2)
shap.summary_plot(shap_values_VCS_p75, df_VCS_p75.drop('VCS_p75', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('VCS 75', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplot(1,3,3)
shap.summary_plot(shap_values_VCS_p95, df_VCS_p95.drop('VCS_p95', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('VCS 95', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplots_adjust(wspace=1.2)
plt.tight_layout()
plt.savefig("sensitivity_analysis_nn_VCS.pdf")
plt.show()

#########
# VAS
#########
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
shap.summary_plot(shap_values_VAS_p50, df_VAS_p50.drop('VAS_p50', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('VAS 50', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplot(1,3,2)
shap.summary_plot(shap_values_VAS_p75, df_VAS_p75.drop('VAS_p75', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('VAS 75', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplot(1,3,3)
shap.summary_plot(shap_values_VAS_p95, df_VAS_p95.drop('VAS_p95', axis=1), plot_type="bar", max_display=5, plot_size = (21, 12), show = False)
plt.title('VAS 95', fontsize=25)
plt.xlabel('Stopień wrażliwości',fontsize=15)

plt.subplots_adjust(wspace=1.2)
plt.tight_layout()
plt.savefig("sensitivity_analysis_nn_VAS.pdf")
plt.show()






