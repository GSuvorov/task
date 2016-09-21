#Task-1
#Created by Vorobyev Ruslan
#2016
import pandas as pd
import numpy as np
import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression


# Функция для избавления от лишних символов, знаков, делает из загавных букв строчные; далее оставляет от слова только его "главую" часть (стемминг)
def change_words(text):
    tmp = [str.lower(i) for i in text if not ((i>='!') and (i <='~'))]
    tmp = ''.join(tmp)
    words = tmp.split()
    x = len(words)
    for i in range(x):
        if words[i]!='':
            stemmer=Stemmer.Stemmer('russian')
            words[i]=stemmer.stemWord(words[i])
    text1 = ' '.join([i for i in words if not i==''])

    return text1


def main():
    # Считываем таблицы с данными "позитивные" и "отрицательные" отзывы
    negative = pd.read_csv('negative.csv',delimiter=',')
    positive = pd.read_csv('positive.csv',delimiter=',')
   
    # Удаляем лишний столбец
    del positive['Unnamed: 0']
    del negative['Unnamed: 0']
   
    # Объединяем обе таблицы в одну и применяем к столбцу text функцию change_words
    allofit = pd.concat((positive,negative),ignore_index=True)
    allofit['text'] = allofit['text'].apply(change_words)

    # Можем записать получинные данные в новый файл text.txt
    #allofit['text'].to_csv('text.txt', header=False,index=False)

    #
    permall = allofit.iloc[np.random.permutation(len(allofit))]
     
    # Обучаем классификатор
    tfidf = TfidfVectorizer(ngram_range=(1,3))
    vec = tfidf.fit_transform(permall['text'])

    # Кросс-валидация исходного набора данных
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(vec, permall['label'], test_size=0.25, random_state=0)

    #Логистическая регрессия
    clf = LogisticRegression()
    clf.fit(X_train,y_train)

    #Считываем предложение и делаем для него предсказание: насколько оно близко к одному из двух состояний (негативное || позитивное)
    a = input('Введите предложение: \n')
    while a!='stop':
        sentence = change_words(a)
        vc = tfidf.transform([sentence])
        b=clf.predict_proba(vc)
        if b[0][0]>b[0][1]:
            if b[0][0]>0.75:
                print('Это предложение скорее всего негативного характера')
            else:
                print('Это предложение нейтральное, но с оттенками негатива')
        else:
            if b[0][1]>0.75:
                print('Это предложение скорее всего положительного характера')
            else:
                print('Это предложение нейтральное, но с оттенками позитива')
        a = input('\nВведите новое проедложение или stop для останова: \n')
main()