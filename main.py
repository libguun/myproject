import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB



dataset_name = st.sidebar.selectbox("Өгөгдөл сонгох", ("Customer segmentation", "Product segmentation"))
ml_alogs = st.sidebar.selectbox("Машин сургалтын аргууд", ('Decision Tree', 'Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Multi Layer Perceptron', 'Naive Bayes'))

def get_df(dataset_name):
    if dataset_name == "Customer segmentation":
        df = pd.read_csv("train_data_customer_segmentation.csv")
    else:
        df = pd.read_csv("train_data_product_segmentation.csv")
    df
    X = df.iloc[:, 1:4].values 
    y = df.iloc[:, 4].values
    return X,y
X,y = get_df(dataset_name)
st.write("Өгөгдлийн хэмжээ", X.shape)
st.write("Бүлгийн тоо", len(np.unique(y)))

#Data beldelt
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_real = np.concatenate((y_train, y_test))

# Surgaltiin arguud
def select_ml_alog(ml_alogs):
    if ml_alogs == "Decision Tree":
        model = DecisionTreeClassifier() 
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        dt_train_score = model.score(x_train, y_train)
        dt_test_score = model.score(x_test, y_test)

        st.write("Гүйцэтгэл :",dt_test_score*100,"%")

    elif ml_alogs == "Random Forest":
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        rf_train_score = model.score(x_train, y_train)
        rf_test_score = model.score(x_test, y_test)

        st.write("Гүйцэтгэл :",rf_test_score*100,"%")

    elif ml_alogs == "Logistic Regression":
        model = LogisticRegression()
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        lr_train_score = model.score(x_train, y_train)
        lr_test_score = model.score(x_test, y_test)

        st.write("Гүйцэтгэл :",lr_test_score*100,"%")
    elif ml_alogs == "Support Vector Machine":
        model = SVC()
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        svm_train_score = model.score(x_train, y_train)
        svm_test_score = model.score(x_test, y_test)

        st.write("Гүйцэтгэл :",svm_test_score*100,"%")

    elif ml_alogs == "Multi Layer Perceptron":
        model = MLPClassifier()
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        mlp_train_score = model.score(x_train, y_train)
        mlp_test_score = model.score(x_test, y_test)

        st.write("Гүйцэтгэл :",mlp_test_score*100,"%")
    elif ml_alogs == "Naive Bayes":
        model = GaussianNB()
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        nb_train_score = model.score(x_train, y_train)
        nb_test_score = model.score(x_test, y_test)

        st.write("Гүйцэтгэл :",nb_test_score*100,"%")
    
select_ml_alog(ml_alogs)

