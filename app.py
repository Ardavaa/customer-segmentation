import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

with open('.pkl/pipeline.pkl', 'rb') as pickle_in:
    pipeline = pkl.load(pickle_in)

st.title('Customer Classification Prediction')

col1, col2, col3 = st.columns(3)

with col1:
    st.success(
        "Cluster 0 (Green):\n"
        "- Income: $40,000 - $70,000, with some outliers.\n"
        "- Spending: $500 - $1,000; moderate engagement.\n"
        "- Age: Peaks around 50 years.\n"
        "\n "
        "\n "
    )

with col2:
    st.warning(
        "Cluster 1 (Orange):\n"
        "- Income: Above $70,000.\n"
        "- Spending: $1,000 - $2,000; high engagement.\n"
        "- Age: Peaks around 40-50 years.\n"
        "\n "
        "\n "
    )

with col3:
    st.info(
        "Cluster 2 (Blue):\n"
        "- Income: $0 - $40,000.\n"
        "- Spending: Under $500; least spending.\n"
        "- Age: Broad range, with many under 40 years.\n"
        "\n "
        "\n "
    )
Education = st.selectbox('What is your education level?', ['Graduated', 'PhD', 'Master', 'Basic'])
Marital_Status = st.selectbox('What is your marital status?', ['Single', 'In a Relationship'])
Income = st.number_input('What is your income?', min_value=0.0)

# asking for the number of kids and teens at home
Kidhome = st.number_input('How many kids do you have at home?', min_value=0)
Teenhome = st.number_input('How many teens do you have at home?', min_value=0)

# asking for recency and spending amounts
Recency = st.number_input('How recent was your last purchase? (in days)', min_value=0.0)
MntWines = st.number_input('How much did you spend on wines?', min_value=0.0)
MntFruits = st.number_input('How much did you spend on fruits?', min_value=0.0)
MntMeatProducts = st.number_input('How much did you spend on meat products?', min_value=0.0)
MntFishProducts = st.number_input('How much did you spend on fish products?', min_value=0.0)
MntSweetProducts = st.number_input('How much did you spend on sweet products?', min_value=0.0)
MntGoldProds = st.number_input('How much did you spend on gold products?', min_value=0.0)

# asking about the number of purchases and visits
NumDealsPurchases = st.number_input('How many purchases did you make from deals?', min_value=0)
NumWebPurchases = st.number_input('How many purchases did you make online?', min_value=0)
NumCatalogPurchases = st.number_input('How many purchases did you make from catalogs?', min_value=0)
NumStorePurchases = st.number_input('How many purchases did you make in stores?', min_value=0)
NumWebVisitsMonth = st.number_input('How many times did you visit our website this month?', min_value=0)

# asking about complaints, responses, and other personal metrics
Complain = st.number_input('How many complaints have you made?', min_value=0)
Response = st.number_input('How many responses have you given?', min_value=0)
Age = st.number_input('What is your age?', min_value=0)
EnrollmentDate = st.number_input('What year did you enroll?', min_value=0)
Family_Size = st.number_input('What is your family size?', min_value=0)
Num_Total_Purchases = st.number_input('How many total purchases have you made?', min_value=0)
Num_Accepted_Cmp = st.number_input('How many campaigns have you accepted?', min_value=0)
Total_Amount_Spent = st.number_input('What is the total amount you have spent?', min_value=0.0)

columns = [
    'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 
    'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
    'NumWebVisitsMonth', 'Complain', 'Response', 'Age', 'EnrollmentDate', 'Family_Size', 
    'Num_Total_Purchases', 'Num_Accepted_Cmp', 'Total_Amount_Spent'
]

# label mapping for Education
education_mapping = {
    'Graduated': 0,
    'PhD': 1,
    'Master': 2,
    'Basic': 3
}

# label mapping for Marital_Status
marital_status_mapping = {
    'Single': 0,
    'In a Relationship': 1
}

Education_mapped = education_mapping[Education]
Marital_Status_mapped = marital_status_mapping[Marital_Status]


input_data = pd.DataFrame([[
    Education_mapped, Marital_Status_mapped, Income, Kidhome, Teenhome, Recency, MntWines, 
    MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds,
    NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, 
    NumWebVisitsMonth, Complain, Response, Age, EnrollmentDate, Family_Size, 
    Num_Total_Purchases, Num_Accepted_Cmp, Total_Amount_Spent
]], columns=columns)

cluster_labels = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2"
}

# make prediction
if st.button('Predict'):
    prediction = pipeline.predict(input_data)
    
    cluster_label = cluster_labels.get(prediction[0], "Unknown Cluster")
    st.success(f'Predicted Customer Group: {cluster_label}')

st.info('Copyright © Ardava Barus - All rights reserved')
