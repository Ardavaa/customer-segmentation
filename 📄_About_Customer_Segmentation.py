import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt

# feature engineering, selection + preprocessing tools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from feature_engine.encoding import OrdinalEncoder
from feature_engine.outliers import OutlierTrimmer
from datetime import datetime

# models
from sklearn.cluster import KMeans

st.header('Customer Personality Segmentation')
st.subheader('About the project')
st.markdown("""
            Customer Segmentation Analysis is a detailed analysis of a company's ideal customers.
            It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
            """)
st.image('img/segment.png')
st.markdown("""
Customer Segmentation helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.
            """)
st.subheader('Dataset Preview')
st.markdown('The source of the dataset: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data')
df = pd.read_csv('dataset/marketing_campaign.csv', sep='\t')
st.dataframe(df)

st.markdown(
"""
For more detail explanation and implementation about the cleaning method, feature engineering, clustering, and PCA, please visit my Kaggle Notebook on https://www.kaggle.com/code/ardava/customer-segmentation-clustering-pca#Customer-Segmentation-%7C-Clustering-&-PCA
""")

#################################################
# The following code is for the customer segmentation personality implementation on my Kaggle website
#################################################

df_cust = pd.read_csv('dataset/marketing_campaign.csv', sep='\t')
df_cust.dropna(inplace=True)
cat_cols = [var for var in df_cust.columns if df_cust[var].dtype == 'O']

df_cust['Age'] = datetime.now().year - df_cust['Year_Birth']
df_cust['EnrollmentDate'] = pd.to_datetime(df_cust['Dt_Customer'], format='%d-%m-%Y').apply(lambda x: x.year)

# removing unnecessary columns
df_cust = df_cust.drop(['Year_Birth', 'Dt_Customer', 'ID'], axis=1)

df_cust['Education'] = df_cust['Education'].replace({'2n Cycle': 'Master'})
df_cust['Marital_Status'] = df_cust['Marital_Status'].replace({
    'Married': 'In a Relationship',
    'Together': 'In a Relationship',
    'Divorced': 'Single',
    'Widow': 'Single',
    'Alone': 'Single',
    'YOLO': 'Single',
    'Absurd': 'Single',
})

people_per_household = df_cust['Marital_Status'].replace({
    'In a Relationship': 2,
    'Single': 1
})

df_cust['Family_Size'] = df_cust['Kidhome'] + df_cust['Teenhome'] + people_per_household
cat_cols = [var for var in df_cust.columns if df_cust[var].dtype == 'O']

purchases = ['NumCatalogPurchases', 'NumStorePurchases', 'NumWebPurchases']
df_cust['Num_Total_Purchases'] = df_cust[purchases].sum(axis=1)

promotions = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df_cust['Num_Accepted_Cmp'] = df_cust[promotions].sum(axis=1)

mnt_products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_cust['Total_Amount_Spent'] = df_cust[mnt_products].sum(axis=1)

cat_cols = [var for var in df_cust.columns if df_cust[var].dtype == 'O']

label_encoder = OrdinalEncoder(
    encoding_method='arbitrary',
    variables=cat_cols
)

label_encoder.fit(df_cust)

# Transform the data
df_cust = label_encoder.transform(df_cust)

num_cols = [var for var in df_cust.columns if df_cust[var].dtype != 'O']

# removing outliers on specific columns
remove_outlier = OutlierTrimmer(
    capping_method='iqr',
    tail='right',
    variables=['Age', 'Income']
)

df_cust = remove_outlier.fit_transform(df_cust)
df_cust.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

min_max_cols = ['Income', 'Recency', 'Age', 'Family_Size', 'EnrollmentDate']
ss_scaler_cols = ['Kidhome', 'Teenhome', 'MntWines', 'MntFruits', 'MntMeatProducts',
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                  'Num_Total_Purchases', 'Num_Accepted_Cmp', 'Total_Amount_Spent']

# copying original df to new df for scaling
df_cust_scaled = df_cust.copy()

# min max scaling
min_max_scaler = MinMaxScaler()
df_cust_scaled[min_max_cols] = min_max_scaler.fit_transform(df_cust[min_max_cols])

# standard scaling
ss_scaler = StandardScaler()
df_cust_scaled[ss_scaler_cols] = ss_scaler.fit_transform(df_cust[ss_scaler_cols])

# reduce dimensionality into 3 components
pca = PCA(n_components=3)
pca_df = pd.DataFrame(pca.fit_transform(df_cust_scaled), columns=['PC1', 'PC2', 'PC3'])

st.subheader('Visualization Of Data After Implementing PCA')
st.markdown("""
In this particular dataset, we can perform PCA to reduce the dimensionality of our dataset.
In this case, the optimal number of components is 3 for the PCA. Here is the preview of the data after PCA:
""")
# 3D interactive scatter plot
fig = px.scatter_3d(
    pca_df,
    x='PC1',
    y='PC2',
    z='PC3',
    title='3D Visualization Of Data After PCA'
)

fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    )
)
st.plotly_chart(fig)

st.subheader('KMeans Clustering')
st.markdown("For clustering, we'll be using KMeans Clustering. We'll determine the optimal number of clusters using the elbow method.")

# determine optimal number of clusters using elbow method
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=0)
#     kmeans.fit(pca_df)
#     wcss.append(kmeans.inertia_)

# # plot elbow method
# fig_elbow = plt.figure()
# plt.plot(range(1, 11), wcss, marker='o')
# plt.title('Elbow Method for Optimal Number of Clusters')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# st.pyplot(fig_elbow)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(pca_df)
    wcss.append(kmeans.inertia_)

# Prepare data for Altair
df_wcss = pd.DataFrame({
    'Number of clusters': range(1, 11),
    'WCSS': wcss
})

# Create Altair chart
chart = alt.Chart(df_wcss).mark_line().encode(
    x=alt.X('Number of clusters:O', title='Number of clusters'),
    y=alt.Y('WCSS:Q', title='WCSS'),
    tooltip=['Number of clusters', 'WCSS']
).properties(
    title='Elbow Method for Optimal Number of Clusters'
) + alt.Chart(df_wcss).mark_point().encode(
    x=alt.X('Number of clusters:O'),
    y=alt.Y('WCSS:Q')
)

# Display chart in Streamlit
st.altair_chart(chart, use_container_width=True)

st.markdown('From the elbow method, we can see that the optimal number of clusters is 3. Let\'s see how the clusters look like.')

kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(pca_df)

# add cluster to original dataframe
df_cust['Cluster_KMeans'] = y_kmeans

# 3D interactive scatter plot of clustered data
# create the 3D scatter plot with the specified color map
fig_clusters = px.scatter_3d(
    pca_df,
    x='PC1',
    y='PC2',
    z='PC3',
    color=y_kmeans,
    title='3D Visualization Of Clustered Data After PCA (KMeans)'
)

fig_clusters.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    )
)

st.plotly_chart(fig_clusters)

st.markdown(
"""
After performing KMeans Clustering, we conclude that there will be 3 groups (cluster) for each customer personality.
Let's analyze our clusters with visualizations.

"""
)

st.subheader('Data Visualization based on Clusters')

###########################################################
# distribution of the clusters (KMeans)
fig_cluster_dist = alt.Chart(df_cust).mark_bar().encode(
    x=alt.X('Cluster_KMeans:O', title='Cluster'),
    y=alt.Y('count():Q', title='Count'),
    color='Cluster_KMeans:N',
    tooltip=['Cluster_KMeans:N', 'count():Q']
).properties(
    title='Distribution Of The Clusters (KMeans)'
)
st.altair_chart(fig_cluster_dist, use_container_width=True)

st.markdown(
"""
The highest count of records is in Cluster 2 (1048 records), followed by Cluster 0 (589 records), and Cluster 1 (568 records).
"""
)

# Distribution of Income by Cluster (KMeans)
################## DONE #####################
# income by Cluster (KMeans)
fig_income_by_cluster = px.histogram(
    df_cust,
    x='Income',
    color='Cluster_KMeans',
    nbins=50,
    barmode='stack',
    labels={'Income': 'Income', 'count': 'Count'}
)

fig_income_by_cluster.update_layout(
    title='Distribution of Income by Cluster (KMeans)',
    xaxis_title='Income',
    yaxis_title='Count'
)

st.plotly_chart(fig_income_by_cluster, use_container_width=True)
st.markdown(
"""
From the visualization above, we can see that Cluster 1 tends to have the highest income, followed by Cluster 0 (moderate income), and Cluster 2 (low income).
"""
)


################## DONE #####################
# scatter plot for Income vs Num_Total_Purchases
fig_income_purchases = alt.Chart(df_cust).mark_point().encode(
    x=alt.X('Num_Total_Purchases:Q', title='Number of Total Purchases'),
    y=alt.Y('Income:Q', title='Income'),
    color='Cluster_KMeans:N',
    tooltip=['Num_Total_Purchases:Q', 'Income:Q', 'Cluster_KMeans:N']
).properties(
    title='Customer Income and Number of Total Purchases'
)

st.altair_chart(fig_income_purchases, use_container_width=True)

################## DONE #####################
# scatter plot for Income vs Total_Amount_Spent
fig_income_spent = alt.Chart(df_cust).mark_point().encode(
    x=alt.X('Total_Amount_Spent:Q', title='Total Amount Spent'),
    y=alt.Y('Income:Q', title='Income'),
    color='Cluster_KMeans:N',
    tooltip=['Total_Amount_Spent:Q', 'Income:Q', 'Cluster_KMeans:N']
).properties(
    title='Customer Income and Total Amount Spent'
)
st.altair_chart(fig_income_spent, use_container_width=True)


################## DONE #####################
# Total Amount Spent by Cluster (KMeans)
fig_spent_box = alt.Chart(df_cust).mark_boxplot().encode(
    x=alt.X('Cluster_KMeans:O', title='Cluster'),
    y=alt.Y('Total_Amount_Spent:Q', title='Total Amount Spent'),
    color='Cluster_KMeans:N'
).properties(
    title='Total Amount Spent by Cluster (KMeans)'
)
st.altair_chart(fig_spent_box, use_container_width=True)

st.markdown(
"""
Cluster 2 tends to have the highest total amount of spent and total of purchases, while Cluster 0 tends to have moderately total amount of spent but have almost the same total purchases like Cluster 1.
For Cluster 1, it tends to have the lowest total amount of spent and total of purchases.
"""
)
################## DONE #####################
# Number of Accepted Campaigns (KMeans)
df_cust['Num_Accepted_Cmp'] = df_cust['Num_Accepted_Cmp'].astype(int)

# Create the stacked bar chart
fig_accepted_campaigns = alt.Chart(df_cust).mark_bar().encode(
    x=alt.X('Cluster_KMeans:O', title='Cluster'),
    y=alt.Y('sum(Num_Accepted_Cmp):Q', title='Number of Accepted Campaigns'),
    color='Cluster_KMeans:N',
    tooltip=['Cluster_KMeans:N', 'sum(Num_Accepted_Cmp):Q']
).properties(
    title='Number of Accepted Campaigns (KMeans)',
    width=600,
    height=400
).configure_mark(
    opacity=0.8
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
)

st.altair_chart(fig_accepted_campaigns, use_container_width=True)
st.markdown(
"""
Cluster 1 has the highest number of accepted campaigns, followed by Cluster 0 and Cluster 2.
"""
)
################## DONE #####################
# Age Distribution by Cluster (KMeans)
# Data preparation
df_age_dist = df_cust.groupby(['Cluster_KMeans', 'Age']).size().reset_index(name='Count')

# Plotting
fig_age_dist = alt.Chart(df_age_dist).mark_bar().encode(
    x=alt.X('Age:Q', title='Age'),
    y=alt.Y('Count:Q', title='Count'),
    color='Cluster_KMeans:N',
    tooltip=['Cluster_KMeans:N', 'Age:Q', 'Count:Q']
).properties(
    title='Age Distribution by Cluster (KMeans)',
    width=600
).configure_mark(
    opacity=0.8
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
)
st.altair_chart(fig_age_dist, use_container_width=True)


st.subheader('Insights Conclusion')

col1, col2= st.columns(2)
st.success(
    "**Cluster 0**:\n"
    "- Income Distribution: Primarily between `$40,000` and `$70,000`. The income is relatively well-distributed with a few outliers on both the higher and lower ends.\n"
    "- Spending Behavior: Moderate spender, with most spending between `$500` and `$1,000`. There is a reasonable number of accepted campaigns, indicating moderate engagement with promotional offers.\n"
    "- Age Distribution: The age distribution peaks around 50 years, indicating that this cluster likely consists of middle-aged individuals.\n")

st.warning(
        "**Cluster 1**:\n"
    "- Income Distribution: High-income individuals with most having incomes above `$70,000`.\n"
    "- Spending Behavior: Highest spenders, with amounts ranging between `$1,000` and `$2,000`. They show significant engagement in marketing campaigns, likely because they have higher disposable incomes.\n"
    "- Age Distribution: The age distribution is spread out, with a peak around 40-50 years, suggesting that this cluster contains relatively younger but financially successful individuals.\n")

st.info(
        "**Cluster 2**:\n"
    "- Income Distribution: This cluster consists of low-income individuals, predominantly earning between `$0` and `$40,000`.\n"
    "- Spending Behavior: These individuals spend the least, with the majority spending under `$500`.\n"
    "- Age Distribution: The age distribution shows a broad spread but with a significant portion of younger individuals, possibly under 40 years of age.\n")

st.header(' ')

st.info("Copyright © Ardava Barus - All rights reserved")
