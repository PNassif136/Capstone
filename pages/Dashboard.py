# Import libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from numpy import nan
import pickle
import math
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pycountry
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from numpy import mean, std, percentile
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
import warnings
from sklearn.preprocessing import PowerTransformer
from millify import millify #display large numbers in reduced form

st.set_page_config(
    page_title='Interactive Dashboard',
    page_icon=':star:',
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
    )

# Read the processed CSV file into a Pandas dataframe
# Wrap it in a function and cache it for faster reruns

@st.cache()
def load_visuals():
    df = pd.read_csv("kfh_visuals_zip.csv", compression='gzip')
    return df
df = load_visuals()

option = option_menu(None, ["Home", "Visuals", "Statistics", "Predictions"],
    icons=['house-fill', 'kanban-fill', "cpu-fill", 'gear-wide-connected'],
    menu_icon="cast", default_index=0, orientation="horizontal")

if option == 'Home':
    st.success('Welcome to the Interactive Dashboard!')
    
    # Import GIF from Giphy using Markdown + align to center
    st.markdown(f'''
            <p align="center">
                <img src="https://media.giphy.com/media/qgQUggAC3Pfv687qPC/giphy.gif" alt="Programming" style="border-style: solid"/>
             </p>
            ''', unsafe_allow_html=True)
    
    # Display the number of rows and columns in a sentence
    st.write('')  #Line space but not markdown since it affects page scroll
    st.write("This dataset has: ", df.shape[0], " transactions and ", df.shape[1], " features")

    # Display df.head() under an expandable tab
    with st.expander("Quick Glance"):
        glance = st.slider('Select desired number of transactions to display', min_value=1, max_value=100, value=5)
        st.write(df.head(glance))

    # Review the summary statistics under an expandable tab
    with st.expander("Summary Statistics"):
        st.write(df[['AmountPaid']].describe())

    # Change basic design of containers using CSS
    st.markdown("""
    <style>
        div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 1% 5% 5%;
        border-radius: 5px;
        color: rgb(30, 103, 119);
        overflow-wrap: break-word;
        
    }

    /* breakline for metric text         */
    div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: black;
        font-size: medium;
        font-weight: bold;
    }
    </style>
    """
    , unsafe_allow_html=True)

    # Place metric cards in containers
    col1, col2, col3 = st.columns([1.5,1.5,1.5])
    col4, col5 = st.columns(2)

    # Plug relevant data into the containers
    col1.metric("Total Transactions", millify(df.shape[0], precision=2))
    col2.metric("Most Enrolled Tier", "Silver", "- Black")
    col3.metric("Most Used Multiplier", "1x", "- 15x")
    col4.metric("Most Used Card", "Visa Platinum Select", "- Visa Infinite Metal Silver")
    col5.metric("Most Popular MCC", "Miscellaneous Stores", "- Car Rental")

if option == 'Visuals':
    # Import CSV with more modifications to Continents, MCC, and features
    # Source: https://surendraredd.github.io/Books/examples.html
    @st.cache()
    def load_visuals():
        df = pd.read_csv("kfh_visuals_zip.csv", compression='gzip')
        return df
    df = load_visuals()

    st.header('Exploratory Data Analysis')
    st.code('Scroll down to visually explore millions of transactions')

    # Add a slider that filters total number of records and updates all subsequent values
    with st.expander('', expanded=True):
        records = st.slider('Select total desired transactions', df.shape[0], 1, df.shape[0], key="2")
        st.write('The below visuals now reflect ', records, 'transactions based on your selection')
        df = df.iloc[0:records]

    #Create a container/box of graphs so they look like a dashboard
    with st.container():
        col1, col2= st.columns([1.5,1])
        with col1:
            # Visualize percentage of customers by tier
            labels = df['Tier'].dropna().unique()
            values = df['Tier'].value_counts()

            pie = go.Figure(data=[go.Pie(labels=labels,
                                      values=values,
                                      textinfo='label+percent')])
            pie.layout.update(height=450, width=450,showlegend=False, plot_bgcolor='rgba(0,0,0,0)',   #Scales graph, removes legend, & changes bg color
                                    title="Percentage of Customers by Tier", title_x=0.5,    #Sets title & centers it
                                    xaxis_showgrid=False, yaxis_showgrid=False)
            st.write(pie)

        with col2:
            # Visualize total transactions by quarter
            quarter = df.groupby(by=["Quarter"]).size().reset_index(name="counts")
            fig2 = px.bar(quarter, x='Quarter', y='counts', color='Quarter', text_auto='.2s')
            fig2.layout.update(yaxis_title='Total Count', xaxis_title='Quarter', plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=False, title="Total Transactions by Quarter",
                                        height=450, width=450, title_x=0.5,
                                        xaxis_showgrid=False, yaxis_showgrid=False)
            fig2.update_coloraxes(showscale=False) #hides the color bar
            fig2.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig2.update_yaxes(showticklabels=False)

            fig2.update_layout(
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [1,2,3,4],
                    ticktext = ['Q1', 'Q2', 'Q3', 'Q4'])) #Customizes the X-Axis labels
            st.write(fig2)

        col3, col4= st.columns([1.5,1])
        with col3:
            # Visualize total transactions by continent
            continents = df.groupby(by=["Continent"]).size().reset_index(name="counts")

            fig3 = px.histogram(continents, x='Continent', y='counts', color='Continent', text_auto='.2s')
            fig3.layout.update(yaxis_title='Total Count', xaxis_title='Continent', plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=False, title="Total Transactions by Continent",
                                        height=450, width=450, title_x=0.5,
                                        xaxis_showgrid=False, yaxis_showgrid=False)
            fig3.update_xaxes(tickangle=45, categoryorder='total descending')
            fig3.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig3.update_yaxes(showticklabels=False)
            st.write(fig3)

        with col4:
            # Visualize total transactions by partner type
            partners = df.groupby(by=["Partner"]).size().reset_index(name="counts")

            fig5 = px.histogram(partners, x='Partner', y='counts', color='Partner', text_auto='.2s')
            fig5.layout.update(yaxis_title='Total Count', xaxis_title='Partner Type',
                                        showlegend=False, title="Total Transactions by Partner Type",
                                        height=450, width=450, title_x=0.5, plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis_showgrid=False, yaxis_showgrid=False)
            fig5.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig5.update_yaxes(showticklabels=False)

            fig5.update_layout(
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0,1],
                    ticktext = ['Non-Partner', 'Partner']))
            st.write(fig5)

        col5, col6 = st.columns([1.5,1])
        with col5:
            # Visualize total transactions by marketing type
            marketing = df.groupby(by=["MarketingType"]).size().reset_index(name="counts")
            fig1 = px.histogram(marketing, x='MarketingType', y='counts', color='MarketingType', text_auto='.2s')
            fig1.layout.update(yaxis_title='Total Count', xaxis_title='Marketing Type',
                                    showlegend=False, title="Total Transactions by Marketing Type",
                                    height=450, width=450, title_x=0.5, plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis_showgrid=False, yaxis_showgrid=False)

            fig1.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig1.update_yaxes(showticklabels=False)
            st.write(fig1)

        with col6:
            # Visualize total transactions by card type
            cards = df.groupby(by=["CardType"]).size().reset_index(name="counts")
            fig4 = px.histogram(cards, x='CardType', y='counts', text_auto='.2s')
            fig4.layout.update(yaxis_title='Total Count', xaxis_title='Card Type',
                                        showlegend=False, title="Total Transactions by Card Type",
                                        height=600, width=600, title_x=0.5, plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis_showgrid=False, yaxis_showgrid=False)
            fig4.update_xaxes(categoryorder='total descending')
            fig4.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig4.update_yaxes(showticklabels=False)
            st.write(fig4)

        col7, col8 = st.columns([1.5,1])

        with col7:
            # Visualize usage frequency per points multiplier
            multiplier = df.groupby(by=["PointsMultiplier"]).size().reset_index(name="counts")

            fig7 = px.bar(multiplier, y='counts', color='PointsMultiplier', text_auto='.2s')
            fig7.layout.update(yaxis_title='Total Count', xaxis_title='Points Multiplier', plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=False, title="Usage Frequency Per Points Multiplier",
                                        title_x=0.5, height=450, width=450, margin=dict(pad=20),
                                        xaxis_showgrid=False, yaxis_showgrid=False)

            fig7.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig7.update_yaxes(showticklabels=False)
            fig7.update_coloraxes(showscale=False)

            fig7.update_layout(
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0, 1, 2, 3, 4, 5],
                    ticktext = ['0x', '1x', '1.5x', '5x', '10x', '15x']))
            st.write(fig7)

        with col8:
            # Visualize total transactions by merchant category
            merchants = df.groupby(by=["MerchantCategory"]).size().reset_index(name="counts")

            fig6 = px.histogram(merchants, x='MerchantCategory', y='counts', text_auto='.2s')
            fig6.layout.update(yaxis_title='Total Count', xaxis_title='Merchant Category', plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=False, title="Total Transactions by Merchant Category",
                                        height=600, width=600, title_x=0.5, margin=dict(pad=20), #to add some padding between the x-axis and the labels
                                        xaxis_showgrid=False, yaxis_showgrid=False)
            fig6.update_xaxes(categoryorder='total descending', tickangle=90)
            fig6.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig6.update_yaxes(showticklabels=False)
            st.write(fig6)

        col9, col10 = st.columns([1.5,1])
        with col9:
            # Visualize average spend per tier
            tierspend = df[['AmountPaid', 'Tier']]
            tierspend = tierspend.groupby(['Tier']).mean().reindex(['Green', 'Silver', 'Black']).round()
            fig8 = px.line(tierspend, y="AmountPaid", title='Average Amount Paid Per Tier', markers=True)
            fig8.layout.update(yaxis_title='Average Spend', xaxis_title='Tier',
                                        showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                                        title_x=0.5, height=450, width=450,
                                        xaxis_showgrid=False, yaxis_showgrid=False)

            fig8.add_scatter(x = [fig8.data[0].x[-1]], y = [fig8.data[0].y[-1]],
                                 mode = 'markers + text',
                                 marker = {'color':'red', 'size':14},
                                 showlegend = False)                  #This will add the red marker to highlight the exponential increase in average spend

            st.write(fig8)

        with col10:
            # Visualize total spend per tier
            tierspend2 = df[['AmountPaid', 'Tier']]
            tierspend2 = (tierspend2.groupby(['Tier']).sum().reindex(['Green', 'Silver', 'Black']).round())
            fig9 = px.line(tierspend2, y="AmountPaid", title='Total Amount Paid Per Tier', markers=True)
            fig9.layout.update(yaxis_title='Total Spend', xaxis_title='Tier',
                                        showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                                        title_x=0.5, height=450, width=450,
                                        xaxis_showgrid=False, yaxis_showgrid=False)

            fig9.add_scatter(x = [fig9.data[0].x[-2]], y = [fig9.data[0].y[-2]],
                                 mode = 'markers + text',
                                 marker = {'color':'red', 'size':14},
                                 showlegend = False)

            st.write(fig9)

        col22, col23 = st.columns([1.5,1])
        with col22:
            # Visualize average amount paid per card type
            cardspend = df[['CardType', 'AmountPaid']]
            cardspend = cardspend.groupby(['CardType']).mean()
            fig10 = px.bar(cardspend, y='AmountPaid', text_auto='.2s')

            fig10.layout.update(yaxis_title='Average Spend', xaxis_title='Card Type', plot_bgcolor='rgba(0,0,0,0)',
                                                    showlegend=False, title="Average Amount Paid Per Card Type",
                                                    height=600, width=600, title_x=0.5,
                                                    xaxis_showgrid=False, yaxis_showgrid=False)

            fig10.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig10.update_yaxes(showticklabels=False)
            fig10.update_xaxes(categoryorder='total descending')
            
            st.write(fig10)

        with col23:
            # Visualize total amount paid per card type
            cardspend2 = df[['CardType', 'AmountPaid']]
            cardspend2 = cardspend2.groupby(['CardType']).sum()
            fig11 = px.bar(cardspend2, y='AmountPaid', text_auto='.2s')

            fig11.layout.update(yaxis_title='Total Spend', xaxis_title='Card Type', plot_bgcolor='rgba(0,0,0,0)',
                                                    showlegend=False, title="Total Amount Paid Per Card Type",
                                                    height=600, width=600, title_x=0.5,
                                                    xaxis_showgrid=False, yaxis_showgrid=False)

            fig11.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig11.update_yaxes(showticklabels=False)
            fig11.update_xaxes(categoryorder='total descending')

            st.write(fig11)

        st.write('')  #Line space but not markdown since it affects page scroll

        # Check highest amount paid per card type with all respective details
        with st.expander('*Transactions with Highest Amount Paid per Card Type*'):
            st.dataframe(df.loc[df.groupby(['CardType'])['AmountPaid'].idxmax()].reset_index(drop=True))

        # Assess Customer Lifetime Value (CLV)
        with st.expander('*Customer Lifetime Value Assessment*'):
            customer = pd.read_csv("customer.csv")
            cards = pd.read_csv("cards.csv")
            st.write('- Each customer spent an average of', customer['TotalSpent'].mean().round(1), 'KWD since the program launch.')
            st.write('- Customers earned an average of', customer['PointsRewarded'].mean().round(1), 'Points through card payments.')
            st.write('- They redeemed an average of', customer['PointsRedeemed'].mean().round(1), 'Points against available rewards.')
            st.write('- One customer typically has an average of ', cards['TotalCards'].mean().round(1) , 'Total Cards with KFH')
        
        st.write('')  #Line space but not markdown since it affects page scroll

        ## Interpretation of visuals
        st.info('Key Takeaways:')
        # idx() returns the index/label of the associated value
        st.write('(1) It looks like most transactions are made by the ', values.idxmax(), 'tier, with a total of ',
                values.max(),'transactions. In contrast, only ', values.min(), 'transactions come from the ', values.idxmin(), 'tier, which is \
                normal given its exclusivity.')

        # Create variables to make the sentence numbers dynamic
        partner_v = df['Partner'].value_counts()
        partner_p = df['Partner'].value_counts(normalize=True)
        multiplier_v = df['PointsMultiplier'].value_counts()
        cards_v = df['CardType'].value_counts()
        cards_p = df['CardType'].value_counts(normalize=True)

        st.write('(2) ', "{:.1%}".format(partner_p.max()), 'of transactions occur at Non-Partner Stores, with a total number \
        of ', partner_v.max(), 'transactions.')

        st.write('(3) The average spend per tier (in Kuwaiti Dinar) is the shown below. The results are as expected, \
        given that loyal customers spend more as they reach higher tiers.', tierspend)

        st.write('(4) Interestingly, the same cannot be said about the total spend per tier.\
                The Silver tier generated the highest spend, whilst the Black tier came in last.', tierspend2)

        st.write('(5) The most frequently used Points Multiplier is ', multiplier_v.idxmax(), ', with ', multiplier_v.max(), ' transactions.')

        st.write('(6) ', "{:.0%}".format(cards_p.max()), 'of payments (or ', cards_v.max(), 'transactions) are made using Visa Platinum Select.')
 
## STATISTICS

if option == 'Statistics':
    @st.cache()
    def load_visuals():
        df = pd.read_csv("kfh_visuals_zip.csv", compression='gzip')
        return df
    df = load_visuals()
    
    # Source: https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/
    st.markdown('#')  # Works as a separator for more visual space
    st.subheader('Distribution Overview & Transformation')
    warnings.filterwarnings('ignore')

    # Align the 3 graphs side-by-side
    with st.container():
        col11, col12, col13= st.columns([1,1,1])

    # Plot the original Distribution plot for the output variable
    ## Imported as image from Python file to avoid long execution
    with col11:
        from PIL import Image
        image = Image.open('original.png')
        st.image(image)

    # Plot the Distribution after removing outliers
    ## Imported as image from Python file to avoid long execution
    with col12:
        from PIL import Image
        image = Image.open('outliers.png')
        st.image(image)

    # Plot the Distribution after Yeo-Johnson transformation
    ## Imported as image from Python file to avoid long execution
    with col13:
        from PIL import Image
        image = Image.open('transformed.png')
        st.image(image)

    # Provide insights using HTML
    with st.expander('', expanded=True):
        outliers = df[(df['AmountPaid'] > 285.8335408276428) | (df['AmountPaid'] < -243.94050894451166)]
        total_outliers = outliers.shape[0]
        post_outliers = df['AmountPaid'].shape[0]
        st.markdown(f'''
            ##### Insights 💡
            <ul style="padding-left:20px">
              <li>Spend Distribution had 2 major problems: <b>Outliers</b> and <b>Skewness</b></li>
              <li><u>Total Number of Outliers:</u> <mark class="blue">{total_outliers}</mark></li>
              <li><b>Significant improvement</b> detected after removing outliers</li>
              <li>Normal distribution achieved through <b>Yeo-Johnson Transformaton</b></li>
            </ul>
            ''', unsafe_allow_html=True)

    st.markdown('#')  # Works as a separator for more visual space
    st.subheader('Machine Learning Algorithm Selection')

    # Load ML Algorithms comparison tables
    with st.expander('Performance Evaluation of Models with Transformed Features using MAE', expanded=True):
        comparison3 = pd.read_csv('comparison3.csv')
        st.dataframe(comparison3)

    with st.expander('Performance Evaluation of Models with Transformed Features using RMSE', expanded=True):
        comparison4 = pd.read_csv('comparison4.csv')
        st.dataframe(comparison4)

    with st.expander('Model Performance after Regularization using MAE', expanded=True):
        comparison5 = pd.read_csv('comparison5.csv')
        st.dataframe(comparison5)

    with st.expander('Model Performance after Regularization using RMSE', expanded=True):
        comparison6 = pd.read_csv('comparison6.csv')
        st.dataframe(comparison6)

    # Provide insights using HTML
    with st.expander('', expanded=True):
        st.markdown(f'''
            ##### Insights 💡
            <ul style="padding-left:20px">
              <li><b>eXtreme Gradient Boosting Regressor</b> is the selected algorithm for the prediction tool</mark</li>
              <li>It returned the <b>lowest Mean Absolute Error & Root Mean Squared Error</b> (lower = better)</li>
              <li>With <b>Hyperparameter Tuning,</b> the MAE is reduced to <b>0.97 KWD</b> and the RMSE to <b>1.15 KWD</b></li>
            </ul>
            ''', unsafe_allow_html=True)

    st.markdown('#')  #Works as a separator for more visual space
    st.subheader('Feature Importance')

    # Visualize importance of each feature according to the selected model
    ## Imported as image from Python file to avoid long execution
    from PIL import Image
    image = Image.open('importance.png')
    st.image(image, caption='Feature Importance according to the Model')

    # Provide insights using HTML
    with st.expander('', expanded=True):
        total_outliers = outliers.shape[0]
        post_outliers = df['AmountPaid'].shape[0]
        st.markdown(f'''
            ##### Insights 💡
            <ul style="padding-left:20px">
              <li><b>Merchant Category Code</b> is the feature with the strongest influence on Amount Paid</mark</li>
              <li><u>Points Multiplier</u><b> does not strongly affect the Amount Paid</b></li>
              <li><u>Top 3 Merchant Categories</u> are Miscellaneous, Clothing, And Retail Outlet Stores</li>
              <li>Future campaigns should test the impact of top features on spend</li>
              <li>Findings <b><u>contradict</u></b> the current significant weight given to Points Multiplier</li>
            </ul>
            ''', unsafe_allow_html=True)

## MACHINE LEARNING
# If 'Predictions' is pressed in the main tab
if option == 'Predictions':
    st.markdown('#')  # Works as a separator for more visual space
    st.subheader('Card Spend Prediction Tool')
    
    # Import GIF from Giphy using Markdown + align to center + resize
    st.markdown(f'''
            <p align="center">
                <img src="https://cdn.dribbble.com/users/2514124/screenshots/6617938/800x600.gif" alt="prediction" width="350" height="250"/>
             </p>
            ''', unsafe_allow_html=True)

    # Define functions to deal with labeled and one-hot encoded variables during selection
    @st.cache(suppress_st_warning=True)
    def get_fvalue(val):
        feature_dict = {"No":1,"Yes":2}
        for key,value in feature_dict.items():
            if val == key:
                return value

    def get_value(val,my_dict):
        for key,value in my_dict.items():
            if val == key:
                return value

    # Load dataset
    @st.cache()
    def load_visuals():
        df = pd.read_csv("kfh_visuals_zip.csv", compression='gzip')
        return df
    df = load_visuals()

    # Split the dataset into train and test to avoid data leakage
    X = df.drop('AmountPaid', axis=1)
    y = df['AmountPaid']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=1207)

    # Apply Yeo-Johnson Transformation (after split to avoid data leakage)
    # Source: https://gist.github.com/ShamimaMoni/e06883bd0a6c0d94a208d3fa7ba5b18c
    from scipy import stats
    transformed_train, lmbda = stats.yeojohnson(y_train)
    transformed_test, lmbda = stats.yeojohnson(y_test)

    # Revert back to original names to avoid confusion
    y_train = pd.DataFrame(transformed_train)
    y_test = pd.DataFrame(transformed_test)

    # Create dictionary values for certain features
    partner_dict = {'No': 0, 'Yes': 1}
    multiplier_dict = {'0x': 0, '1x': 1,
                       '1.5x': 1.5, '5x': 5, '10x': 10, '15x': 15}
    quarter_dict = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    tier_dict = {'Black': 1, 'Green': 2, 'Silver': 3}
    marketing_type_dict = {'Base': 1, 'Campaign': 2}
    
    # Create dropdown select boxes for each feature
    # Place them in containers, two per line
    with st.container():
        col14, col15= st.columns([1,1])
        with col14:
            ispartner = st.selectbox('Partner?', tuple(partner_dict.keys()), key=2)
        with col15:
            multiplier = st.selectbox('Select Points Multiplier', tuple(multiplier_dict.keys()), key=3)

    with st.container():
        col16, col17= st.columns([1,1])
        with col16:
            quarter = st.selectbox('Select Quarter', tuple(quarter_dict.keys()), key=4)
        with col17:
            cardtype = st.selectbox('Select Card Type', ['Hesabi prepaid2', 'MasterCard Classic',
                                                 'MasterCard Gold', 'MasterCard Platinum Premium',
                                                 'Tayseer T12 Classic', 'Tayseer T12 Gold', 'Tayseer T12 Platinum Premium',
                                                 'Tayseer T12 Platinum Standard', 'Tayseer T3 Classic', 'Tayseer T3 Gold',
                                                 'Visa Charge Diamond', 'Visa Classic', 'Visa Gold',
                                                 'Visa Infinite Metal Black', 'Visa Infinite Metal Silver',
                                                 'Visa Infinite Metal Veneer', 'Visa Platinum Premium',
                                                 'Visa Platinum Select', 'Visa Tayseer Diamond'], key=9)

    with st.container():
        col18, col19= st.columns([1,1])
        with col18:
            tier = st.selectbox('Select Tier', tuple(tier_dict.keys()), key=5)
        with col19:
            marketing = st.selectbox('Campaign?', tuple(marketing_type_dict.keys()), key=6)

    with st.container():
        col20, col21= st.columns([1,1])
        with col20:
            continent = st.selectbox('Select Continent', ['Africa', 'Asia', 'Europe',
                        'North America', 'Oceania', 'South America'], key=7)
        with col21:
            mcc = st.selectbox('Select Merchant Category', ['Agricultural Services', 'Airlines',
                        'Business Services', 'Car Rental', 'Clothing Stores',
                        'Contracted Services', 'Government Services', 'Lodging',
                        'Miscellaneous Stores', 'Professional Services',
                        'Retail Outlet Services', 'Transportation Services','Utility Services'], key=8)

    # Connect each possible selection to its respective feature
    card_0, card_1, card_2, card_3, card_4, card_5, card_6, card_7, card_8, card_9, card_10, card_11, card_12, card_13, card_14, card_15, card_16, card_17, card_18 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    if cardtype == 'Hesabi prepaid2':
        card_0 = 1
    elif cardtype == 'MasterCard Classic':
        card_1 = 1
    elif cardtype == 'MasterCard Gold':
        card_2 = 1
    elif cardtype == 'MasterCard Platinum Premium':
        card_3 = 1
    elif cardtype == 'Tayseer T12 Classic':
        card_4 = 1
    elif cardtype == 'Tayseer T12 Gold':
        card_5 = 1
    elif cardtype == 'Tayseer T12 Platinum Premium':
        card_6 = 1
    elif cardtype == 'Tayseer T12 Platinum Standard':
        card_7 = 1
    elif cardtype == 'Tayseer T3 Classic':
        card_8 = 1
    elif cardtype == 'Tayseer T3 Gold':
        card_9 = 1
    elif cardtype == 'Visa Charge Diamond':
        card_10 = 1
    elif cardtype == 'Visa Classic':
        card_11 = 1
    elif cardtype == 'Visa Gold':
        card_12 = 1
    elif cardtype == 'Visa Infinite Metal Black':
        card_13 = 1
    elif cardtype == 'Visa Infinite Metal Silver':
        card_14 = 1
    elif cardtype == 'Visa Infinite Metal Veneer':
        card_15 = 1
    elif cardtype == 'Visa Platinum Premium':
        card_16 = 1
    elif cardtype == 'Visa Platinum Select':
        card_17 = 1
    elif cardtype =='Visa Tayseer Diamond':
        card_18 = 1

    mcc_0, mcc_1, mcc_2, mcc_3, mcc_4, mcc_5, mcc_6, mcc_7, mcc_8, mcc9, mcc10, mcc11, mcc12 = 0,0,0,0,0,0,0,0,0,0,0,0,0
    if mcc == 'Agricultural Services':
        mcc_0 = 1
    elif mcc == 'Airlines':
        mcc_1 = 1
    elif mcc == 'Business Services':
        mcc_2 = 1
    elif mcc == 'Car Rental':
        mcc_3 = 1
    elif mcc == 'Clothing Stores':
        mcc_4 = 1
    elif mcc == 'Contracted Services':
        mcc_5 = 1
    elif mcc == 'Government Services':
        mcc_6 = 1
    elif mcc == 'Lodging':
        mcc_7 = 1
    elif mcc == 'Miscellaneous Stores':
        mcc_8 = 1
    elif mcc == 'Professional Services':
        mcc_9 = 1
    elif mcc == 'Retail Outlet Services':
        mcc_10 = 1
    elif mcc == 'Transportation Services':
        mcc_11 = 1
    elif mcc == 'Utility Services':
        mcc_12 = 1

    tier_0, tier_1, tier_2 = 0,0,0
    if tier == 'Black' :
        tier_0 = 1
    elif tier == 'Green' :
        tier_1 = 1
    elif tier == 'Silver':
        tier_2 = 1

    continent_0, continent_1, continent_2, continent_3, continent_4, continent_5 = 0,0,0,0,0,0
    if continent == 'Africa':
        continent_0 = 1
    elif continent == 'Asia':
        continent_1 = 1
    elif continent == 'Europe':
        continent_2 = 1
    elif continent == 'North America':
        continent_3 = 1
    elif continent == 'Oceania':
        continent_4 = 1
    elif continent == 'South America':
        continent_5 = 1

    marketing_0, marketing_1 = 0,0
    if marketing == 'Base':
        marketing_0 = 1
    if marketing == 'Campaign':
        marketing_1 = 1

    # Place them inside function to unify selections in a dataframe
    # Source: https://www.datacamp.com/tutorial/streamlit
    data1 = {'Partner?':ispartner,
            'Points Multiplier':multiplier,
            'Quarter':quarter,
            'Card Type':[card_0, card_1, card_2, card_3, card_4, card_5, card_6, card_7,card_8, card_9, card_10, card_11, card_12, card_13, card_14, card_15, card_16, card_17, card_18],
            'Tier':[tier_0, tier_1, tier_2],
            'Campaign?':[marketing_0, marketing_1],
            'Continent':[continent_0, continent_1, continent_2, continent_3, continent_4, continent_5],
            'Merchant Category':[mcc_0, mcc_1, mcc_2, mcc_3, mcc_4, mcc_5, mcc_6, mcc_7, mcc_8, mcc9, mcc10, mcc11, mcc12]
            }

    feature_list=[get_fvalue(ispartner),get_value(multiplier,multiplier_dict),get_value(quarter,quarter_dict),
                 data1['Card Type'][0],data1['Card Type'][1],data1['Card Type'][2],data1['Card Type'][3],
                 data1['Card Type'][4],data1['Card Type'][5],data1['Card Type'][6],data1['Card Type'][7],
                 data1['Card Type'][8],data1['Card Type'][9],data1['Card Type'][10],data1['Card Type'][11],
                 data1['Card Type'][12],data1['Card Type'][13],data1['Card Type'][14],data1['Card Type'][15],
                 data1['Card Type'][16],data1['Card Type'][17],data1['Card Type'][18],data1['Tier'][0],
                 data1['Tier'][1],data1['Tier'][2],data1['Campaign?'][0],data1['Campaign?'][1],
                 data1['Continent'][0],data1['Continent'][1],data1['Continent'][2],data1['Continent'][3],
                 data1['Continent'][4],data1['Continent'][5],data1['Merchant Category'][0],data1['Merchant Category'][1],
                 data1['Merchant Category'][2],data1['Merchant Category'][3],data1['Merchant Category'][4],data1['Merchant Category'][5],
                 data1['Merchant Category'][6],data1['Merchant Category'][7],data1['Merchant Category'][8],data1['Merchant Category'][9],
                 data1['Merchant Category'][10],data1['Merchant Category'][11],data1['Merchant Category'][12]]

    # Merge and reshape into a single sample
    single_sample = pd.DataFrame(feature_list)
    single_sample = np.array(feature_list).reshape(1,-1)

    # Create button that initiates the prediction of Amount Paid based on selected features
    if st.button('Predict Customer Spend'):
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model("xgb.model")
        prediction = model.predict(single_sample)

        # Define a function that inverts the Yeo-Jhonson transformation to get the values in KWD
        def invert_yeojhonson(value, lmbda):
          if value>= 0 and lmbda == 0:
            return math.exp(value) - 1
          elif value >= 0 and lmbda != 0:
            return (value * lmbda + 1) ** (1 / lmbda) - 1
          elif value < 0 and lmbda != 2:
            return 1 - (-(2 - lmbda) * value + 1) ** (1 / (2 - lmbda))
          elif value < 0 and lmbda == 2:
            return 1 - math.exp(-value)

        # Apply the invert function
        inverted = pd.DataFrame([invert_yeojhonson(x, lmbda) for x in prediction])
        st.write('')
        st.write('##### This customer will spend', '{:.2f}'.format(inverted.iloc[0,0]), ' Kuwaiti Dinars')
