import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fpdf import FPDF
import numpy as np
from langchain.llms import GooglePalm
from sqlalchemy import create_engine
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import streamlit as st



def preprocess(df):
  df.rename(columns={'Rank':'Ranks'},inplace=True)
  df.dropna()
  df.drop_duplicates()
  return df

def top_5_graph(df):
  top_countries = df.sort_values('Total', ascending=False).head(10)

  plt.figure(figsize=(10, 6))
  sns.barplot(x='Total', y='Country', data=top_countries)
  plt.title('Top 10 Countries by Total Medals')
  plt.xlabel('Total Medals')
  plt.ylabel('Country')
  plt.savefig('top.png')
  plt.show()
  plt.close()
  
def pie_(df):
  top_5_countries = df.sort_values('Total', ascending=False).head(5)
  plt.figure(figsize=(8, 8))
  plt.pie(top_5_countries['Total'], labels=top_5_countries['Country'], autopct='%1.1f%%', startangle=140)
  plt.title('Medal Distribution Among Top 5 Countries')
  plt.axis('equal')
  plt.savefig('pie.png')
  plt.show()
  plt.close()  

def corr_(df):
  corr = df[['Gold', 'Silver', 'Bronze', 'Total']].corr()

  plt.figure(figsize=(8, 6))
  sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Correlation Matrix of Medals')
  plt.savefig('corr_.png')
  plt.show()
  plt.close()  
  
def kmeans(df):
  
  X_ = df[['Gold', 'Silver', 'Bronze', 'Total']]
  kmeans = KMeans(n_clusters=3)
  df['Cluster'] = kmeans.fit_predict(X_)
  return df
  
def kmeans_graph(df):  
  plt.figure(figsize=(8, 6))
  plt.scatter(df['Gold'], df['Silver'], c=df['Cluster'], cmap='viridis', marker='o')
  plt.title('K-Means Clustering of Countries based on Medal Counts')
  plt.xlabel('Gold Medals')
  plt.ylabel('Silver Medals')
  plt.savefig('kmeans.png')
  plt.show()

  plt.close()

def describe_statistic(df):
  summary_stats = df[['Gold', 'Silver', 'Bronze', 'Total']].describe()
  summary_stats.index.rename('Measures',inplace=True)
  summary_stats=summary_stats.round(2)
  summary_stats.reset_index(inplace=True)
  return summary_stats    
  
  

def linearn_regression(df):
  model=LinearRegression()  
  x_=df.iloc[:,3:-2]
  y_=df.iloc[:,-2]
  ss=StandardScaler()
  x_s=ss.fit_transform(x_)
  X_train, X_test, y_train, y_test = train_test_split(x_s, y_, test_size=0.20, random_state=42)
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)  
  y_pred=y_pred.round(0).astype(int)
  acc=accuracy_score(y_test,y_pred)
  y_test1=np.array(y_test)
  plt.plot(y_test1, label='Actual Values', color='blue', marker='o')  
  plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--', marker='x') 
  plt.savefig('lr.png')
  plt.show()
  return acc

def df_to_pdf(pdf, dataframe):
    pdf.set_font('Arial','', 10)
    for col in dataframe.columns:
        pdf.cell(40, 10, col, 1)
    pdf.ln()  

    for i in range(len(dataframe)):
        for col in dataframe.columns:
            pdf.cell(40, 10, str(dataframe.iloc[i][col]), 1)
        pdf.ln()

def df1_to_pdf(pdf, dataframe):
    pdf.set_font('Arial','', 10)
    for col in dataframe.columns:
        pdf.cell(30, 10, col, 1)
    pdf.ln()  

    for i in range(len(dataframe)):
        for col in dataframe.columns:
            pdf.cell(30, 10, str(dataframe.iloc[i][col]), 1)
        pdf.ln()

def pdf_gen(df,summary_stats,acc):
  pdf = FPDF()
  pdf.add_page()
  pdf.set_font('Arial', 'B', 16)
  pdf.cell(200, 15, 'AI-Employee',1,0,'C')
  pdf.ln(20)
  pdf.cell(50, 10, 'Data - Analysis')
  pdf.ln(10)
  df_to_pdf(pdf,summary_stats)
  pdf.ln(10)
  pdf.set_font('Arial', 'B', 16)
  pdf.cell(200,10,'EDA',align='C')
  pdf.image('top.png',10,150,200)
  pdf.add_page()
  pdf.image('corr_.png',10,5,200)
  pdf.image('pie.png',10,140,170)
  pdf.add_page()
  pdf.cell(100,10,'Using K-means',10,0,'C')
  pdf.image('kmeans.png',10,20,150)
  pdf.ln(130)
  pdf.cell(200,10,'Clusters')
  pdf.ln(10)
  df1_to_pdf(pdf,df[df['Cluster']==2].head(6))
  pdf.ln(5)
  df1_to_pdf(pdf,df[df['Cluster']==1].head(6))
  pdf.ln(5)
  df1_to_pdf(pdf,df[df['Cluster']==0].head(6))
  pdf.add_page()
  pdf.set_font('Arial', 'B', 16)
  pdf.cell(200,10,'Linear-Regression',10,0,'C')
  pdf.image('lr.png',10,20,180)
  pdf.set_font('Arial', '', 13)
  pdf.ln(150)
  pdf.cell(200,10,f'Accuracy-{acc}')
  pdf.ln(10)
  pdf.cell(100,10,'Summary-')
  pdf.set_font('Arial', 'B', 10)
  pdf.ln(10)
  pdf.cell(200,10,f"The Country with highest no. of medals {df['Country'].head(1)[0]}")
  pdf.ln(7)
  pdf.cell(200,10,f"The Country with lowest no. of medals {df.iloc[-1,1]}")


  # pdf.output('tt.pdf', 'F')
  return pdf

def df_to_sql_db(df):
  engine =create_engine('mysql+pymysql://root:root@localhost:3306/ai_employee')  
  df.to_sql('olympics',con=engine,if_exists='replace',index=False)
  return engine

def sql_llm_google_palm(query,option,engine):
  with open("api_key.txt", "r") as file:
    api_key = file.read().strip()
  llm=GooglePalm(google_api_key=api_key,temperature=0.5)
  if option=="Is Query about provided data?":
    db=SQLDatabase(engine,sample_rows_in_table_info=3)
    sql_llm=SQLDatabaseChain.from_llm(llm ,db , verbose=True)
    ans=sql_llm.run(query)
    st.subheader(f"AI-employee:{ans}")
  else:  
    ans=llm.invoke(query)
    st.write(f"AI-employee:{ans}")
  return ans 
  
  

  

  
   