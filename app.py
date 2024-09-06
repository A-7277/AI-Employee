import streamlit as st
import pandas as pd
import helper


data=st.sidebar.file_uploader('Upload your file Here!')
if data is not None:
  try:
    df=pd.read_csv(data)
  
  except Exception as e:
        try:
            df = pd.read_excel(data)
           
        except Exception as e:
            try:
              df = pd.read_excel(data)
          
            except Exception as e:
               st.write(f"Error loading Excel file: {e}")
  # st.dataframe(df)
  
  df=helper.preprocess(df)
  
  #Machine learning statitic algorithm - 1
  summary_stats=helper.describe_statistic(df)
  
  #EDA
  helper.top_5_graph(df)
  helper.pie_(df)
  helper.corr_(df)
  df=helper.kmeans(df)
  helper.kmeans_graph(df)
  
  acc=helper.linearn_regression(df)
  
  pdf=helper.pdf_gen(df,summary_stats,acc)
  
  pdf_output = pdf.output(dest='S').encode('latin1')

  # Add a download button in Streamlit
  st.sidebar.title('Download your Data-Analysis report From Here!')
  st.sidebar.download_button(
      label="Download PDF",
      data=pdf_output,
      file_name="Data_analysis_report.pdf",
      mime="application/pdf"
  )
  # st.dataframe(df)
  engine=helper.df_to_sql_db(df)
  # if 'conversation' not in st.session_state:
  #   st.session_state.conversation = []
  option = st.selectbox(
    "Select your Query Category",
    ("Is Query about provided data?", "General knowledge Query"),)

  st.write("You selected:", option) 

  prompt = st.chat_input("Ask your Query Here!")
  if prompt:
      
      # st.session_state.conversation.append(f"You: {user_input}")
    
      helper.sql_llm_google_palm(prompt,option,engine)
      # st.session_state.conversation.append(f"AI: {bot_response}")
      # st.subheader(f"AI-employee:{bot_response}")

  # for chat in st.session_state.conversation:
  #     st.write(chat)
      
  
  
  
         