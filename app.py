import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff

st.set_page_config(page_title="Quantsistent - Identifique estratégias consistentes para seus trades", page_icon=":bar_chart:",layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

col100, col200, col300 = st.columns([1,1,1],gap='large')

with col100:   
    symbol = st.text_input('Ticker: ', value='AAPL')
    symbol = symbol.upper()
    st.session_state['symbol']=symbol

with col200:
    start_date = st.date_input('Data inicial', value=datetime(2023, 1, 1))
    st.session_state['start_date']=start_date
    
with col300:
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    end_date = st.date_input('Data final', value=today)
    end_date= end_date + timedelta(days=1)
    st.session_state['end_date'] = end_date


selected_page = option_menu(menu_title = None, options =['Estratégias Bull', 'Estratégias Sell'], icons=['graph-up-arrow', 'graph-down-arrow'], default_index=0, orientation="horizontal")    

if selected_page == 'Estratégias Bull':   
    def fetch_data(symbol, start_date, end_date):
        extensions = ["", ".SA", ".L", ".DE", ".TO", ".PA", ".AX",".T",".SS", ".NS", ".HK", ".SI"]
        country = ['US', 'Brazil', 'UK', 'Germany', 'Canada','France','Australia','Japan', 'China', 'India', 'Hong Kong', 'Singapore']
        counter = 0
        for extension in extensions:
            symbol_with_extension = symbol + extension
            data = yf.download(symbol_with_extension, start=start_date, end=end_date)
            if not data.empty:               
                st.session_state['data'] = data
                country_stock = country[counter]
                st.session_state['symbol']=symbol
                st.session_state['country_market'] = country_stock
                st.markdown(f"**País: {country_stock}    Ticker: {symbol}     De: {start_date}     Até: {end_date}**")
                break
            else:
                counter = counter+1
                #continue
        if not data.empty:
            #st.markdown("<h4 style='text-align: center; color: grey;'>Data downloaded! Explore the menu to access comprehensive insights and analysis tools for the asset you've selected using data science.</h4>", unsafe_allow_html=True)
            data_descending = st.session_state.data
            data_descending = data_descending.sort_index(ascending=False)
            #st.write(data_descending)
        else:
            data = None           
            st.session_state['data'] = data
            st.markdown("<h4 style='text-align: center; color: grey;'>Data was not downloaded. Check if the symbol you typed in is right.</h4>", unsafe_allow_html=True)                 
          
    fetch_data(symbol, start_date, end_date)

    
    
############ CM
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #006400; padding: 10px; border-radius: 0px;'>Cruzamento de médias</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: green; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)         
    #st.title('Cruzamento de médias')
       
    col0, col1 = st.columns([1,1],gap='large')
    col2, col3 = st.columns([1,1],gap='large')

    with col0:
      st.write(" ")
      st.write(" ")
      st.write("A estratégia se inicia na confirmação do fechamento do dia seguinte ao rompimento das médias selecionadas. Ela se encerra no cruzamento contrário das médias.")

      
        
      stock_data = st.session_state.data.copy()
    
      selected_short = st.slider('**Selecione a menor média:**', min_value=0, max_value=200, value=8, step=1)
      selected_long = st.slider('**Selecione a maior média:**', min_value=selected_short+1, max_value=200, value=21, step=1)
        
      # Calculating Exponential Moving Averages (EMA)
      stock_data['EMA_Short'] = stock_data['Close'].ewm(span=selected_short, adjust=False).mean()
      stock_data['EMA_Long'] = stock_data['Close'].ewm(span=selected_long, adjust=False).mean()
      
      # Condition for being above EMA
      stock_data['Above_EMA'] = (stock_data['EMA_Short'] > stock_data['EMA_Long'])
      
      # Trading strategy
      in_trade = False
      entry_price = 0
      total_return = 0
      trades_buy = []
      trades_sell = []
      trades_periodo = []
      three = 0
      trades_drawdown = []
      trades_high = []
      
      # Logic for trade execution
      for index, row in stock_data.iterrows():
          if row['Above_EMA'] and not in_trade:
              three += 1
              if three > 1:
                  in_trade = True
                  entry_price = row['Close']
                  trades_buy.append(entry_price)
                  three = 0
                  inicio = index
                  drawdown = row['Close']
                  highest =  row['Close']
          elif in_trade == True and row['High'] > highest:
              highest = row['High']
          elif in_trade == True and row['Low'] < drawdown:
              drawdown = row['Low']
          elif not row['Above_EMA'] and in_trade:
              in_trade = False
              exit_price = row['Close']
              trades_sell.append(exit_price)
              trade_return = (exit_price - entry_price) / entry_price
              total_return += trade_return
              fim = index
              periodo = fim - inicio
              trades_periodo.append(periodo)
              drawdown = ((drawdown/entry_price)-1)*100
              drawdown = round(drawdown, 2)
              trades_drawdown.append(drawdown)
              if exit_price > highest:
                  highest = exit_price 
              high = ((highest/entry_price)-1)*100
              high = round(high, 2)
              trades_high.append(high)
          else:
              three = 0
      
      # Adjusting buy/sell lists to match lengths
      if len(trades_buy) != len(trades_sell):
          trades_buy = trades_buy[:-1]
      
      # Creating DataFrame for trades
      trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                             'Drawdown': trades_drawdown, 'Max Return': trades_high})
      
      # Calculating returns and capital
      trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
      trades['Return'] = round(trades.Return, 2)
      return_list = trades['Return'].to_list()
      capital = 100
      for i in return_list:
          capital = capital + capital * (i / 100)
      capital = capital - 100
      capital = round(capital, 2)
      
      # Displaying results in Streamlit
    
            
      capital = 100
      total_return = 1
      evolution = []
    
      for index, r_value in trades['Return'].items():
          total_return *= 1 + (trades.loc[index, 'Return'])/100
          total_return_per = (total_return-1)*100
          evolution.append(total_return_per)
      global_r = (total_return - 1) * 100 
      global_r = round(global_r,2)
      st.markdown(f"<h5 style='text-align: left; color: grey;'>Retorno global das posições encerradas: {global_r} %</h5>", unsafe_allow_html=True)
    
      mediana = trades.Return.median()
      mediana = round(mediana, 2)
      st.write(f'**Retorno mediano por trade: {mediana}**')
    
    
    with col1:    
      fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
      fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
      st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

    with col2:
      fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
      fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
      st.plotly_chart(fig_combined, use_container_width=True)
    

    with col3:
      st.write('**Trades individuais**')
      st.dataframe(trades, use_container_width=True)



############ BB
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #006400; padding: 10px; border-radius: 0px;'>Bandas de Bollinger</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: green; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)         
  
    #st.title('Bandas de Bollinger')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)

    col4, col5 = st.columns([1,1],gap='large')
    col6, col7 = st.columns([1,1],gap='large')


    with col4:
      st.write(" ")
      st.write(" ")
      st.write("A estratégia se inicia quando o preço atinge o valor da banda inferior de Bollinger e se encerra quando o preço atinge o valor da banda superior.")
        
      stock_data = st.session_state.data.copy()
    
      window = st.slider('**Selecione a média da Banda de Bollinger:**', min_value=0, max_value=200, value=20, step=1)
      std_multiplier = 2
      distance = st.slider('**Selecione a diferença percentual entre as bandas:**', min_value=0, max_value=10, value=3, step=1)
        

      stock_data['MA'] = stock_data['Close'].rolling(window=window).mean()
      stock_data['std'] = stock_data['Close'].rolling(window=window).std()
    
      stock_data['Upper_BB'] = stock_data['MA'] + std_multiplier * stock_data['std']
      stock_data['Lower_BB'] = stock_data['MA'] - std_multiplier * stock_data['std']
    
      stock_data['Condition'] = (stock_data['Upper_BB']/stock_data['Lower_BB']) > (1 + (distance/100))


      in_trade = False
      entry_price = 0
      exit_price = 0
      total_return = 0

      trades_buy =[]
      trades_sell =[]
      trades_periodo=[]
      trades_drawdown = []
      trades_high = []

      for index, row in stock_data.iterrows():
          if not in_trade and row['Low'] <= row['Lower_BB'] and row['Condition']==True:
              in_trade = True
              entry_price = row['Close']
              entry_price=round(entry_price,2)
              trades_buy.append(entry_price)
              inicio = index
              drawdown = row['Close']
              highest =  row['Close']
          elif in_trade == True and row['High'] > highest:
              highest = row['High']
          elif in_trade == True and row['Low'] < drawdown:
              drawdown = row['Low']
          elif in_trade and row['High'] >= row['Upper_BB']:
              in_trade = False
              exit_price = row['Close']
              exit_price=round(exit_price,2)
              trades_sell.append(exit_price)
              fim = index
              periodo = fim - inicio
              trades_periodo.append(periodo)
              drawdown = ((drawdown/entry_price)-1)*100
              drawdown = round(drawdown, 2)
              trades_drawdown.append(drawdown)
              if exit_price > highest:
                  highest = exit_price 
              high = ((highest/entry_price)-1)*100
              high = round(high, 2)
              trades_high.append(high)

      if len(trades_buy) != len(trades_sell):
        trades_buy = trades_buy[:-1]
      
      # Creating DataFrame for trades
      trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                             'Drawdown': trades_drawdown, 'Max Return': trades_high})
      
      # Calculating returns and capital
      trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
      trades['Return'] = round(trades.Return, 2)
      return_list = trades['Return'].to_list()
      capital = 100
      for i in return_list:
          capital = capital + capital * (i / 100)
      capital = capital - 100
      capital = round(capital, 2)
      
      # Displaying results in Streamlit
    
            
      capital = 100
      total_return = 1
      evolution = []
    
      for index, r_value in trades['Return'].items():
          total_return *= 1 + (trades.loc[index, 'Return'])/100
          total_return_per = (total_return-1)*100
          evolution.append(total_return_per)
      global_r = (total_return - 1) * 100 
      global_r = round(global_r,2)
      st.markdown(f"<h5 style='text-align: left; color: grey;'>Retorno global das posições encerradas: {global_r} %</h5>", unsafe_allow_html=True)
    
      mediana = trades.Return.median()
      mediana = round(mediana, 2)
      st.write(f'**Retorno mediano por trade: {mediana}**')
    
    
    with col5:    
      fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
      fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
      st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

    with col6:
      fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
      fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
      st.plotly_chart(fig_combined, use_container_width=True)
    

    with col7:
      st.write('**Trades individuais**')
      st.dataframe(trades, use_container_width=True)


############ RM
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #006400; padding: 10px; border-radius: 0px;'>Retorno à média</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: green; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)         
    #st.title('Retorno a média')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col8, col9 = st.columns([1,1],gap='large')
    col10, col11 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col8:
          st.write(" ")
          st.write(" ")
          st.write("A estratégia se inicia quando o preço atinge o valor da banda inferior de Bollinger e se encerra quando o preço atinge o valor da média da Banda de Bollinger.")
                       
          stock_data = st.session_state.data.copy()
        
          media = st.slider('**Selecione a média da Banda de Bollinger:**', min_value=1, max_value=200, value=20, step=1)
          window = 20
          std_multiplier = 2
          distance_mr = st.slider('**Selecione a diferença percentual entre as bandas:**', min_value=0, max_value=10, value=3, step=1, key='slider1')
  
          stock_data['MEDIA'] = stock_data['Close'].rolling(window=media).mean()
    
          stock_data['MA'] = stock_data['Close'].rolling(window=window).mean()
          stock_data['std'] = stock_data['Close'].rolling(window=window).std()
    
          stock_data['Upper_BB'] = stock_data['MA'] + std_multiplier * stock_data['std']
          stock_data['Lower_BB'] = stock_data['MA'] - std_multiplier * stock_data['std']
    
          stock_data['Condition'] = (stock_data['Upper_BB']/stock_data['Lower_BB']) > (1+(distance_mr/100))
    
          in_trade = False
          entry_price = 0
          exit_price = 0
          total_return = 0
    
          trades_buy =[]
          trades_sell =[]
          trades_periodo=[]
          trades_drawdown = []
          trades_high=[]
    
          for index, row in stock_data.iterrows():
              if not in_trade and row['Low'] <= row['Lower_BB'] and row['Condition']==True:
                  in_trade = True
                  entry_price = row['Lower_BB']
                  entry_price=round(entry_price,2)
                  trades_buy.append(entry_price)
                  inicio = index
                  drawdown = row['Lower_BB']
                  highest =  row['Lower_BB']
              elif in_trade == True and row['High'] > highest:
                  highest = row['High']
              elif in_trade == True and row['Low'] < drawdown:
                  drawdown = row['Low']
              elif in_trade and row['High'] >= row['MEDIA']:
                  in_trade = False
                  exit_price = row['MEDIA']
                  exit_price=round(exit_price,2)
                  trades_sell.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)                       
                  drawdown = ((drawdown/entry_price)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price > highest:
                      highest = exit_price 
                  high = ((highest/entry_price)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)
    
          if len(trades_buy) != len(trades_sell):
            trades_buy = trades_buy[:-1]
          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
          
          # Displaying results in Streamlit
        
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Retorno global das posições encerradas: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Retorno mediano por trade: {mediana}**')
    

      with col9:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col10:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col11:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)    



############ RSI
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #006400; padding: 10px; border-radius: 0px;'>Índice de Força Relativa</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: green; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)      
    #st.title('Índice de Força Relativa')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col12, col13 = st.columns([1,1],gap='large')
    col14, col15 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col12:
          st.write(" ")
          st.write(" ")          
          st.write("A estratégia se inicia quando o Índice de Força Relativa (IFR ou RSI) atinge o valor mínimo definido nos parâmetros e se encerra quando o preço atinge o valor máximo definido.")

          stock_data = st.session_state.data.copy()
        
          window_length = st.slider('**Selecione a Média:**', min_value=0, max_value=200, value=14, step=1)
          window_high = st.slider('**Selecione o valor de sobrecompra do IFR:**', min_value=60, max_value=100, value=70, step=1)
          window_low = st.slider('**Selecione o valor de sobrevenda do IFR:**', min_value=0, max_value=40, value=30, step=1)

          delta = stock_data['Close'].diff()
          gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
          loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
          RS = gain / loss
          RSI = 100 - (100 / (1 + RS))
          stock_data['RSI'] = RSI

          in_trade = False
          entry_price = 0
          exit_price = 0
          total_return = 0

          trades_buy =[]
          trades_sell =[]
          trades_periodo=[]
          trades_drawdown = []
          trades_high= []

          for index, row in stock_data.iterrows():
              if not in_trade and row['RSI'] <= window_low:
                  in_trade = True
                  entry_price = row['Close']
                  entry_price=round(entry_price,2)
                  trades_buy.append(entry_price)
                  inicio = index
                  drawdown = row['Close']
                  highest =  row['Close']
              elif in_trade == True and row['High'] > highest:
                  highest = row['High']
              elif in_trade == True and row['Low'] < drawdown:
                  drawdown = row['Low']
              elif in_trade and row['RSI'] >= window_high:
                  in_trade = False
                  exit_price = row['Close']
                  exit_price=round(exit_price,2)
                  trades_sell.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)            
                  drawdown = ((drawdown/entry_price)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price > highest:
                      highest = exit_price 
                  high = ((highest/entry_price)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)

          if len(trades_buy) != len(trades_sell):
            trades_buy = trades_buy[:-1]
          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
          
          # Displaying results in Streamlit
        
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Retorno global das posições encerradas: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Retorno mediano por trade: {mediana}**')

      with col13:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col14:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col15:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)   



############ MACD
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #006400; padding: 10px; border-radius: 0px;'>MACD</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: green; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)      
    #st.title('MACD')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col16, col17 = st.columns([1,1],gap='large')
    col18, col19 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col16:
          st.write(" ")
          st.write(" ")
          st.write("The strategy involves using Exponential Moving Averages (EMAs) on the closing price and volume. Users can select the EMA values for both parameters using sliders. The strategy identifies whether the closing price is above the EMA and if the volume is also above the EMA. When the conditions are met, it executes a trade, calculating buy and sell points based on certain criteria for high and low values.")
            
          stock_data = st.session_state.data.copy()
        
          short_window = st.slider('**Selecione a média (EMA) curta:**', min_value=0, max_value=40, value=12, step=1)
          long_window = st.slider('**Selecione a média (EMA) longa:**', min_value=0, max_value=40, value=26, step=1)
          n_consecutive_true_count = st.slider('**Selecione o número de valores crescentes consecutivos do MACD:**', min_value=0, max_value=10, value=3, step=1)
          
          signal_window=9

          short_ema = stock_data['Close'].ewm(span=short_window, adjust=False).mean()
          long_ema = stock_data['Close'].ewm(span=long_window, adjust=False).mean()

          stock_data['short_ema'] = short_ema
          stock_data['long_ema'] = long_ema

          macd_line = short_ema - long_ema

          signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

          histogram = macd_line - signal_line

          stock_data['Histogram'] = histogram
          stock_data['Increases'] = stock_data['Histogram'].diff().gt(0)
          stock_data['MACD'] = macd_line
          stock_data['Signal_Line'] = signal_line


          entry_price = 0
          exit_price = 0
          total_return = 0


          consecutive_true_count = 0
          consecutive_false_count = 0
          in_trade = False
          trades_buy = []
          trades_sell = []
          trades_periodo = []
          trades_drawdown = []
          trades_high=[]

          for index, row in stock_data.iterrows():
              if row['Increases']:
                  consecutive_true_count += 1
                  consecutive_false_count = 0
              else:
                  consecutive_true_count = 0
                  consecutive_false_count += 1

              if consecutive_true_count == n_consecutive_true_count and not in_trade:
                  in_trade = True
                  entry_price = row['Close']
                  entry_price = round(entry_price, 2)
                  trades_buy.append(entry_price)
                  inicio = index
                  drawdown = row['Close']
                  highest =  row['Close']
              elif in_trade == True and row['High'] > highest:
                  highest = row['High']
              elif in_trade and row['Low'] < drawdown:
                  drawdown = row['Low']

              elif in_trade and consecutive_false_count == n_consecutive_true_count:
                  in_trade = False
                  exit_price = row['Close']
                  exit_price = round(exit_price, 2)
                  trades_sell.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)            
                  drawdown = ((drawdown/entry_price)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price > highest:
                      highest = exit_price 
                  high = ((highest/entry_price)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)

          if len(trades_buy) != len(trades_sell):
            trades_buy = trades_buy[:-1]
          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
          
          # Displaying results in Streamlit
        
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Retorno global das posições encerradas: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Retorno mediano por trade: {mediana}**')


      with col17:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col18:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col19:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)   



############ VOLUME
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #006400; padding: 10px; border-radius: 0px;'>Volume</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: green; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)      
    #st.title('Volume')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col20, col21 = st.columns([1,1],gap='large')
    col22, col23 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col20:
          st.write(" ")
          st.write(" ")
          st.write("The strategy involves using Exponential Moving Averages (EMAs) on the closing price and volume. Users can select the EMA values for both parameters using sliders. The strategy identifies whether the closing price is above the EMA and if the volume is also above the EMA. When the conditions are met, it executes a trade, calculating buy and sell points based on certain criteria for high and low values.")
            
          stock_data = st.session_state.data.copy()
        
          short_window = st.slider('**Selecione o valor da Média:**', min_value=0, max_value=200, value=20, step=1, key='slider_vol_short')          
          n_volume = st.slider('**Selecione o valor percentual de aumento de volume:**', min_value=0, max_value=100, value=50, step=1,key='slider_vol')
          
          stock_data['Volume_M'] = stock_data['Volume'].rolling(window=short_window, min_periods=1).mean()

          stock_data['Color'] = stock_data.apply(lambda row: 'Green' if row['Close'] > row['Open'] else 'Red', axis=1)

          stock_data['Condition'] = (stock_data['Volume'] >= (1+(n_volume/100)) * stock_data['Volume_M'])

          entry_price = 0
          exit_price = 0
          total_return = 0

          in_trade = False
          trades_buy = []
          trades_sell = []
          trades_periodo = []
          trades_drawdown = []
          trades_high = []

          for index, row in stock_data.iterrows():
              if not in_trade and row['Condition'] and row['Color'] == 'Green':
                  in_trade = True
                  entry_price = row['Close']
                  entry_price=round(entry_price,2)
                  trades_buy.append(entry_price)
                  inicio = index
                  drawdown = row['Close']
                  first_mean = row['Volume_M']
                  highest =  row['Close']
              elif in_trade == True and row['High'] > highest:
                  highest = row['High']
              elif in_trade and row['Low'] < drawdown:
                  drawdown = row['Low']
              elif in_trade and row['Volume'] < first_mean:
                  in_trade = False
                  exit_price = row['Close']
                  exit_price=round(exit_price,2)
                  trades_sell.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)
                  drawdown = ((drawdown/entry_price)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price > highest:
                      highest = exit_price 
                  high = ((highest/entry_price)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)

          if len(trades_buy) != len(trades_sell):
            trades_buy = trades_buy[:-1]
          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
          
          # Displaying results in Streamlit
        
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Retorno global das posições encerradas: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Retorno mediano por trade: {mediana}**')


      with col21:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col22:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col23:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)   


####################################################################################################
####################################################################################################
####################################################################################################
else:
    def fetch_data(symbol, start_date, end_date):
        extensions = ["", ".SA", ".L", ".DE", ".TO", ".PA", ".AX",".T",".SS", ".NS", ".HK", ".SI"]
        country = ['US', 'Brazil', 'UK', 'Germany', 'Canada','France','Australia','Japan', 'China', 'India', 'Hong Kong', 'Singapore']
        counter = 0
        for extension in extensions:
            symbol_with_extension = symbol + extension
            data = yf.download(symbol_with_extension, start=start_date, end=end_date)
            if not data.empty:               
                st.session_state['data'] = data
                st.markdown(f"**Ticker: {symbol} De: {start_date} Até: {end_date}**")
                country_stock = country[counter]
                st.session_state['symbol']=symbol
                st.session_state['country_market'] = country_stock
                st.markdown(f"**País: {country_stock}**")
                break
            else:
                counter = counter+1
                #continue
        if not data.empty:
            #st.markdown("<h4 style='text-align: center; color: grey;'>Data downloaded! Explore the menu to access comprehensive insights and analysis tools for the asset you've selected using data science.</h4>", unsafe_allow_html=True)
            data_descending = st.session_state.data
            data_descending = data_descending.sort_index(ascending=False)
            #st.write(data_descending)
        else:
            data = None           
            st.session_state['data'] = data
            st.markdown("<h4 style='text-align: center; color: grey;'>Data was not downloaded. Check if the symbol you typed in is right.</h4>", unsafe_allow_html=True)                 
          
    fetch_data(symbol, start_date, end_date)

    
    
############ CM
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #8B0000; padding: 10px; border-radius: 0px;'>Cruzamento de médias</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: #FF6347; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)      
    #st.title('Cruzamento de médias')
       
    col0, col1 = st.columns([1,1],gap='large')
    col2, col3 = st.columns([1,1],gap='large')

    with col0:
      st.write(" ")
      st.write(" ")      
      st.write("A estratégia se inicia na confirmação do fechamento do dia seguinte ao rompimento das médias selecionadas. Ela se encerra no cruzamento contrário das médias.")

      
        
      stock_data = st.session_state.data.copy()
    
      selected_short = st.slider('**Selecione a menor média:**', min_value=0, max_value=200, value=8, step=1)
      selected_long = st.slider('**Selecione a maior média:**', min_value=selected_short+1, max_value=200, value=21, step=1)
        
      # Calculating Exponential Moving Averages (EMA)
      stock_data['EMA_Short'] = stock_data['Close'].ewm(span=selected_short, adjust=False).mean()
      stock_data['EMA_Long'] = stock_data['Close'].ewm(span=selected_long, adjust=False).mean()
      
      # Condition for being above EMA
      stock_data['Condition'] = (stock_data['EMA_Short'] > stock_data['EMA_Long'])
      
      in_trade = False
      entry_price = 0
      total_return = 0
      trades_buy = []
      trades_sell = []
      trades_periodo = []
      three = 0
      trades_drawdown = []
      trades_high = []
      
      # Logic for trade execution
      for index, row in stock_data.iterrows():
          if row['Condition'] and not in_trade:
              three += 1
              if three > 1:
                  in_trade = True
                  entry_price = row['Close']
                  trades_sell.append(entry_price)
                  three = 0
                  inicio = index
                  drawdown = row['Close']
                  highest =  row['Close']
          elif in_trade == True and row['High'] > drawdown:
              drawdown = row['High']
          elif in_trade == True and row['Low'] < highest:
              highest = row['Low']
          elif not row['Condition'] and in_trade:
              in_trade = False
              exit_price = row['Close']
              trades_buy.append(exit_price)
              trade_return = (exit_price - entry_price) / entry_price
              total_return += trade_return
              fim = index
              periodo = fim - inicio
              trades_periodo.append(periodo)
              drawdown = ((entry_price/drawdown)-1)*100
              drawdown = round(drawdown, 2)
              trades_drawdown.append(drawdown)
              if exit_price < highest:
                  highest = exit_price 
              high = ((entry_price/highest)-1)*100
              high = round(high, 2)
              trades_high.append(high)
          else:
              three = 0
      
      # Adjusting buy/sell lists to match lengths
      if len(trades_buy) != len(trades_sell):
          trades_sell = trades_sell[:-1]
      
      # Creating DataFrame for trades
      trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                             'Drawdown': trades_drawdown, 'Max Return': trades_high})
      
      # Calculating returns and capital
      trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
      trades['Return'] = round(trades.Return, 2)
      return_list = trades['Return'].to_list()
      capital = 100
      for i in return_list:
          capital = capital + capital * (i / 100)
      capital = capital - 100
      capital = round(capital, 2)
            
      capital = 100
      total_return = 1
      evolution = []
    
      for index, r_value in trades['Return'].items():
          total_return *= 1 + (trades.loc[index, 'Return'])/100
          total_return_per = (total_return-1)*100
          evolution.append(total_return_per)
      global_r = (total_return - 1) * 100 
      global_r = round(global_r,2)
      st.markdown(f"<h5 style='text-align: left; color: grey;'>Global return of closed positions: {global_r} %</h5>", unsafe_allow_html=True)
    
      mediana = trades.Return.median()
      mediana = round(mediana, 2)
      st.write(f'**Median return per trade: {mediana}**')
    
    
    with col1:    
      fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
      fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
      st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

    with col2:
      fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
      fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
      st.plotly_chart(fig_combined, use_container_width=True)
    

    with col3:
      st.write('**Trades individuais**')
      st.dataframe(trades, use_container_width=True)



############ BB
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #8B0000; padding: 10px; border-radius: 0px;'>Bandas de Bollinger</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: #FF6347; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)      
    #st.title('Bandas de Bollinger')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)

    col4, col5 = st.columns([1,1],gap='large')
    col6, col7 = st.columns([1,1],gap='large')


    with col4:
      st.write(" ")
      st.write(" ")
      st.write("A estratégia se inicia quando o preço atinge o valor da banda inferior de Bollinger e se encerra quando o preço atinge o valor da banda superior.")
        
      stock_data = st.session_state.data.copy()
    
      window = st.slider('**Selecione a média da Banda de Bollinger:**', min_value=0, max_value=200, value=20, step=1)
      std_multiplier = 2
      distance = st.slider('**Selecione a diferença percentual entre as bandas:**', min_value=0, max_value=10, value=3, step=1)
        

      stock_data['MA'] = stock_data['Close'].rolling(window=window).mean()
      stock_data['std'] = stock_data['Close'].rolling(window=window).std()
    
      stock_data['Upper_BB'] = stock_data['MA'] + std_multiplier * stock_data['std']
      stock_data['Lower_BB'] = stock_data['MA'] - std_multiplier * stock_data['std']
    
      stock_data['Condition'] = (stock_data['Upper_BB']/stock_data['Lower_BB']) > (1 + (distance/100))


      in_trade = False
      entry_price = 0
      exit_price = 0
      total_return = 0

      trades_buy =[]
      trades_sell =[]
      trades_periodo=[]
      trades_drawdown = []
      trades_high = []

      for index, row in stock_data.iterrows():
          if not in_trade and row['High'] >= row['Upper_BB'] and row['Condition']==True:
              in_trade = True
              entry_price = row['Close']
              entry_price=round(entry_price,2)
              trades_sell.append(entry_price)
              inicio = index
              drawdown = row['Close']
              highest =  row['Close']
          elif in_trade == True and row['High'] > drawdown:
              drawdown = row['High']
          elif in_trade == True and row['Low'] < highest:
              highest = row['Low']
          elif in_trade and row['Low'] <= row['Lower_BB']:
              in_trade = False
              exit_price = row['Close']
              exit_price=round(exit_price,2)
              trades_buy.append(exit_price)
              fim = index
              periodo = fim - inicio
              trades_periodo.append(periodo)
              drawdown = ((entry_price/drawdown)-1)*100
              drawdown = round(drawdown, 2)
              trades_drawdown.append(drawdown)
              if exit_price < highest:
                  highest = exit_price 
              high = ((entry_price/highest)-1)*100
              high = round(high, 2)
              trades_high.append(high)



      if len(trades_buy) != len(trades_sell):
        trades_sell = trades_sell[:-1]

      
      # Creating DataFrame for trades
      trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                             'Drawdown': trades_drawdown, 'Max Return': trades_high})
      
      # Calculating returns and capital
      trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
      trades['Return'] = round(trades.Return, 2)
      return_list = trades['Return'].to_list()
      capital = 100
      for i in return_list:
          capital = capital + capital * (i / 100)
      capital = capital - 100
      capital = round(capital, 2)
            
      capital = 100
      total_return = 1
      evolution = []
    
      for index, r_value in trades['Return'].items():
          total_return *= 1 + (trades.loc[index, 'Return'])/100
          total_return_per = (total_return-1)*100
          evolution.append(total_return_per)
      global_r = (total_return - 1) * 100 
      global_r = round(global_r,2)
      st.markdown(f"<h5 style='text-align: left; color: grey;'>Global return of closed positions: {global_r} %</h5>", unsafe_allow_html=True)
    
      mediana = trades.Return.median()
      mediana = round(mediana, 2)
      st.write(f'**Median return per trade: {mediana}**')
    
    
    with col5:    
      fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
      fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
      st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

    with col6:
      fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
      fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
      st.plotly_chart(fig_combined, use_container_width=True)
    

    with col7:
      st.write('**Trades individuais**')
      st.dataframe(trades, use_container_width=True)


############ RM
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #8B0000; padding: 10px; border-radius: 0px;'>Retorno à média</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: #FF6347; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)     
    #st.title('Retorno a média')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col8, col9 = st.columns([1,1],gap='large')
    col10, col11 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col8:
          st.write(" ")
          st.write(" ")
          st.write("A estratégia se inicia quando o preço atinge o valor da banda inferior de Bollinger e se encerra quando o preço atinge o valor da média da Banda de Bollinger.")
                       
          stock_data = st.session_state.data.copy()
        
          media = st.slider('**Selecione a média da Banda de Bollinger:**', min_value=1, max_value=200, value=20, step=1)
          window = 20
          std_multiplier = 2
          distance_mr = st.slider('**Selecione a diferença percentual entre as bandas:**', min_value=0, max_value=10, value=3, step=1, key='slider1')
  
          stock_data['MEDIA'] = stock_data['Close'].rolling(window=media).mean()
    
          stock_data['MA'] = stock_data['Close'].rolling(window=window).mean()
          stock_data['std'] = stock_data['Close'].rolling(window=window).std()
    
          stock_data['Upper_BB'] = stock_data['MA'] + std_multiplier * stock_data['std']
          stock_data['Lower_BB'] = stock_data['MA'] - std_multiplier * stock_data['std']
    
          stock_data['Condition'] = (stock_data['Upper_BB']/stock_data['Lower_BB']) > (1+(distance_mr/100))
    
          in_trade = False
          entry_price = 0
          exit_price = 0
          total_return = 0

          trades_buy =[]
          trades_sell =[]
          trades_periodo=[]
          trades_drawdown = []
          trades_high=[]

          for index, row in stock_data.iterrows():
              if not in_trade and row['High'] >= row['Upper_BB'] and row['Condition']==True:
                  in_trade = True
                  entry_price = row['Upper_BB']
                  entry_price=round(entry_price,2)
                  trades_sell.append(entry_price)
                  inicio = index
                  drawdown = row['Upper_BB']
                  highest =  row['Upper_BB']
              elif in_trade == True and row['High'] > drawdown:
                  drawdown = row['High']
              elif in_trade == True and row['Low'] < highest:
                  highest = row['Low']
              elif in_trade and row['Low'] <= row['MEDIA']:
                  in_trade = False
                  exit_price = row['MEDIA']
                  exit_price=round(exit_price,2)
                  trades_buy.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)
                  drawdown = ((entry_price/drawdown)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price < highest:
                      highest = exit_price 
                  high = ((entry_price/highest)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)



          if len(trades_buy) != len(trades_sell):
            trades_sell = trades_sell[:-1]

          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Global return of closed positions: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Median return per trade: {mediana}**')

    

      with col9:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col10:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col11:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)    



############ RSI
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #8B0000; padding: 10px; border-radius: 0px;'>Índice de Força Relativa</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: #FF6347; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)     
    #st.title('')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col12, col13 = st.columns([1,1],gap='large')
    col14, col15 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col12:
          st.write(" ")
          st.write(" ")          
          st.write("A estratégia se inicia quando o Índice de Força Relativa (IFR ou RSI) atinge o valor mínimo definido nos parâmetros e se encerra quando o preço atinge o valor máximo definido.")

          stock_data = st.session_state.data.copy()
        
          window_length = st.slider('**Selecione a Média:**', min_value=0, max_value=200, value=14, step=1)
          window_high = st.slider('**Selecione o valor de sobrecompra do IFR:**', min_value=60, max_value=100, value=70, step=1)
          window_low = st.slider('**Selecione o valor de sobrevenda do IFR:**', min_value=0, max_value=40, value=30, step=1)

          delta = stock_data['Close'].diff()
          gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
          loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
          RS = gain / loss
          RSI = 100 - (100 / (1 + RS))
          stock_data['RSI'] = RSI

          in_trade = False
          entry_price = 0
          exit_price = 0
          total_return = 0

          trades_buy =[]
          trades_sell =[]
          trades_periodo=[]
          trades_drawdown = []
          trades_high= []

          for index, row in stock_data.iterrows():
              if not in_trade and row['RSI'] >= window_high:
                  in_trade = True
                  entry_price = row['Close']
                  entry_price=round(entry_price,2)
                  trades_sell.append(entry_price)
                  inicio = index
                  drawdown = row['Close']
                  highest =  row['Close']
              elif in_trade == True and row['High'] > drawdown:
                  drawdown = row['High']
              elif in_trade == True and row['Low'] < highest:
                  highest = row['Low']
              elif in_trade and row['RSI'] >= window_low:
                  in_trade = False
                  exit_price = row['Close']
                  exit_price=round(exit_price,2)
                  trades_buy.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)
                  drawdown = ((entry_price/drawdown)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price < highest:
                      highest = exit_price 
                  high = ((entry_price/highest)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)



          if len(trades_buy) != len(trades_sell):
            trades_sell = trades_sell[:-1]

          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Global return of closed positions: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Median return per trade: {mediana}**')

      with col13:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col14:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col15:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)   



############ MACD
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #8B0000; padding: 10px; border-radius: 0px;'>MACD</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: #FF6347; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)    
    #st.title('MACD')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col16, col17 = st.columns([1,1],gap='large')
    col18, col19 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col16:
          st.write(" ")
          st.write(" ")
          st.write("The strategy involves using Exponential Moving Averages (EMAs) on the closing price and volume. Users can select the EMA values for both parameters using sliders. The strategy identifies whether the closing price is above the EMA and if the volume is also above the EMA. When the conditions are met, it executes a trade, calculating buy and sell points based on certain criteria for high and low values.")
            
          stock_data = st.session_state.data.copy()
        
          short_window = st.slider('**Selecione a média (EMA) curta:**', min_value=0, max_value=40, value=12, step=1)
          long_window = st.slider('**Selecione a média (EMA) longa:**', min_value=0, max_value=40, value=26, step=1)
          n_consecutive_true_count = st.slider('**Selecione o número de valores crescentes consecutivos do MACD:**', min_value=0, max_value=10, value=3, step=1)
          
          signal_window=9

          short_ema = stock_data['Close'].ewm(span=short_window, adjust=False).mean()
          long_ema = stock_data['Close'].ewm(span=long_window, adjust=False).mean()

          stock_data['short_ema'] = short_ema
          stock_data['long_ema'] = long_ema

          macd_line = short_ema - long_ema

          signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

          histogram = macd_line - signal_line

          stock_data['Histogram'] = histogram
          stock_data['Increases'] = stock_data['Histogram'].diff().gt(0)
          stock_data['MACD'] = macd_line
          stock_data['Signal_Line'] = signal_line


          entry_price = 0
          exit_price = 0
          total_return = 0


          consecutive_true_count = 0
          consecutive_false_count = 0
          in_trade = False
          trades_buy = []
          trades_sell = []
          trades_periodo = []
          trades_drawdown = []
          trades_high=[]

          for index, row in stock_data.iterrows():
              if row['Increases']:
                  consecutive_true_count += 1
                  consecutive_false_count = 0
              else:
                  consecutive_true_count = 0
                  consecutive_false_count += 1
              if consecutive_false_count == n_consecutive_true_count and not in_trade:
                  in_trade = True
                  entry_price = row['Close']
                  entry_price = round(entry_price, 2)
                  trades_sell.append(entry_price)
                  inicio = index
                  drawdown = row['Close']
                  highest =  row['Close']
              elif in_trade == True and row['High'] > drawdown:
                  drawdown = row['High']
              elif in_trade and row['Low'] < highest:
                  highest = row['Low']
              elif in_trade and consecutive_true_count == n_consecutive_true_count:
                  in_trade = False
                  exit_price = row['Close']
                  exit_price = round(exit_price, 2)
                  trades_buy.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)            
                  drawdown = ((entry_price/drawdown)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price < highest:
                      highest = exit_price 
                  high = ((entry_price/highest)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)



          if len(trades_buy) != len(trades_sell):
            trades_sell = trades_sell[:-1]

          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Global return of closed positions: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Median return per trade: {mediana}**')


      with col17:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col18:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col19:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)   



############ VOLUME
    
    st.markdown(f"<h2 style='text-align: left; color: white; background-color: #8B0000; padding: 10px; border-radius: 0px;'>Volume</h2>", unsafe_allow_html=True)         
    st.markdown(f"<h6 style='text-align: left; color: white; background-color: #FF6347; padding: 10px; border-radius: 0px;'></h6>", unsafe_allow_html=True)    
    ##st.title('Volume')

    symbol = st.session_state.symbol
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    end_date = end_date - timedelta(days=1)
    
    col20, col21 = st.columns([1,1],gap='large')
    col22, col23 = st.columns([1,1],gap='large')

    if selected_page != "Sell":

      # Partition 1
      with col20:
          st.write(" ")
          st.write(" ")
          st.write("The strategy involves using Exponential Moving Averages (EMAs) on the closing price and volume. Users can select the EMA values for both parameters using sliders. The strategy identifies whether the closing price is above the EMA and if the volume is also above the EMA. When the conditions are met, it executes a trade, calculating buy and sell points based on certain criteria for high and low values.")
            
          stock_data = st.session_state.data.copy()
        
          short_window = st.slider('**Selecione o valor da Média:**', min_value=0, max_value=200, value=20, step=1, key='slider_vol_short')          
          n_volume = st.slider('**Selecione o valor percentual de aumento de volume:**', min_value=0, max_value=100, value=50, step=1,key='slider_vol')
          
          stock_data['Volume_M'] = stock_data['Volume'].rolling(window=short_window, min_periods=1).mean()

          stock_data['Color'] = stock_data.apply(lambda row: 'Green' if row['Close'] > row['Open'] else 'Red', axis=1)

          stock_data['Condition'] = (stock_data['Volume'] >= (1+(n_volume/100)) * stock_data['Volume_M'])

          entry_price = 0
          exit_price = 0
          total_return = 0

          in_trade = False
          trades_buy = []
          trades_sell = []
          trades_periodo = []
          trades_drawdown = []
          trades_high = []

          for index, row in stock_data.iterrows():
              if not in_trade and row['Condition'] and row['Color'] == 'Red':
                  in_trade = True
                  entry_price = row['Close']
                  entry_price=round(entry_price,2)
                  trades_sell.append(entry_price)
                  inicio = index
                  drawdown = row['Close']
                  first_mean = row['Volume_M']
                  highest =  row['Close']
              elif in_trade == True and row['High'] > drawdown:
                  drawdown = row['High']
              elif in_trade and row['Low'] < highest:
                  highest = row['Low']
              elif in_trade and row['Volume'] < first_mean:
                  in_trade = False
                  exit_price = row['Close']
                  exit_price=round(exit_price,2)
                  trades_buy.append(exit_price)
                  fim = index
                  periodo = fim - inicio
                  trades_periodo.append(periodo)            
                  drawdown = ((entry_price/drawdown)-1)*100
                  drawdown = round(drawdown, 2)
                  trades_drawdown.append(drawdown)
                  if exit_price < highest:
                      highest = exit_price 
                  high = ((entry_price/highest)-1)*100
                  high = round(high, 2)
                  trades_high.append(high)



          if len(trades_buy) != len(trades_sell):
            trades_sell = trades_sell[:-1]

          
          # Creating DataFrame for trades
          trades = pd.DataFrame({'Buy': trades_buy, 'Sell': trades_sell, 'Period': trades_periodo, 
                                 'Drawdown': trades_drawdown, 'Max Return': trades_high})
          
          # Calculating returns and capital
          trades['Return'] = (trades['Sell'] / trades['Buy'] - 1) * 100
          trades['Return'] = round(trades.Return, 2)
          return_list = trades['Return'].to_list()
          capital = 100
          for i in return_list:
              capital = capital + capital * (i / 100)
          capital = capital - 100
          capital = round(capital, 2)
                
          capital = 100
          total_return = 1
          evolution = []
        
          for index, r_value in trades['Return'].items():
              total_return *= 1 + (trades.loc[index, 'Return'])/100
              total_return_per = (total_return-1)*100
              evolution.append(total_return_per)
          global_r = (total_return - 1) * 100 
          global_r = round(global_r,2)
          st.markdown(f"<h5 style='text-align: left; color: grey;'>Global return of closed positions: {global_r} %</h5>", unsafe_allow_html=True)
        
          mediana = trades.Return.median()
          mediana = round(mediana, 2)
          st.write(f'**Median return per trade: {mediana}**')


      with col21:    
          fig_combined_cumulative = px.line(evolution, title='Retorno cumulativo da estratégia')
          fig_combined_cumulative.update_layout(title='Retorno cumulativo da estratégia', xaxis_title='Trades', yaxis_title='Return (percentage)',showlegend=False)
          st.plotly_chart(fig_combined_cumulative, use_container_width=True)        
    

      with col22:
          fig_combined = px.bar(trades, x=trades.index, y=['Max Return','Drawdown','Return'], title='Retorno Potencial, Retorno e Drawdown por trade', color_discrete_sequence=['navy', 'red', 'cornflowerblue'])
          fig_combined.update_layout(title='Retorno Potencial, Retorno e Drawdown por trade', xaxis_title='Trades', yaxis_title='Percentage',  **{'barmode': 'overlay'})
          st.plotly_chart(fig_combined, use_container_width=True)
    

      with col23:
          st.write('**Trades individuais**')
          st.dataframe(trades, use_container_width=True)   
