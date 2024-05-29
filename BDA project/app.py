import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.functions import col, to_date
import pandas as pd
from pyspark.sql import DataFrame
import plotly.graph_objs as go
import plotly.express as px
from pyspark.sql.functions import countDistinct
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go
import os


spark = SparkSession.builder \
    .appName("Energy Consumption Analysis") \
    .getOrCreate()

@st.cache_data
def load_data():
    data = pd.read_csv("acorn_details.csv", encoding='latin-1')
    return data


def merge_csv(input_dir, output_file):
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Sort files to ensure they are merged in the correct order
    all_files.sort()

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)

def fit_arima_model(train):
    endog = train['avg_energy']
    model = sm.tsa.ARIMA(endog, order=(7, 1, 0))
    results = model.fit()
    return results

def fit_sarima_model(train):
    endog = train['avg_energy']
    model = sm.tsa.SARIMAX(endog, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    results = model.fit()
    return results

def calculate_performance(train, test, model_name):
    if model_name == 'ARIMA':
        results = fit_arima_model(train)
        predict = results.predict(start=len(train), end=len(train)+len(test)-1)
    elif model_name == 'SARIMA':
        results = fit_sarima_model(train)
        predict = results.predict(start=len(train), end=len(train)+len(test)-1)
    elif model_name == 'SARIMAX':
        endog = train['avg_energy']
        exog = sm.add_constant(train[['weather_cluster', 'holiday_ind']])
        model = sm.tsa.SARIMAX(endog, exog=exog, order=(1,1,1), seasonal_order=(1,1,0,12))
        results = model.fit()

        predict = results.predict(start=len(train), end=len(train)+len(test)-1, exog=sm.add_constant(test[['weather_cluster', 'holiday_ind']]))
    
    test['predicted'] = predict.values

    test['residual'] = abs(test['avg_energy'] - test['predicted'])
    MAE = test['residual'].sum() / len(test)
    MAPE = (abs(test['residual']) / test['avg_energy']).sum() * 100 / len(test)
    st.write("Mean Absolute Error (MAE):", MAE)
    st.write("Mean Absolute Percentage Error (MAPE):", MAPE)


def main():
    input_directory = 'chunks'  # Replace with your directory containing split CSV files
    output_file = 'daily_dataset.csv'  # Replace with your desired output file path

    merge_csv(input_directory, output_file)

    st.set_page_config(layout="wide") 

    st.title("Smart Meters Data Visualization and Forecasting :partly_sunny:")

    data = load_data()

    st.sidebar.image(r"assets\bml-color-logo.svg", use_column_width=True)
    
    st.sidebar.text("Big Data Analytics")

    st.sidebar.divider()

    page = st.sidebar.selectbox("Navigation", ["Weather", "Acron Groups Info", "Forecasting","Forecasting LCLid"], key="sidebar")

    st.markdown(
        """
        <style>
            .st-emotion-cache-10trblm{
                text-align:center;
            }
            .st-emotion-cache-183lzff{
                margin: auto;
            }
            p{
                text-align:center;
            }
            .sidebar .sidebar-content {
                min-width: 250px;
            }
            .st-emotion-cache-uf99v8 {
                align-items: start;
                margin-left: 0px;
            }
            .st-emotion-cache-ocqkz7{
                # width:1250px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if page == "Weather":
        show_home_page()
    elif page == "Acron Groups Info":
        show_acron_groups_info_page(data)
    elif page == "Forecasting":
        show_forecasting_page()
    elif page == "Forecasting LCLid":
        show_forecasting_page_two()

def show_home_page():
    st.header("Weather Data Visualization Page :sun_behind_rain_cloud:")

    weather_df = spark.read.csv('weather_daily_darksky.csv', header=True, inferSchema=True)
    weather = weather_df.withColumn('day', to_date(col('time').cast('timestamp')))

    weather = weather.select('temperatureMax', 'windBearing', 'dewPoint', 'cloudCover', 'windSpeed',
                            'pressure', 'apparentTemperatureHigh', 'visibility', 'humidity',
                            'apparentTemperatureLow', 'apparentTemperatureMax', 'uvIndex',
                            'temperatureLow', 'temperatureMin', 'temperatureHigh',
                            'apparentTemperatureMin', 'moonPhase', 'day') \
                    .na.drop()
    
    col3, col4 = st.columns([1, 1])

    with col3:
        block_num = st.selectbox("Select Block Number:", list(range(0, 112)), index=0)

    file_path = f"daily_dataset/daily_dataset/block_{block_num}.csv"
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df = df.select("day", "LCLid", "energy_sum")

    energy_df =  df

    with col4:
        year_options = [2011, 2012, 2013, 2014, "All"]
        year_filter = st.selectbox("Select Year:", year_options)

    if year_filter == "All":
        energy_df = energy_df
    else:
        energy_df = energy_df.filter(F.year('day') == int(year_filter))

    energy_df = energy_df.groupBy("day").agg({"energy_sum": "sum", "LCLid": "count"})
    energy_df = energy_df.withColumnRenamed("sum(energy_sum)", "energy_sum")
    energy_df = energy_df.withColumnRenamed("count(LCLid)", "house_count")
    energy_df = energy_df.withColumn("average_energy", energy_df["energy_sum"] / energy_df["house_count"])

    weather_energy = energy_df.join(weather, on='day')
    weather_energy = weather_energy.sort('day')
    weather_energy = weather_energy.toPandas()

    trace1 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['temperatureMax'],
        mode='lines',
        name='Temperature Max'
    )

    trace2 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['temperatureMin'],
        mode='lines',
        name='Temperature Min'
    )

    trace3 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['average_energy'],
        mode='lines',
        name='Average Energy/Household',
        yaxis='y2'
    )

    layout = go.Layout(
        title='Energy Consumption and Temperature',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Temperature'),
        yaxis2=dict(title='Average Energy/Household', overlaying='y', side='right')
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.update_layout(height=600, width=1200)
    st.plotly_chart(fig)

    st.divider()


    trace1 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['humidity'],
        mode='lines',
        name='Humidity',
        yaxis='y1'
    )

    trace2 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['average_energy'],
        mode='lines',
        name='Average Energy/Household',
        yaxis='y2'
    )

    layout = go.Layout(
        title='Energy Consumption and Humidity',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Humidity'),
        yaxis2=dict(title='Average Energy/Household', overlaying='y', side='right', color='royalblue')
    )


    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(height=600, width=1200)
    st.plotly_chart(fig)

    st.divider()

    trace1 = go.Scatter(
    x=weather_energy['day'],
    y=weather_energy['cloudCover'],
    mode='lines',
    name='Cloud Cover',
    yaxis='y1'
    )

    trace2 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['average_energy'],
        mode='lines',
        name='Average Energy/Household',
        yaxis='y2'
    )

    layout = go.Layout(
        title='Energy Consumption and Cloud Cover',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Cloud Cover'),
        yaxis2=dict(title='Average Energy/Household', overlaying='y', side='right', color='rgb(53, 126, 209)')
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(height=600, width=1200)
    st.plotly_chart(fig)

    st.divider()

    trace1 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['visibility'],
        mode='lines',
        name='Visibility',
        yaxis='y1'
    )

    trace2 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['average_energy'],
        mode='lines',
        name='Average Energy/Household',
        yaxis='y2'
    )

    layout = go.Layout(
        title='Energy Consumption and Visibility',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Visibility', color='rgb(239, 135, 45)'),
        yaxis2=dict(title='Average Energy/Household', overlaying='y', side='right', color='rgb(53, 126, 209)')
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(height=600, width=1200)
    st.plotly_chart(fig)

    st.divider()

    trace1 = go.Scatter(
    x=weather_energy['day'],
    y=weather_energy['windSpeed'],
    mode='lines',
    name='Wind Speed',
    yaxis='y1'
    )

    trace2 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['average_energy'],
        mode='lines',
        name='Average Energy/Household',
        yaxis='y2'
    )

    layout = go.Layout(
        title='Energy Consumption and Wind Speed',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Wind Speed', color='rgb(239, 135, 45)'),
        yaxis2=dict(title='Average Energy/Household', overlaying='y', side='right', color='rgb(53, 126, 209)')
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(height=600, width=1200)
    st.plotly_chart(fig)

    st.divider()

    trace1 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['uvIndex'],
        mode='lines',
        name='UV Index',
        yaxis='y1'
    )

    trace2 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['average_energy'],
        mode='lines',
        name='Average Energy/Household',
        yaxis='y2'
    )

    layout = go.Layout(
        title='Energy Consumption and UV Index',
        xaxis=dict(title='Day'),
        yaxis=dict(title='UV Index', color='rgb(239, 135, 45)'),
        yaxis2=dict(title='Average Energy/Household', overlaying='y', side='right', color='rgb(53, 126, 209)')
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(height=600, width=1200)
    st.plotly_chart(fig)

    st.divider()

    trace1 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['dewPoint'],
        mode='lines',
        name='Dew Point',
        yaxis='y1'
    )

    trace2 = go.Scatter(
        x=weather_energy['day'],
        y=weather_energy['average_energy'],
        mode='lines',
        name='Average Energy/Household',
        yaxis='y2'
    )

    layout = go.Layout(
        title='Energy Consumption and Dew Point',
        xaxis=dict(title='Day'),
        yaxis=dict(title='Dew Point', color='rgb(239, 135, 45)'),
        yaxis2=dict(title='Average Energy/Household', overlaying='y', side='right', color='rgb(53, 126, 209)')
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(height=600, width=1200)
    st.plotly_chart(fig)

    

def show_acron_groups_info_page(data):
    st.header("Acron Groups Information :brain:	")
    st.write("This page displays information about the number of ACORN groups within each main category.")

    grouped_data = data.groupby("MAIN CATEGORIES").sum().reset_index()

    acorn_counts = data.iloc[:, 3:].sum()
    fig_pie = px.pie(names=acorn_counts.index, values=acorn_counts.values)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')  # Show labels inside pie slices
    fig_pie.update_layout(margin=dict(l=250, r=10, b=50, t=30), height=400, width=600, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')  # Adjust layout and show legend

    fig_bar = px.bar(grouped_data, x='MAIN CATEGORIES', y=grouped_data.columns[1:], title='Number of ACORN Groups by Main Category')
    fig_bar.update_layout(height=600, width=800, margin=dict(l=0, r=50, t=50, b=150))  # Adjust the plot size and margin

    col1, col2 = st.columns([1, 1]) 
    with col1:
        st.plotly_chart(fig_bar)
    with col2:
        st.plotly_chart(fig_pie)

    st.divider()

    categories = data['CATEGORIES'].unique().tolist()
    col3, col4, col5 = st.columns([1, 1, 3])

    with col3:
        selected_category = st.selectbox("Select a category:", categories)

    references = data[data['CATEGORIES'] == selected_category]['REFERENCE'].unique().tolist()

    with col4:
        selected_reference = st.selectbox("Select a reference:", references)

    filtered_data = data[(data['CATEGORIES'] == selected_category) & (data['REFERENCE'] == selected_reference)]

    acorn_counts = filtered_data.iloc[:, 3:].sum()
    fig = px.pie(names=acorn_counts.index, values=acorn_counts.values, title=f'ACORN Distribution for {selected_category} and {selected_reference}')
    fig.update_layout(height=500, width=1000, margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig)

    original_df = spark.read.csv("daily_dataset.csv", header=True)
    informations_df = spark.read.csv("informations_households.csv", header=True)

    joined_df = original_df.join(informations_df, on='LCLid', how='inner')
    total_energy_per_acorn_grouped = joined_df.groupBy('Acorn_grouped').agg(F.sum('energy_sum').alias('total_energy_consumed'))
    total_energy_df = total_energy_per_acorn_grouped.toPandas()
    fig_bar = px.bar(total_energy_df, y='Acorn_grouped', x='total_energy_consumed', orientation='h',
                    title='Total Energy Consumed per Acorn_grouped', labels={'total_energy_consumed': 'Total Energy Consumed'})
    fig_bar.update_traces(marker_line_width=1, marker_line_color="black", opacity=0.6)  # Add vertical lines
    fig_bar.update_layout(yaxis_title='Acorn_grouped', xaxis_title='Total Energy Consumed')
    fig_pie = px.pie(total_energy_df, names='Acorn_grouped', values='total_energy_consumed', title='Energy Consumption Distribution by Acorn_grouped',
                    labels={'Acorn_grouped': 'Acorn Grouped', 'total_energy_consumed': 'Total Energy Consumed'})
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')  # Show percentages inside pie slices
    fig_pie.update_layout(showlegend=False)

    col1, col2 = st.columns([1, 1]) 
    with col1:
        st.plotly_chart(fig_bar)
    with col2:
        st.plotly_chart(fig_pie)

    col3, col4, col5 = st.columns([1, 1, 3])

    with col3: 
        acorn_options = ["ACORN-A","ACORN-B","ACORN-C","ACORN-D","ACORN-E","ACORN-F","ACORN-G","ACORN-H","ACORN-I","ACORN-J","ACORN-K","ACORN-L","ACORN-M","ACORN-N","ACORN-O","ACORN-P","ACORN-Q"] 
        acorn_filter = st.selectbox("Select ACORN:", acorn_options)

    with col4:
        year_options = [2011, 2012, 2013, 2014, "All"]
        year_filter = st.selectbox("Select Year:", year_options)

    if year_filter == "All":
        filtered_df = joined_df.filter((F.col('Acorn') == acorn_filter))
    else:
        filtered_df = joined_df.filter((F.col('Acorn') == acorn_filter) & (F.year('day') == int(year_filter)))

    daily_energy_consumption = filtered_df.groupBy('day').agg(F.sum('energy_sum').alias('total_energy_consumed'))

    daily_energy_df = daily_energy_consumption.toPandas()

    fig_scatter = px.scatter(daily_energy_df, x='day', y='total_energy_consumed', title=f'Energy Consumption per Day for {acorn_filter} in {year_filter}',
                            labels={'total_energy_consumed': 'Total Energy Consumed', 'day': 'Day'})
    fig_scatter.update_layout(xaxis_title='Day', yaxis_title='Total Energy Consumed',height=500, width=1100)
    st.plotly_chart(fig_scatter)

def show_forecasting_page():
    st.header("Forecasting :chart_with_upwards_trend:")
    weather_energy = pd.read_csv('weather_energy.csv')
    weather_energy['day'] = pd.to_datetime(weather_energy['day'])
    weather_energy['Year'] = weather_energy['day'].dt.year
    weather_energy['Month'] = weather_energy['day'].dt.month
    weather_energy.set_index('day', inplace=True)

    train = weather_energy.iloc[0:(len(weather_energy)-30)]
    test = weather_energy.iloc[len(train):(len(weather_energy)-1)]

    st.subheader('Select Model')
    selected_model = st.selectbox("Model:", ('ARIMA', 'SARIMA', 'SARIMAX'))

    if selected_model == 'ARIMA':
        results = fit_arima_model(train)
        predict = results.predict(start=len(train), end=len(train)+len(test)-1)
    elif selected_model == 'SARIMA':
        results = fit_sarima_model(train)
        predict = results.predict(start=len(train), end=len(train)+len(test)-1)
    elif selected_model == 'SARIMAX':
        endog = train['avg_energy']
        exog = sm.add_constant(train[['weather_cluster', 'holiday_ind']])
        model = sm.tsa.SARIMAX(endog, exog=exog, order=(7,1,1), seasonal_order=(1,1,0,12))
        results = model.fit()
        predict = results.predict(start=len(train), end=len(train)+len(test)-1, exog=sm.add_constant(test[['weather_cluster', 'holiday_ind']]))

    test['predicted'] = predict.values

    last_date = test.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    if selected_model == 'SARIMAX':
        exog_combinations = [
            (0, 0), (1, 1),
            (1, 0), (1, 1),
            (2, 0), (2, 1),
        ]
        exog_combinations = [(1,) + combo for combo in exog_combinations]
        exog_combo_index = st.selectbox("Select exogenous variable combination(0: Cluster 0, Not Holiday, 1: Cluster 1, Not Holiday, 2: Cluster 2, Not Holiday, 3: Cluster 0, Holiday, 4: Cluster 1, Holiday, 5: Cluster 2, Holiday):", range(len(exog_combinations)))
        exog_combo = exog_combinations[exog_combo_index]
        exog_next_day = np.array([exog_combo])
        next_day_prediction = results.forecast(steps=1, exog=sm.add_constant(exog_next_day))
        prediction_str = str(next_day_prediction)
        predicted_value = prediction_str.split()[1]
        st.subheader("Predicted Value for Next Day's Consumption")
        st.write(f"Predicted value for {next_date}: {predicted_value}")

    else:
        next_day_prediction = results.forecast(steps=1)
        st.subheader("Predicted Value for Next Day's Consumption")
        prediction_str = str(next_day_prediction)
        predicted_value = prediction_str.split()[1]
        st.write(f"Predicted value for {next_date}: {predicted_value}")

    st.subheader("Last 5 Days of Test Set - Real vs Predicted Values")
    st.table(test[['avg_energy', 'predicted']].tail().reset_index().rename(columns={'day': 'Date'}))

    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=train.index, y=train['avg_energy'], mode='lines', name='Actual'))
    fig_train.add_trace(go.Scatter(x=train.index, y=results.fittedvalues, mode='lines', name='Fitted'))
    fig_train.update_layout(title='Training Data and Fitted Values', xaxis_title='Day', yaxis_title='Energy')

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=test.index, y=test['avg_energy'], mode='lines', name='Actual', line=dict(color='red')))
    fig_test.add_trace(go.Scatter(x=test.index, y=test['predicted'], mode='lines', name='Predicted'))
    fig_test.update_layout(title='Actual vs Predicted', xaxis_title='Day', yaxis_title='Energy')

    fig_fitted = go.Figure()
    fig_fitted.add_trace(go.Scatter(x=train.index, y=results.fittedvalues, mode='lines', name='Fitted'))
    fig_fitted.add_trace(go.Scatter(x=test.index, y=test['predicted'], mode='lines', name='Predicted'))
    fig_fitted.update_layout(title='Fitted Values', xaxis_title='Day', yaxis_title='Energy')


    st.subheader("Graphs")
    st.plotly_chart(fig_train, use_container_width=True)
    st.plotly_chart(fig_test, use_container_width=True)
    st.plotly_chart(fig_fitted, use_container_width=True)


    st.subheader("Model Performance Comparison")

    st.write("ARIMA Model Performance:")
    calculate_performance(train, test, 'ARIMA')

    st.write("SARIMA Model Performance:")
    calculate_performance(train, test, 'SARIMA')

    st.write("SARIMAX Model Performance:")
    calculate_performance(train, test, 'SARIMAX')

def show_forecasting_page_two():


    data = pd.read_csv('daily_dataset.csv')
    
    lclids = data['LCLid'].unique()

    selected_lclid = st.selectbox("Select an LCLid:", lclids)

    model_data = data[data['LCLid'] == selected_lclid]
    model_data = model_data.set_index('day')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_data.index, y=model_data['energy_sum'], mode='lines'))
    fig.update_layout(title=f"Energy Sum for LCLid: {selected_lclid}",width=1100, xaxis_title='Day', yaxis_title='Energy Sum')

    st.plotly_chart(fig)

    # Fit SARIMA model
    endog = model_data['energy_sum']
    model = sm.tsa.SARIMAX(endog, order=(1, 1, 5), seasonal_order=(1, 1, 1, 12))
    results = model.fit()

    train = model_data.iloc[0:(len(model_data)-30)]
    test = model_data.iloc[len(train):]

    predict = results.predict(start=len(train), end=len(train)+len(test)-1)
    test['predicted'] = predict.values

    st.subheader("Next Day Prediction")
    st.write("Predicted Energy Sum for the next day:", predict.iloc[-1])

    st.subheader("Last 5 Predictions")
    st.dataframe(test[['energy_sum', 'predicted']].tail())

    fig, ax = plt.subplots(figsize=(12, 6))
    test['energy_sum'].plot(ax=ax, color='red', label='Actual')
    test['predicted'].plot(ax=ax, label='Predicted')
    ax.set_title(f'Energy Sum for LCLid: {selected_lclid}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Energy Sum')
    ax.legend()

    st.pyplot(fig)

if __name__ == "__main__":
    main()
