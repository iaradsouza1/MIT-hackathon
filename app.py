import os
import streamlit as st
from datetime import date
import compute_metrics as cm
import pandas as pd
import plots
from datetime import date

states_names = [
    'AC', 'AL', 'AM', 'AP',
    'BA', 'CE', 'DF', 'ES',
    'GO', 'MA', 'MG', 'MS',
    'MT', 'PA', 'PB', 'PE',
    'PI', 'PR', 'RJ', 'RN',
    'RO', 'RR', 'RS', 'SC',
    'SE', 'SP', 'TO'
]

RESULTS = 'results.pkl'
GEOJSON = 'data/brasil_estados.geojson'

def main():

    # TO DO: fix download
    # today = date.today()
    # file_current = 'data/Covid-19-Brasil_' + str(today) + '.csv'

    # # TO DO: CREATE A SCRIPT TO CALL compute_metrics.py
    # if not os.path.isfile(file_current):
    #     cm.main()

    st.title('COVID-19 directionality app')

    df_results = plots.pickle_to_df(RESULTS, GEOJSON)
    last_day = st.date_input('Date') 
    page = st.sidebar.radio('Select level:', ('Nation-wide statistics', 'By state statistics'), index=0)

    # Show map
    st.plotly_chart(plots.plot_map(df_results, last_day, GEOJSON))

    if page == 'Nation-wide statistics':
        st.header('Best Rt estimation for all states')
        st.image('plot/All.png', use_column_width=True)
        st.header('Infection spreading by states, according to their level of social distancing adopted by local governments:')
        st.image('plot/High.png', use_column_width=True)
        st.header('States currently not undercontrol:')
        st.image('plot/not_undercontrol.png')
        st.header('States currently undercontrol:')
        st.image('plot/undercontrol.png')

    elif page == 'By state statistics':
        state = st.sidebar.selectbox('Select a state:', states_names)

        st.header('Check new cases by day')
        st.plotly_chart(plots.plot_new_cases(state))

        st.header('Check Rt by day')    
        st.plotly_chart(plots.plot_rt_state(state, df_results))


if __name__ == "__main__":
    main()