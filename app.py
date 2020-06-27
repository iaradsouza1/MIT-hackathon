import os
import streamlit as st
from datetime import date
import compute_metrics as cm

states_names = [
    'AC', 'AL', 'AM', 'AP',
    'BA', 'CE', 'DF', 'ES',
    'GO', 'MA', 'MG', 'MS',
    'MT', 'PA', 'PB', 'PE',
    'PI', 'PR', 'RJ', 'RN',
    'RO', 'RR', 'RS', 'SC',
    'SE', 'SP', 'TO'
]

def main():

    today = date.today()
    file_current = 'data/Covid-19-Brasil_' + str(today) + '.csv'

    # TO DO: CREATE A SCRIPT TO CALL compute_metrics.py
    if not os.path.isfile(file_current):
        cm.main()

    st.title('COVID-19 directionality app')
    page = st.sidebar.radio('Select level:', ('Nation-wide statistics', 'By state statistics'), index=0)

    if page == 'Nation-wide statistics':
        st.header('Best Rt estimation for all states')
        st.image('plot/All.png', use_column_width=True)
        st.header('Infection spreading by states, according to their level of social distancing adopted by local governments:')
        st.image('plot/High.png', use_column_width=True)
        st.header('States currently not undercontrol:')
        st.image('plot/not_undercontrol.png', use_column_width=True)
        st.header('States currently undercontrol:')
        st.image('plot/undercontrol.png', use_column_width=True)

    elif page == 'By state statistics':
        state = st.sidebar.selectbox('Select a state:', states_names)
        new_cases = 'plot/' + str(state) + '_new_cases_per_day.png'
        posteriors = 'plot/' + str(state) + '_posteriors.png'
        hdis = 'plot/' + str(state) + '_hdis.png'

        st.header(f'How many cases of COVID-19 {state} has in a daily basis?')
        st.image(new_cases, use_column_width=True)
        st.header('Estimate Rt likelihood')
        st.image(posteriors, use_column_width=True)
        st.header('Best Rt estimation by day')
        st.image(hdis, use_column_width=True)

if __name__ == "__main__":
    main()