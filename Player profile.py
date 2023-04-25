import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(page_title="Аналитика", page_icon=":bar_chart:", layout="wide")

import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    if username == 'kvas' or username == 'kaplin':
        import pandas as pd
        # import openpyxl
        import seaborn as sns
        #import xlsxwriter
        import streamlit as st
        from io import BytesIO
        import requests
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt

        # In[ ]:
        # st.set_page_config(page_title="Аналитика", page_icon=":bar_chart:", layout="wide")

        # In[4]:


        st.title('Индивидуальный профиль футболиста')


        data2 = pd.DataFrame()

        # In[6]:

        data1 = ['Unnamed: 0', 'InStat Index', 'Matches played',
                 'Minutes played', 'Starting lineup appearances',
                 'Goals', 'Assists', 'Expected assists', 'Offsides', 'Yellow cards',
                 'Red cards', 'Shots', 'Penalty',
                 'Penalty kicks scored. %', 'Passes',
                 'Accurate passes. %', 'Key passes', 'Crosses', 'Accurate crosses. %',
                 'Lost balls', 'Lost balls in own half', 'Ball recoveries',
                 "Ball recoveries in opponent's half", 'xG (Expected goals)',
                 'Challenges', 'Challenges won. %', 'Attacking challenges',
                 'Air challenges', 'Dribbles', 'Tackles', 'Ball interceptions',
                 'Free ball pick ups', 'Defensive challenges',
                 'Challenges in defence won. %', 'Age', 'Height', 'Air challenges won. %',
                 'Challenges in attack won. %',
                 'Successful dribbles. %', 'Tackles won. %', 'Fouls',
                 'Fouls suffered', 'Key passes accurate', "Shots on target. %"]
        data34 = ['Goals', 'Assists', 'Expected assists', 'Offsides', 'Yellow cards',
                  'Red cards', 'Shots', 'Penalty',
                  'Passes',
                  'Key passes', 'Crosses',
                  'Lost balls', 'Lost balls in own half', 'Ball recoveries',
                  "Ball recoveries in opponent's half", 'xG (Expected goals)',
                  'Challenges', 'Attacking challenges',
                  'Air challenges', 'Dribbles', 'Tackles', 'Ball interceptions',
                  'Free ball pick ups', 'Defensive challenges', 'Fouls',
                  'Fouls suffered', 'Key passes accurate']

        # In[7]:
        optionals_pt = st.expander("Playing time:", True)
        pt = optionals_pt.slider(
            'Select a range',
            0.0, 1000.0, (1000.0, 2500.0))
        clicked = st.button('Заменить')
        if clicked:
            st.cache_data.clear()



        # In[3]:

        @st.cache_data
        def nas0(lig, kef):
            global data1
            global data34

            spreadsheet_id = "1HJ6JxCxHm4OJMMcDo2w9uMldHk3yich7acxizJKMK8k"
            file_name = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv'.format(spreadsheet_id)
            r = requests.get(file_name)
            df = pd.read_csv(BytesIO(r.content))

            for i in df:
                df['League'] = lig

            for i in df.columns:
                if '%' in i:
                    df[i] = df[i].str.replace(r'\D', '', regex=True)

            for i in data1:
                df[i] = pd.to_numeric(df[i], errors='coerce').fillna(0).astype(float)

            for i in data34:
                df[i] = df[i] / (df['Minutes played'] / 90)

            df = df.rename(columns={'Minutes played': 'Minutes_played'})
            df = df.query('@pt[0]<=Minutes_played<=@pt[1]')

            df.rename(columns={'Unnamed: 1': 'Name'}, inplace=True)
            data = df

            for i in data:
                data['sh'] = kef * (0.7 * data['xG (Expected goals)'] + 0.5 * data['Shots'] / 100 * data[
                    'Shots on target. %'] / 100 + data['Goals']) / ((0.7 * data['xG (Expected goals)'] + 0.5 * data[
                    'Shots'] / 100 * data['Shots on target. %'] / 100 + data['Goals']).max())
                data['Defence'] = kef * (((data['Challenges in defence won. %'] + 0.01 * data[
                    'Challenges in defence won. %'] * data['Defensive challenges'] * 2) + (
                                                      data['Tackles won. %'] + 0.01 * data['Tackles won. %'] * data[
                                                  'Tackles'] * 2)) / ((data['Challenges in defence won. %'] + 0.01 *
                                                                       data['Challenges in defence won. %'] * data[
                                                                           'Defensive challenges'] * 2) + (
                                                                                  data['Tackles won. %'] + 0.01 * data[
                                                                              'Tackles won. %'] * data[
                                                                                      'Tackles'] * 2)).max())
                data['air'] = kef * ((data['Air challenges won. %'] + 0.01 * data['Air challenges won. %'] * data[
                    'Air challenges'] * 2) / ((data['Air challenges won. %'] + 0.01 * data['Air challenges won. %'] *
                                               data['Air challenges'] * 2).max()))
                data['Recovery'] = kef * (
                            (data['Ball interceptions'] + data['Ball recoveries'] + data['Free ball pick ups']) / (
                        (data['Ball interceptions'] + data['Free ball pick ups'] + data['Ball recoveries']).max()))
                data['Distribution'] = kef * (
                            data['Accurate passes. %'] + 0.01 * data['Accurate passes. %'] * data['Passes'] + 0.3 * (
                                data['Accurate crosses. %'] + 0.01 * data['Accurate crosses. %'] * data[
                            'Crosses'] * 1.3)) / ((data['Accurate passes. %'] + 0.01 * data['Accurate passes. %'] *
                                                   data['Passes'] + 0.3 * (data['Accurate crosses. %'] + 0.01 * data[
                            'Accurate crosses. %'] * data['Crosses'] * 1.3)).max())
                data['Take on'] = kef * ((data['Successful dribbles. %'] + 0.01 * data['Successful dribbles. %'] * data[
                    'Dribbles'] * 10) / ((data['Successful dribbles. %'] + 0.01 * data['Successful dribbles. %'] * data[
                    'Dribbles'] * 10).max()))
                data['Chance creation'] = kef * (
                            0.9 * data['Key passes'] + data['Assists'] + 0.9 * data['Expected assists']) / (
                                                      0.9 * data['Key passes'] + data['Assists'] + 0.9 * data[
                                                  'Expected assists']).max()
                data['Rank'] = data['sh'] + data['Defence'] + data['Recovery'] + data['Distribution'] + data[
                    'Take on'] + data['air'] + data['Chance creation']

            data = data.sort_values('Rank', ascending=False)

            return data

            # In[10]:

        data = nas0('РПЛ', 1)
        data2 = data
        datagr = data2
        data2 = data2[
            ['Name', 'Position', 'sh', 'Defence', 'Recovery', 'Distribution', 'Take on', 'air', 'Chance creation',
             'Rank', 'Team', 'League', 'Age', 'Minutes_played']].sort_values('Rank', ascending=False).head(1000)


        # player profile
        leaguepp = st.selectbox("Выбор лиги:", data2["League"].unique())
        positionpp = st.selectbox("Выбор позиции:", data2["Position"].unique())
        dfpp = data2.query(
            "League ==@leaguepp & Position ==@positionpp"
        )

        # In[26]:

        name = st.selectbox("Выберите футболиста:", dfpp["Name"].unique())
        df_selection1 = dfpp.query(
            "Name== @name")
        df_selection1 = df_selection1.reset_index(drop=True)

        df_selection1 = df_selection1[
            ['sh', 'Defence', 'Recovery', 'Distribution', 'Take on', 'air', 'Chance creation']]

        df_selection1.at[1, 'sh'] = dfpp['sh'].median()
        df_selection1.at[1, 'Defence'] = dfpp['Defence'].median()
        df_selection1.at[1, 'Recovery'] = dfpp['Recovery'].median()
        df_selection1.at[1, 'Distribution'] = dfpp['Distribution'].median()
        df_selection1.at[1, 'Take on'] = dfpp['Take on'].median()
        df_selection1.at[1, 'air'] = dfpp['air'].median()
        df_selection1.at[1, 'Chance creation'] = dfpp['Chance creation'].median()
        df_selection1 = df_selection1.reset_index(drop=True)

        #строим радар футболиста
        ps = datagr.query("Name == @name & Position==@positionpp")
        ps = ps[['Name', 'Age', 'Foot', 'Height', 'Nationality', 'Team']]
        st.dataframe(ps)

        fig = go.Figure()

        colors = ["tomato", "dodgerblue", "yellow"]
        for i in range(2):
            fig.add_trace(
                go.Scatterpolar(
                    r=df_selection1.loc[i].values.tolist() + df_selection1.loc[i].values.tolist()[:1],
                    theta=df_selection1.columns.tolist() + df_selection1.columns.tolist()[:1],
                    fill='toself',

                    fillcolor=colors[i], line=dict(color=colors[i]),
                    showlegend=True, opacity=0.3
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.05]
                )
            ),
            title={
                'text': "Рейтинг игрока(красный) относительно игроков этой позиции",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        st.plotly_chart(fig)


        # графики
        st.title('График')
        # In[29]:

        GA_selection2 = datagr.query("League == @leaguepp & Position == @positionpp")
        GA_selection2.columns = GA_selection2.columns.str.replace('.', '')
        GA_selection2.columns = GA_selection2.columns.str.replace("'", '')

        j = st.selectbox("Выберите 2 параметр:", GA_selection2[['Matches played',
                                                                'Minutes_played', 'Starting lineup appearances',
                                                                'Goals',
                                                                'xG (Expected goals)', 'Assists', 'Expected assists',
                                                                'Offsides',
                                                                'Yellow cards', 'Red cards', 'Shots',
                                                                'Shots on target %', 'Passes',
                                                                'Accurate passes %', 'Key passes',
                                                                'Key passes accurate', 'Penalty',
                                                                'Penalty kicks scored %', 'Crosses',
                                                                'Accurate crosses %', 'Lost balls',
                                                                'Lost balls in own half', 'Ball recoveries',
                                                                "Ball recoveries in opponents half", 'Challenges',
                                                                'Challenges won %',
                                                                'Attacking challenges', 'Challenges in attack won %',
                                                                'Defensive challenges', 'Challenges in defence won %',
                                                                'Air challenges',
                                                                'Air challenges won %', 'Dribbles',
                                                                'Successful dribbles %', 'Tackles',
                                                                'Tackles won %', 'Ball interceptions',
                                                                'Free ball pick ups', 'Fouls',
                                                                'Fouls suffered']].columns)

        # In[5]:

        # In[30]:
        st.set_option('deprecation.showPyplotGlobalUse', False)

        def veg(i,j):
            st.vega_lite_chart({'data': GA_selection2,
                                'layer': [{'mark': {'type': 'circle'},
                                           'encoding': {
                                               'x': {'field': i, 'type': 'quantitative', "scale": {"zero": False}},
                                               'y': {'field': j, 'type': 'quantitative', "scale": {"zero": False}},
                                               'size': {'field': i, 'type': 'quantitative'},
                                               'color': {'field': 'Name', 'type': 'nominal'}}},
                                          {"mark": "rule",
                                           "encoding": {
                                               "x": {"aggregate": "median", "field": i},
                                               "size": {"value": 2}}},
                                          {"mark": "rule",
                                           "encoding": {
                                               "y": {"aggregate": "median", "field": j},
                                               "size": {"value": 2}}}
                                          ]}, use_container_width=True)
        if positionpp=='LD' or positionpp=='RD':
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                ["Движение", "Перехват владения", "Отбор", "Игра в воздухе", "Передачи", "Владение", "Навесы"])
            with tab1:
                veg('Dribbles', 'Successful dribbles %')
            with tab2:
                veg('Ball interceptions', 'Ball recoveries')
            with tab3:
                veg('Tackles', 'Tackles won %')
            with tab4:
                veg('Air challenges', 'Air challenges won %')
            with tab5:
                veg('Passes', 'Accurate passes %')
            with tab6:
                veg('Lost balls', 'Ball recoveries')
            with tab7:
                veg('Crosses', 'Accurate crosses %')

        if positionpp=='CD':
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["Перехват владения", "Отбор",  "Передачи", "Игра в воздухе", "Владение"])
            with tab1:
                veg('Ball interceptions', 'Ball recoveries')
            with tab2:
                veg('Tackles', 'Tackles won %')
            with tab3:
                veg('Passes', 'Accurate passes %')
            with tab4:
                veg('Air challenges', 'Air challenges won %')
            with tab5:
                veg('Lost balls in own half', 'Ball recoveries')

        if positionpp=='DM' or positionpp=='CM':
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
                ["Перехват владения", "Отбор",  "Передачи", "Игра в воздухе", "Владение", "Развитие пасов", "Все испытания",
                 "Результаты созидания"])
            with tab1:
                veg('Ball interceptions', 'Ball recoveries')
            with tab2:
                veg('Tackles', 'Tackles won %')
            with tab3:
                veg('Passes', 'Accurate passes %')
            with tab4:
                veg('Air challenges', 'Air challenges won %')
            with tab5:
                veg('Lost balls in own half', 'Ball recoveries')
            with tab6:
                veg('Key passes', 'Accurate passes %')
            with tab7:
                veg('Challenges', 'Challenges won %')
            with tab8:
                veg('Key passes', 'Expected assists')

        if positionpp == 'LM' or positionpp == 'RM':
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                ["Движение", "Результаты созидания(передачи)", "Действия в атаке", "Удары", "Передачи",
                 "Воздействие на оборону соперника", "Навесы"])
            with tab1:
                veg('Dribbles', 'Successful dribbles %')
            with tab2:
                veg('Key passes', 'Expected assists')
            with tab3:
                veg('Attacking challenges', 'Challenges in attack won %')
            with tab4:
                veg('Shots', 'Goals')
            with tab5:
                veg('Passes', 'Accurate passes %')
            with tab6:
                veg('Dribbles', 'Fouls suffered')
            with tab7:
                veg('Crosses', 'Accurate crosses %')

        if positionpp == 'F':
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                ["Движение", "Результаты созидания(передачи)", "Действия в атаке", "Реализация", "Передачи",
                 "Воздействие на оборону соперника", "Игра в воздухе"])
            with tab1:
                veg('Dribbles', 'Successful dribbles %')
            with tab2:
                veg('Key passes', 'Expected assists')
            with tab3:
                veg('Attacking challenges', 'Challenges in attack won %')
            with tab4:
                veg('xG (Expected goals)', 'Goals')
            with tab5:
                veg('Passes', 'Accurate passes %')
            with tab6:
                veg('Dribbles', 'Fouls suffered')
            with tab7:
                veg('Air challenges', 'Air challenges won %')


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')