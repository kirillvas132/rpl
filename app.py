#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#import openpyxl
import seaborn as sns
#import xlsxwriter
import streamlit as st
from io import BytesIO
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# In[ ]:


st.set_page_config(page_title="Аналитика", page_icon=":bar_chart:", layout="wide")


# In[4]:


row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Анализ футболистов на основе статистических показателей(14.04)')


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
       'Challenges in defence won. %', 'Age', 'Height','Air challenges won. %',
         'Challenges in attack won. %',
         'Successful dribbles. %', 'Tackles won. %','Fouls',
       'Fouls suffered', 'Key passes accurate',"Shots on target. %"]
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


#optionals_team = st.expander("Игровое время", False)
#tm = optionals_team.selectbox("Выберите:", ("Ближе к основе", "Ближе к запасу"))


# In[8]:


if st.button('после каждой замены времени!'):
    st.experimental_memo.clear()


# In[3]:





@st.cache_data
def nas0(lig, kef):
    data = pd.DataFrame()#global data
    global data1
    global data34
    global tm

    spreadsheet_id = "1HJ6JxCxHm4OJMMcDo2w9uMldHk3yich7acxizJKMK8k"
    file_name = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv'.format(spreadsheet_id)
    r = requests.get(file_name)
    df = pd.read_csv(BytesIO(r.content))

    for i in df:
        df['League']= lig

    for i in df.columns:
        if '%' in i:
            df[i] = df[i].str.replace(r'\D', '', regex=True)

    for i in data1:
        df[i] = pd.to_numeric(df[i], errors='coerce').fillna(0).astype(float)
    

    for i in data34:
        df[i] = df[i]/(df['Minutes played']/90)
    
    #if tm == "Ближе к основе":
      #  df = df[(df['Minutes played']>1.3*df['Minutes played'].mean())]
   # else:
     #   df = df[(df['Minutes played']<=1.3*df['Minutes played'].mean())]
    df = df.rename(columns={'Minutes played': 'Minutes_played'})
    df = df.query('@pt[0]<=Minutes_played<=@pt[1]')
    
    df.rename(columns={'Unnamed: 1':'Name'}, inplace=True) 
    data = df

    for i in data: 
        data['sh'] = kef*(0.7*data['xG (Expected goals)'] + 0.5*data['Shots']/100*data['Shots on target. %']/100 + data['Goals'])/((0.7*data['xG (Expected goals)'] + 0.5*data['Shots']/100*data['Shots on target. %']/100 + data['Goals']).max())
        data['Defence'] = kef*(((data['Challenges in defence won. %']+0.01*data['Challenges in defence won. %']*data['Defensive challenges']*2)+(data['Tackles won. %']+0.01*data['Tackles won. %']*data['Tackles']*2))/((data['Challenges in defence won. %']+0.01*data['Challenges in defence won. %']*data['Defensive challenges']*2)+(data['Tackles won. %']+0.01*data['Tackles won. %']*data['Tackles']*2)).max())
        data['air'] = kef*((data['Air challenges won. %']+0.01*data['Air challenges won. %']*data['Air challenges']*2)/((data['Air challenges won. %']+0.01*data['Air challenges won. %']*data['Air challenges']*2).max()))
        data['Recovery'] = kef*((data['Ball interceptions']+data['Ball recoveries']+data['Free ball pick ups'])/((data['Ball interceptions']+data['Free ball pick ups']+data['Ball recoveries']).max()))
        data['Distribution'] = kef*(data['Accurate passes. %']+0.01*data['Accurate passes. %']*data['Passes']+0.3*(data['Accurate crosses. %']+0.01*data['Accurate crosses. %']*data['Crosses']*1.3))/((data['Accurate passes. %']+0.01*data['Accurate passes. %']*data['Passes']+0.3*(data['Accurate crosses. %']+0.01*data['Accurate crosses. %']*data['Crosses']*1.3)).max())
        data['Take on'] = kef*((data['Successful dribbles. %']+0.01*data['Successful dribbles. %']*data['Dribbles']*10)/((data['Successful dribbles. %']+0.01*data['Successful dribbles. %']*data['Dribbles']*10).max()))
        data['Chance creation'] = kef*(0.9*data['Key passes']+data['Assists']+0.9*data['Expected assists'])/(0.9*data['Key passes']+data['Assists']+0.9*data['Expected assists']).max()
        data['Rank'] = data['sh']+data['Defence']+data['Recovery']+data['Distribution']+data['Take on']+data['air']+data['Chance creation']


    data = data.sort_values('Rank', ascending = False)

    return data    



# In[10]:


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader('Российские лиги')
data = nas0('РПЛ', 1)
data2 = data
datagr = data2
data2 = data2[['Name','Position','sh','Defence','Recovery','Distribution','Take on','air','Chance creation','Rank','Team', 'League', 'Age', 'Minutes_played']].sort_values('Rank', ascending = False).head(1000)


# In[2]:


league = st.multiselect("Выбор лиг:", data2["League"].unique(), default=data2["League"].unique())
position = st.multiselect("Выбор позиций:",data2["Position"].unique(), default=data2["Position"].unique())

df_selection_league = data2.query("League ==@league")
optionals_team = st.expander("Команды", False)
team = optionals_team.multiselect("Select the Team:",df_selection_league["Team"].unique(),default=df_selection_league["Team"].unique())

optionals = st.expander("Возраст", True)
age = optionals.slider(
    'Выберите диапазон',
    float(data2["Age"].min()), float(data2["Age"].max()), (float(data2["Age"].min()), float(data2["Age"].max())))


# In[15]:


df_selection = data2.query(
    "Position == @position & Team ==@team & League ==@league & @age[0]<=Age<=@age[1]"
)


# In[16]:


df_selection = df_selection.reset_index(drop=True).style.background_gradient(cmap='PiYG')


# In[17]:


st.dataframe(df_selection)


# In[ ]:


st.text('*sh-удары, defence-защита(отборы, подкаты), recovery-возврат владения(перехваты, свободные подборы), take on - дриблинг и ведение мяча, \nair - игра в воздухе, chance creation - создание моментов, rank - итоговый рейтинг')


# # Technical profiles

# In[18]:


tp = data2


# In[19]:


for i in tp:
    tp['gr_air_blocker'] = tp['Defence'] + tp['air']
    tp['air_bl_pm'] = tp['air'] + tp['Distribution']
    tp['air_bl_fm'] = tp['air'] + tp['Recovery']
    tp['gr_bl_fm'] = tp['Defence'] + tp['Recovery']
    tp['gr_bl_pm'] = tp['Defence'] + tp['Distribution']
    tp['fm_pm'] = tp['Recovery'] + tp['Distribution']
    tp['def_inf'] = tp['air'] + tp['Defence'] + tp['Recovery'] + tp['Take on']
    tp['pm_inf'] = tp['Distribution'] + tp['Take on']
    tp['def_sh'] = tp['air'] + tp['Defence'] + tp['Recovery'] + tp['Distribution'] +tp['sh']
    tp['pm_creator'] = tp['Distribution'] + tp['Chance creation']
    tp['inf_creator'] = tp['Take on'] + tp['Chance creation']
    tp['sh_inf'] = tp['sh'] + tp['Take on']
    tp['sh_creator'] = tp['air'] + tp['Recovery']
    tp['ar_tm'] = tp['air'] + tp['Take on'] + tp['Chance creation'] + tp['Distribution'] + 2*tp['sh']
    tp['tm_sh'] = tp['sh'] + tp['air']


# In[20]:


tp = tp[['Name','Position', 'Team', 'League', 'gr_air_blocker','air_bl_pm','air_bl_fm','gr_bl_fm','gr_bl_pm','fm_pm','def_inf','pm_inf','def_sh','pm_creator','inf_creator','sh_inf','sh_creator','ar_tm','tm_sh']]


# In[21]:


st.title('Технический профиль')
st.markdown('Подробнее в исследовании [CIES Football Observatory Monthly Report n°74 - April 2022](https://www.football-observatory.com/IMG/sites/mr/mr74/en/)')

prof={'профиль':['Ground-to-air blocker','Air blocker playmaker','Air blocker filter man','Ground blocker filter man',
                 'Ground blocker playmaker', 'Filter man playmaker', 'Defensive infiltrator', 'Playmaker infiltrator',
                 'Defensive shooter', 'Playmaker creator', 'Infiltrator creator', 'Shooter infiltrator',
                 'Shooter creator','Allrounder target man', 'Target man shooter'],
      'качества': ['защита, игра головой', 'игра в воздухе, игра в пас', 'возврат мяча, игра в воздухе', 'защита, возврат мяча', 'защита, игра в пас',
                'возврат мяча команде, игра в пас', 'защита, игра в воздухе, возврат мяча команде, дриблинг и ведение мяча', 'игра в пас, дриблинг и ведение мяча', 
                 'защита, игра в воздухе, возврат мяча, игра в пас, удары', 'игра в пас, создание моментов', 
                'дриблинг и ведение мяча, создание моментов', 'удары, дриблинг и ведение мяча', 'удары, создание моментов', 
                 'создание моментов, игра в воздухе, дриблинг и ведение мяча, игра в пас, удары', 'игра в воздухе, удары']}

prof1 = pd.DataFrame(prof)
st.dataframe(prof1, use_container_width = True)
    
position1 = st.multiselect("Выбор позиции:",tp["Position"].unique())
league1 = st.multiselect("Выбор лиги:",tp["League"].unique())


# In[22]:


tp_selection = tp.query(
    "Position == @position1 & League==@league1"
)


# In[23]:


tp_selection = tp_selection.reset_index(drop=True).style.background_gradient(cmap='PiYG')
st.dataframe(tp_selection, use_container_width = True)


# # Индивидуальный профиль

# In[24]:


st.title('Индивидуальный профиль футболиста')


# In[25]:


#player profile
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

df_selection1 = df_selection1[['sh','Defence','Recovery','Distribution','Take on','air','Chance creation']]

df_selection1.at[1,'sh'] = dfpp['sh'].median()
df_selection1.at[1,'Defence'] = dfpp['Defence'].median()
df_selection1.at[1,'Recovery'] = dfpp['Recovery'].median()
df_selection1.at[1,'Distribution'] = dfpp['Distribution'].median()
df_selection1.at[1,'Take on'] = dfpp['Take on'].median()
df_selection1.at[1,'air'] = dfpp['air'].median()
df_selection1.at[1,'Chance creation'] = dfpp['Chance creation'].median()
df_selection1 = df_selection1.reset_index(drop=True)


# In[27]:


if st.button('Анализировать'):
    ps=datagr.query("Name == @name & Position==@positionpp")
    ps = ps[['Name','Age', 'Foot', 'Height','Nationality', 'Team']]
    st.dataframe(ps)
    
    fig = go.Figure()

    colors= ["tomato", "dodgerblue", "yellow"]
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
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    )
    st.plotly_chart(fig)
else:
    st.write('Для начала выберите футболиста')


# # графики

# In[28]:


st.title('График')
st.markdown('График строится исходя из выбранной позиции в индивидуальном профиле игрока')


# In[29]:



GA_selection1 = datagr.query("League == @leaguepp & Position == @positionpp")
i = st.selectbox("Выберите 1 параметр:", GA_selection1[['Matches played',
                                                       'Minutes_played', 'Starting lineup appearances',
                                                        'Goals', 'Assists', 'Expected assists', 'Offsides', 'Yellow cards',
                                                       'Red cards', 'Shots', 'Penalty',
                                                     'Penalty kicks scored. %', 'Passes',
                                                       'Accurate passes. %', 'Key passes', 'Crosses', 'Accurate crosses. %',
                                                       'Lost balls', 'Lost balls in own half', 'Ball recoveries',
                                                       "Ball recoveries in opponent's half", 'xG (Expected goals)',
                                                       'Challenges', 'Challenges won. %', 'Attacking challenges',
                                                       'Air challenges', 'Dribbles', 'Tackles', 'Ball interceptions',
                                                       'Free ball pick ups', 'Defensive challenges',
                                                       'Challenges in defence won. %', 'Age', 'Height','Air challenges won. %',
                                                         'Challenges in attack won. %',
                                                         'Successful dribbles. %', 'Tackles won. %','Fouls',
                                                       'Fouls suffered', 'Key passes accurate',"Shots on target. %"]].columns)
j = st.selectbox("Выберите 2 параметр:", GA_selection1[['Matches played',
                                                       'Minutes_played', 'Starting lineup appearances',
                                                        'Goals', 'Assists', 'Expected assists', 'Offsides', 'Yellow cards',
                                                       'Red cards', 'Shots', 'Penalty',
                                                     'Penalty kicks scored. %', 'Passes',
                                                       'Accurate passes. %', 'Key passes', 'Crosses', 'Accurate crosses. %',
                                                       'Lost balls', 'Lost balls in own half', 'Ball recoveries',
                                                       "Ball recoveries in opponent's half", 'xG (Expected goals)',
                                                       'Challenges', 'Challenges won. %', 'Attacking challenges',
                                                       'Air challenges', 'Dribbles', 'Tackles', 'Ball interceptions',
                                                       'Free ball pick ups', 'Defensive challenges',
                                                       'Challenges in defence won. %', 'Age', 'Height','Air challenges won. %',
                                                         'Challenges in attack won. %',
                                                         'Successful dribbles. %', 'Tackles won. %','Fouls',
                                                       'Fouls suffered', 'Key passes accurate',"Shots on target. %"]].columns)



# In[30]:

st.set_option('deprecation.showPyplotGlobalUse', False)
legend = st.checkbox('Players')
if st.button('start'):
    def petalplot(GA_selection1, i, j):

        def plotlabel(xvar, yvar, label):
            if legend:
                if label == f'{name}':
                    ax.text(xvar+0.002, yvar, label, c ='red', size=18)
                else:
                    ax.text(xvar+0.002, yvar, label, c ='black', size=15)
            else:
                if label == f'{name}':
                    ax.text(xvar+0.002, yvar, label, c ='red', size=18)

        if legend:        
            fig = plt.figure(figsize=(20,20))
        else:
            fig = plt.figure(figsize=(10,10))
        ax = sns.scatterplot(x = i, y = j, data=GA_selection1)

        GA_selection1.apply(lambda x: plotlabel(x[i],  x[j], x['Name']), axis=1)
        plt.title('RPL Analytics')
        plt.xlabel(i)
        plt.ylabel(j)
        ax.vlines(GA_selection1[i].median(), GA_selection1[j].min(), GA_selection1[j].max())
        ax.hlines(GA_selection1[j].median(), GA_selection1[i].min(), GA_selection1[i].max())
        ax.grid()


    #petalplot(GA_selection1, 'Defensive challenges','Challenges in defence won. %')
    st.pyplot(petalplot(GA_selection1, i,j))


# In[ ]:




