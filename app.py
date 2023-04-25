import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(page_title="Аналитика", page_icon=":bar_chart:", layout="wide")

hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

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
    authenticator.logout('Logout', 'main')
    if username == 'kvas':
        from app1 import *

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')



