### USE ST.METRIC !!!!
### CHECK THIS FOR ACCOUNT CREATION STORAGE
### Consider creating a sign up screen later

# Import libraries
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title='Login',
    page_icon='⭐',
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
    )

st.header('')    #to move the login box downwards
st.header('')    #to move the login box downwards
col1, mid, col2 = st.columns([2,0.5,1.5])
with col1:
    st.write('')  #to keep the container intact
with col2:
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
            unsafe_allow_html=True
        )

        no_sidebar_style = """
            <style>
                div[data-testid="stSidebarNav"] {display: none;}
            </style>
        """
        st.markdown(no_sidebar_style, unsafe_allow_html=True)
    add_bg_from_local('background3.png')

    # Show general information on partners in the sidebar
    image = Image.open('RLTDxKFH.jpg')
    image2 = Image.open('RLTDxKFH2.jpeg')
    st.sidebar.image(image)
    st.sidebar.subheader('About Baitak Rewards')
    st.sidebar.write('''Baitak Rewards Program is a Kuwait Finance House Loyalty
                    Program that rewards you for using your KFH cards.''')
    st.sidebar.subheader('About Kuwait Finance House')
    st.sidebar.write('''Kuwait Finance House (KFH)is a pioneer of Shari’a-compliant
                    banking and Islamic finance, offering a range of financial
                    products and services in banking and other sub-sectors.''')
    st.sidebar.subheader('About Related')
    st.sidebar.write('''Related is an award-winning MarTech agency that provides
                    customized loyalty and rewards solutions for enterprises
                    & small businesses''')
    st.sidebar.image(image2)

    # Determine flow based on password input
    st.subheader('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if password != 'MS42':
        if password == '':
            st.warning('Enter Username & Password')
            st.stop()
        elif password != '':
            st.error('Access Denied - Incorrect Username/Password')
            st.stop()
    elif password == 'MS42':
        if username == 'admin':
            st.success('Login Successful  🎉')
            st.info('Please click the sidebar button to get started ↖️')
            no_sidebar_style = """
                <style>
                    div[data-testid="stSidebarNav"] {display: initial;}
                </style>
            """
            st.markdown(no_sidebar_style, unsafe_allow_html=True)
