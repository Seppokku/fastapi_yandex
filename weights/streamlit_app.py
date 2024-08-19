import requests
import streamlit as st
from my_funcs import clean
st.title("Simple FastAPI application")

tab1, tab2= st.tabs(['Image', 'Text'])

def main():
    with tab1:
        # create input form
        image = st.file_uploader("Classify an image", type=['jpg', 'jpeg'])
        if st.button("Classify!") and image is not None:
            # show image
            st.image(image)
            # format data for input format
            files = {"file": image.getvalue()}
            # send data and get the result
            res = requests.post("http://127.0.0.1:8000/clf_image", files=files)#.json()
            #st.write(res)
            st.write(f'Class name: {res.json()["class_name"]}, class index: {res.json()["class_index"]}')

    with tab2:
        txt = st.text_input('Classify text')
        txt = str(clean(txt))
        if st.button('Classify'):
            text = {'text' : txt}
            res = requests.post("http://127.0.0.1:8000/clf_text", json=text)#.json()
            st.write(res.json()["label"])
            st.write(res.json()["prob"])
        

# if __name__ == '__main__':
main()