import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Verificar se o diretório 'temp' existe, caso contrário, criar
if not os.path.exists('temp'):
    os.makedirs('temp')

# Carregar o modelo
model = load_model('modelos/cat_dog_classifier_basic.h5')

st.title("Classificador de Gato ou Cachorro")
st.write("Carregue uma imagem de um gato ou um cachorro para obter a classificação.")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar a imagem
    return img_array

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = os.path.join("temp", uploaded_file.name)
    
    # Salvar o arquivo carregado temporariamente no servidor
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(file_path, caption="Imagem carregada.", use_column_width=True)
    
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = 'Cachorro'
    else:
        result = 'Gato'
    
    st.write(f"Predição: {result}")
