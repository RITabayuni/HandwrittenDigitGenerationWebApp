
import streamlit as st
import torch
import torch.nn.functional as F
from model import CVAE, one_hot 
import matplotlib.pyplot as plt


model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=torch.device("cpu")))
model.eval()

st.title("MNIST Digit Generator")
digit = st.selectbox("Select a digit (0–9)", list(range(10)))

if st.button("Generate 5 images"):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    with torch.no_grad():
        y = one_hot(torch.tensor([digit]*5))
        z = torch.randn(5, model.latent_dim)
        samples = model.decode(z, y).view(-1, 28, 28)
        for i in range(5):
            axs[i].imshow(samples[i], cmap='gray')
            axs[i].axis('off')
    st.pyplot(fig)
