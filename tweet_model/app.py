# streamlit for frontend

import streamlit as st
import tensorflow as tf
from PIL import Image

img = Image.open("disaster.jpg")


st.image(img)

st.write("# Disaster Tweet Prediction")


tweet = st.text_input(
        "Enter tweet to classify",
        "Enter or paste a tweet here",
        key="placeholder",
    )


#load model
@st.cache_resource
def cache_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return (model)

model = cache_model(r"C:\Users\HP\Desktop\tweet_model\model_6_SavedModel_format")


# Load TF Hub Sentence Encoder SavedModel
# model = tf.keras.models.load_model("model_6_SavedModel_format")

def predict_on_sentence(model, sentence):
  """
  Uses model to make a prediction on sentence.

  Returns the sentence, the predicted label and the prediction probability.
  """
  pred_prob = model.predict([sentence])
  pred_label = tf.squeeze(tf.round(pred_prob)).numpy()

  st.write(f"## {sentence}")
  if pred_label == 0:
     st.write(f"This is a non-disaster tweet with probability: {round((1 - pred_prob[0][0]) * 100, 2)}%")

  else:
     st.write(f"This is a disaster tweet with probability: {round(pred_prob[0][0]*100, 2)}%")


if tweet:
    predict_on_sentence(model, tweet)