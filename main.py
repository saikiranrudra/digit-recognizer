import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

if __name__ == "__main__":
    model = load_model('./model/model.h5')
    st.title("✍ Hand Written Digit Recognizer")

    left, right = st.beta_columns(2)

    with left:
        st.write("Write you digit here ")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=20,
            stroke_color="rgba(0, 0, 0, 1)",
            update_streamlit=True,
            background_color="rgba(255, 255, 255, 1)",
            width=308,
            height=308,
            drawing_mode="freedraw",
            key="canvas"
        )

    with right:
        if canvas_result.image_data is not None:
            raw = canvas_result.image_data.astype("float32")
            image = cv2.resize(raw, dsize=(28, 28))
            image = image[:, :, 0] + image[:, :, 1] + image[:, :, 2] + image[:, :, 3]
            image = image / 1020
            image = np.absolute(image)
            image = np.where(image > 0, 1 - image, 0)
            image = np.array([image])
            fig, ax = plt.subplots()
            # sns.heatmap(image, annot=True, ax=ax)
            # st.pyplot(fig)

            st.write(image.shape)
            st.write("Prediction output")
            sns.barplot(
                x=["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"],
                y=list(model.predict(image))[0],
                ax=ax
            )
            st.pyplot(fig)
