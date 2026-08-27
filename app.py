
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)


# ==========================================
# LOAD CNN MODEL
# ==========================================

model = tf.keras.models.load_model(
    "model/waste_model.keras"
)


# ==========================================
# CNN CLASSES
# ==========================================

class_names = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]


# ==========================================
# 10 CLASSES → 5 MAIN CATEGORIES
# ==========================================

categories = {

    "battery": {
        "name": "Аюултай хог",
        "emoji": "🔋",
        "color": "danger",
        "message":
            "Энэ батерей байна! 🔋 "
            "Аюултай хог тул тусгай хогийн саванд хийгээрэй!"
    },

    "biological": {
        "name": "Хүнсний хог",
        "emoji": "🍎",
        "color": "food",
        "message":
            "Энэ бол хүнсний хог байна! 🍎 "
            "Хүнсний хогийн саванд хийгээрэй!"
    },

    "cardboard": {
        "name": "Цаасан хог",
        "emoji": "📦",
        "color": "paper",
        "message":
            "Энэ бол бор цаас байна! 📦 "
            "Цаасны хогийн саванд хийгээрэй!"
    },

    "paper": {
        "name": "Цаасан хог",
        "emoji": "📄",
        "color": "paper",
        "message":
            "Энэ бол цаас байна! 📄 "
            "Цаасны хогийн саванд хийгээрэй!"
    },

    "plastic": {
        "name": "Дахивар хог",
        "emoji": "🧴",
        "color": "recycle",
        "message":
            "Энэ бол хуванцар байна! 🧴 "
            "Дахивар гэсэн тэмдэглэгээтэй хогийн саванд хийгээрэй!"
    },

    "metal": {
        "name": "Дахивар хог",
        "emoji": "🥫",
        "color": "recycle",
        "message":
            "Энэ бол металл байна! 🥫 "
            "Дахивар гэсэн тэмдэглэгээтэй хогийн саванд хийгээрэй!"
    },

    "glass": {
        "name": "Дахивар хог",
        "emoji": "🫙",
        "color": "recycle",
        "message":
            "Энэ бол шил байна! 🫙 "
            "Дахивар гэсэн тэмдэглэгээтэй хогийн саванд хийгээрэй!"
    },

    "clothes": {
        "name": "Бусад хог",
        "emoji": "👕",
        "color": "other",
        "message":
            "Энэ хувцас байна! 👕 "
            "Бусад ангилалын хогийн саванд хийгээрэй!"
    },


    "trash": {
        "name": "Бусад хог",
        "emoji": "🗑️",
        "color": "other",
        "message":
            "Энэ бол ердийн хог байна! 🗑️ "
            "Бусад ангилалын хогийн саванд хийгээрэй!"
    }
}


# ==========================================
# PREDICT IMAGE
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    image = image.resize((180, 180))

    image_array = np.array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        float(
            prediction[0][predicted_index]
        ) * 100
    )

    result = categories[
        predicted_class
    ].copy()

    result["confidence"] = round(
        confidence,
        1
    )

    result["detected"] = predicted_class

    return result


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ==========================================
# AI CLASSIFICATION PAGE
# ==========================================

@app.route(
    "/classify",
    methods=["GET", "POST"]
)
def classify():

    result = None

    image_path = None

    error = None


    if request.method == "POST":

        if "image" not in request.files:

            error = "Зураг сонгоно уу!"

            return render_template(
                "index.html",
                result=result,
                image_path=image_path,
                error=error
            )


        file = request.files["image"]


        if file.filename == "":

            error = "Зураг сонгоно уу!"

            return render_template(
                "index.html",
                result=result,
                image_path=image_path,
                error=error
            )


        # ==================================
        # SAVE UPLOADED IMAGE
        # ==================================

        os.makedirs(
            "static/uploads",
            exist_ok=True
        )


        filename = (
            str(uuid.uuid4())
            + ".jpg"
        )


        image_path = os.path.join(
            "static/uploads",
            filename
        )


        file.save(
            image_path
        )


        # ==================================
        # PREDICT
        # ==================================

        try:

            image = Image.open(
                image_path
            )

            result = predict_image(
                image
            )

        except Exception as e:

            error = (
                "Зургийг боловсруулахад "
                "алдаа гарлаа: "
                + str(e)
            )


    return render_template(
        "index.html",
        result=result,
        image_path=image_path,
        error=error
    )


# ==========================================
# WASTE INFORMATION PAGE
# ==========================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
