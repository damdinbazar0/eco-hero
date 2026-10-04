from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os
import uuid

app = Flask(__name__)


# ==========================================
# LOAD YOLO26 MODEL
# ==========================================

model = YOLO("model/best.pt")


# ==========================================
# YOLO26 CLASSES
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
    "trash"
]


# ==========================================
# 9 CLASSES → 5 MAIN CATEGORIES
# ==========================================

categories = {

    "battery": {
        "name_mn": "Аюултай хог",
        "name_en": "Hazardous Waste",
        "emoji": "🔋",
        "color": "danger",

        "message_mn":
            "Энэ батерей байна! 🔋 "
            "Аюултай хог тул тусгай хогийн саванд хийгээрэй!",

        "message_en":
            "This is a battery! 🔋 "
            "Please put it in the hazardous waste bin!"
    },

    "biological": {
        "name_mn": "Хүнсний хог",
        "name_en": "Food Waste",
        "emoji": "🍎",
        "color": "food",

        "message_mn":
            "Энэ бол хүнсний хог байна! 🍎 "
            "Хүнсний хогийн саванд хийгээрэй!",

        "message_en":
            "This is food waste! 🍎 "
            "Please put it in the food waste bin!"
    },

    "cardboard": {
        "name_mn": "Цаасан хог",
        "name_en": "Paper Waste",
        "emoji": "📦",
        "color": "paper",

        "message_mn":
            "Энэ бол бор цаас байна! 📦 "
            "Цаасны хогийн саванд хийгээрэй!",

        "message_en":
            "This is cardboard! 📦 "
            "Please put it in the paper waste bin!"
    },

    "paper": {
        "name_mn": "Цаасан хог",
        "name_en": "Paper Waste",
        "emoji": "📄",
        "color": "paper",

        "message_mn":
            "Энэ бол цаас байна! 📄 "
            "Цаасны хогийн саванд хийгээрэй!",

        "message_en":
            "This is paper! 📄 "
            "Please put it in the paper waste bin!"
    },

    "plastic": {
        "name_mn": "Дахивар хог",
        "name_en": "Recyclable Waste",
        "emoji": "🧴",
        "color": "recycle",

        "message_mn":
            "Энэ бол хуванцар байна! 🧴 "
            "Дахивар гэсэн тэмдэглэгээтэй хогийн саванд хийгээрэй!",

        "message_en":
            "This is plastic! 🧴 "
            "Please put it in the recycling bin!"
    },

    "metal": {
        "name_mn": "Дахивар хог",
        "name_en": "Recyclable Waste",
        "emoji": "🥫",
        "color": "recycle",

        "message_mn":
            "Энэ бол металл байна! 🥫 "
            "Дахивар гэсэн тэмдэглэгээтэй хогийн саванд хийгээрэй!",

        "message_en":
            "This is metal! 🥫 "
            "Please put it in the recycling bin!"
    },

    "glass": {
        "name_mn": "Дахивар хог",
        "name_en": "Recyclable Waste",
        "emoji": "🫙",
        "color": "recycle",

        "message_mn":
            "Энэ бол шил байна! 🫙 "
            "Дахивар гэсэн тэмдэглэгээтэй хогийн саванд хийгээрэй!",

        "message_en":
            "This is glass! 🫙 "
            "Please put it in the recycling bin!"
    },

    "clothes": {
        "name_mn": "Бусад хог",
        "name_en": "Other Waste",
        "emoji": "👕",
        "color": "other",

        "message_mn":
            "Энэ хувцас байна! 👕 "
            "Бусад ангиллын хогийн саванд хийгээрэй!",

        "message_en":
            "This is clothing! 👕 "
            "Please put it in the other waste bin!"
    },

    "trash": {
        "name_mn": "Бусад хог",
        "name_en": "Other Waste",
        "emoji": "🗑️",
        "color": "other",

        "message_mn":
            "Энэ бол ердийн хог байна! 🗑️ "
            "Бусад ангиллын хогийн саванд хийгээрэй!",

        "message_en":
            "This is general waste! 🗑️ "
            "Please put it in the other waste bin!"
    }
}


# ==========================================
# PREDICT IMAGE WITH YOLO26
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    results = model(
        image,
        verbose=False
    )

    result_data = results[0]

    # Highest-confidence class
    predicted_index = result_data.probs.top1

    predicted_class = result_data.names[
        predicted_index
    ]

    confidence = float(
        result_data.probs.top1conf
    ) * 100

    # Get category information
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

        # ==================================
        # CHECK IMAGE
        # ==================================

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
        # YOLO PREDICTION
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


    # ======================================
    # SHOW PAGE
    # ======================================

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
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )