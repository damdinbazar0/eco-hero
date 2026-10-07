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
# 9 CLASSES → 4 MAIN CATEGORIES
# ==========================================

categories = {

    # ЦААС
    "paper": {
        "name_mn": "Цаас",
        "name_en": "Paper",
        "emoji": "📄",
        "color": "paper",

        "message_mn":
            "Энэ бол цаас байна! 📄 "
            "Цэнхэр хогийн саванд хийгээрэй!",

        "message_en":
            "This is paper! 📄 "
            "Please put it in the blue bin!"
    },

    "cardboard": {
        "name_mn": "Цаас",
        "name_en": "Paper",
        "emoji": "📦",
        "color": "paper",

        "message_mn":
            "Энэ бол картон цаас байна! 📦 "
            "Цэнхэр өнгийн хогийн саванд хийгээрэй!",

        "message_en":
            "This is cardboard! 📦 "
            "Please put it in the blue bin!"
    },


    # ШИЛ, ЛААЗ
    "glass": {
        "name_mn": "Шил, лааз",
        "name_en": "Glass & Cans",
        "emoji": "🫙",
        "color": "glass",

        "message_mn":
            "Энэ бол шил байна! "
            "Улбар шар өнгийн хогийн саванд хийгээрэй!",

        "message_en":
            "This is glass! "
            "Please put it in the orange bin!"
    },

    "metal": {
        "name_mn": "Шил, лааз",
        "name_en": "Glass & Cans",
        "emoji": "🥫",
        "color": "glass",

        "message_mn":
            "Энэ бол лааз эсвэл металл байна! "
            "Улбар шар өнгийн хогийн саванд хийгээрэй!",

        "message_en":
            "This is a can or metal item! "
            "Please put it in the orange bin!"
    },


    # ХУВАНЦАР
    "plastic": {
        "name_mn": "Хуванцар",
        "name_en": "Plastic",
        "emoji": "♻️",
        "color": "plastic",

        "message_mn":
            "Энэ бол хуванцар байна! "
            "Ногоон өнгийн хогийн саванд хийгээрэй!",

        "message_en":
            "This is plastic! "
            "Please put it in the green bin!"
    },


    # БУСАД ХОГ
    "battery": {
        "name_mn": "Бусад хог",
        "name_en": "Other Waste",
        "emoji": "🗑️",
        "color": "other",

        "message_mn":
            "Энэ бол бусад хог байна! "
            "Хар өнгийн саванд хийгээрэй!",

        "message_en":
            "This is other waste! "
            "Please put it in the black bin!"
    },

    "biological": {
        "name_mn": "Бусад хог",
        "name_en": "Other Waste",
        "emoji": "🗑️",
        "color": "other",

        "message_mn":
            "Энэ бол бусад хог байна! "
            "Хар өнгийн саванд хийгээрэй!",

        "message_en":
           "This is other waste! "
            "Please put it in the black bin!"
    },

    "clothes": {
        "name_mn": "Бусад хог",
        "name_en": "Other Waste",
        "emoji": "🗑️",
        "color": "other",

        "message_mn":
            "Энэ бол бусад хог байна!"
            "Хар өнгийн саванд хийгээрэй!",

        "message_en":
            "This is other waste! "
            "Please put it in the black bin!"
    },

    "trash": {
        "name_mn": "Бусад хог",
        "name_en": "Other Waste",
        "emoji": "🗑️",
        "color": "other",

        "message_mn":
            "Энэ бол бусад хог байна! "
            "Хар өнгийн саванд хийгээрэй!",

        "message_en":
            "This is other waste! "
            "Please put it in the black bin!"
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

        # CHECK IMAGE
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


        # SAVE UPLOADED IMAGE
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


        # YOLO PREDICTION
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
# COMIC PAGE
# ==========================================

@app.route("/comic")
def comic():
    return render_template("comic.html")


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
