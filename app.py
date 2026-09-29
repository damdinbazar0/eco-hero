
from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os
import uuid

app = Flask(__name__)


# ==========================================
# LOAD YOLO26 MODEL
# ==========================================

model = YOLO(
    "runs/classify/train/weights/best.pt"
)


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
            "Бусад ангиллын хогийн саванд хийгээрэй!"
    },

    "trash": {
        "name": "Бусад хог",
        "emoji": "🗑️",
        "color": "other",
        "message":
            "Энэ бол ердийн хог байна! 🗑️ "
            "Бусад ангиллын хогийн саванд хийгээрэй!"
    }
}


# ==========================================
# PREDICT IMAGE WITH YOLO26
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    # YOLO26 performs its own resizing/preprocessing
    results = model(
        image,
        verbose=False
    )

    result_data = results[0]

    # Get highest-confidence class
    predicted_index = result_data.probs.top1

    predicted_class = result_data.names[
        predicted_index
    ]

    confidence = float(
        result_data.probs.top1conf
    ) * 100

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

