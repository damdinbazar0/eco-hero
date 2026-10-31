import asyncio
from pathlib import Path
import edge_tts

VOICES = {
    "mn": "mn-MN-YesuiNeural",
    "en": "en-US-JennyNeural",
}

MESSAGES = {
    "paper": {
        "mn": "Энэ бол цаас байна! Цэнхэр хогийн саванд хийгээрэй!",
        "en": "This is paper! Please put it in the blue bin!",
    },
    "cardboard": {
        "mn": "Энэ бол картон цаас байна! Цэнхэр өнгийн хогийн саванд хийгээрэй!",
        "en": "This is cardboard! Please put it in the blue bin!",
    },
    "glass": {
        "mn": "Энэ бол шил байна! Улбар шар өнгийн хогийн саванд хийгээрэй!",
        "en": "This is glass! Please put it in the orange bin!",
    },
    "metal": {
        "mn": "Энэ бол лааз эсвэл металл байна! Улбар шар өнгийн хогийн саванд хийгээрэй!",
        "en": "This is a can or metal item! Please put it in the orange bin!",
    },
    "plastic": {
        "mn": "Энэ бол хуванцар байна! Ногоон өнгийн хогийн саванд хийгээрэй!",
        "en": "This is plastic! Please put it in the green bin!",
    },
    "battery": {
        "mn": "Энэ бол батарей байна! Тусгай батарей цуглуулах саванд хийгээрэй!",
        "en": "This is a battery! Please take it to a special battery collection bin!",
    },
    "biological": {
        "mn": "Энэ бол бусад хог байна! Хар өнгийн саванд хийгээрэй!",
        "en": "This is other waste! Please put it in the black bin!",
    },
    "clothes": {
        "mn": "Энэ бол бусад хог байна! Хар өнгийн саванд хийгээрэй!",
        "en": "This is other waste! Please put it in the black bin!",
    },
    "trash": {
        "mn": "Энэ бол бусад хог байна! Хар өнгийн саванд хийгээрэй!",
        "en": "This is other waste! Please put it in the black bin!",
    },
}

async def main():
    for lang, voice in VOICES.items():
        folder = Path("static/audio") / lang
        folder.mkdir(parents=True, exist_ok=True)
        for waste, translations in MESSAGES.items():
            destination = folder / f"{waste}.mp3"
            tts = edge_tts.Communicate(
                text=translations[lang], voice=voice, rate="-5%"
            )
            await tts.save(str(destination))
            print(f"Created: {destination}")
    print("All 18 audio files created!")

if __name__ == "__main__":
    asyncio.run(main())
