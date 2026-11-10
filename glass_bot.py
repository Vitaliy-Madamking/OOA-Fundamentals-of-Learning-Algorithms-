"""Бот для определения сорта стекла по химическому составу (датасет Glass Identification, UCI / Kaggle uciml/glass).
Запуск Telegram-бота:  pip install python-telegram-bot joblib scikit-learn pandas
                       export BOT_TOKEN=<токен от @BotFather>;  python glass_bot.py
Проверка без Telegram: python glass_bot.py --console
"""
import os, re, sys
import joblib
import pandas as pd

BUNDLE = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "glass_model.joblib"))
PIPE, FEATURES, NAMES = BUNDLE["pipeline"], BUNDLE["features"], BUNDLE["type_names"]
HELP = ("Пришлите состав стекла в процентах, например:\n"
        "Na=13.4 Mg=3.5 Al=1.2 K=0.5 Ca=8.7 Ba=0\n"
        f"Нужные признаки: {', '.join(FEATURES)}. Пример можно получить командой /example")
EXAMPLE = "Na=14.4 Mg=0.5 Al=2.1 K=0.3 Ca=8.5 Ba=1.0"


def parse(text):
    vals = {k.capitalize(): float(v.replace(",", ".")) for k, v in re.findall(r"([A-Za-z]{1,2})\s*[=:]\s*(-?\d+(?:[.,]\d+)?)", text)}
    missing = [f for f in FEATURES if f not in vals]
    if missing:
        raise ValueError("Не хватает значений: " + ", ".join(missing))
    if any(not 0 <= vals[f] <= 100 for f in FEATURES):
        raise ValueError("Содержание оксидов должно быть в пределах 0-100 %")
    return pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)


def answer(text):
    """Возвращает текст ответа бота на сообщение пользователя."""
    try:
        x = parse(text)
    except ValueError as e:
        return f"⚠️ {e}\n\n{HELP}"
    proba = PIPE.predict_proba(x)[0]
    order = proba.argsort()[::-1][:3]
    classes = PIPE.classes_
    lines = [f"🔎 Наиболее вероятный сорт: {NAMES[classes[order[0]]]} (тип {classes[order[0]]}), уверенность {proba[order[0]]:.0%}", "Другие варианты:"]
    lines += [f"  • {NAMES[classes[i]]}: {proba[i]:.0%}" for i in order[1:]]
    if proba[order[0]] < 0.5:
        lines.append("⚠️ Уверенность низкая — результат следует перепроверить лабораторно.")
    return "\n".join(lines)


def main():
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

    async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Привет! Я определяю сорт стекла по химическому составу.\n" + HELP)

    async def example(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(EXAMPLE + "\n\n" + answer(EXAMPLE))

    async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(answer(update.message.text))

    app = Application.builder().token(os.environ["BOT_TOKEN"]).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("example", example))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()


if __name__ == "__main__":
    if "--console" in sys.argv:
        print(HELP)
        while True:
            try:
                print(answer(input("> ")))
            except (EOFError, KeyboardInterrupt):
                break
    else:
        main()
