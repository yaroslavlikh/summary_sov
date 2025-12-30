import telebot
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

import config
from handlers.handlers import load_handlers
from database.init_db import init_db
import os
import uvicorn


TOKEN = config.get_key_bot()

bot = telebot.TeleBot(TOKEN)
load_handlers(bot)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("DB initialized")
    yield
    print("Shutdown")

app = FastAPI(
    title="SummarySov Bot",
    lifespan=lifespan
)

port = int(os.environ.get("PORT", 8080))
uvicorn.run(app, host="0.0.0.0", port=port)

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = telebot.types.Update.de_json(data)
    bot.process_new_updates([update])
    return {"ok": True}

@app.get("/")
def root():
    return {"status": "bot is alive"}