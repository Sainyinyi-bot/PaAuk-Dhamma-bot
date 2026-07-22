import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


if not TELEGRAM_BOT_TOKEN or not GOOGLE_API_KEY:
    raise ValueError(
        "Environment Variables မသတ်မှတ်ရသေးပါ။"
    )


# Hugging Face Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# FAISS Load
vectorstore = FAISS.load_local(
    ".",
    embeddings,
    allow_dangerous_deserialization=True
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "မင်္ဂလာပါ 🙏\n\n"
        "ဓမ္မစာအုပ်များနှင့် ပတ်သက်သော မေးခွန်းများကို မေးနိုင်ပါသည်။"
    )


async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE):

    question = update.message.text

    await update.message.reply_text(
        "ရှာဖွေပြီး ဖြေကြားနေပါသည်... 🙏"
    )

    docs = retriever.invoke(question)

    context_text = ""

    sources = []

    for doc in docs:
        context_text += doc.page_content + "\n\n"

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "?"
        )

        sources.append(
            f"{source} - Page {page}"
        )


    prompt = f"""
သင်သည် ဓမ္မစာပေ အကူအညီပေးသော AI ဖြစ်သည်။

အောက်ပါ စာအုပ်မှ အချက်အလက်ကိုသာ အခြေခံပြီး
မြန်မာဘာသာဖြင့် ယဉ်ကျေးစွာ ဖြေပါ။

စာအုပ်အချက်အလက်:
{context_text}

မေးခွန်း:
{question}
"""


    response = llm.invoke(prompt)


    answer_text = response.content

    answer_text += "\n\n📚 ကိုးကားချက်:\n"

    answer_text += "\n".join(
        list(set(sources))
    )


    await update.message.reply_text(
        answer_text
    )


def main():

    app = Application.builder().token(
        TELEGRAM_BOT_TOKEN
    ).build()


    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            answer
        )
    )


    print("Bot Running...")

    app.run_polling()


if __name__ == "__main__":
    main()
