import streamlit as st
import pandas as pd
import google.generativeai as genai
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ---------------------------
# TEXT SUMMARIZER (Sumy)
# ---------------------------
def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


# ---------------------------
# GEMINI SETUP & EXPAND
# ---------------------------
model = None

def setup_gemini(api_key):
    """Configure Gemini with API key."""
    global model
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")


def expand_summary(summary, title):
    """Generate expanded summary using Gemini (streaming)."""
    if model is None:
        yield "❌ Gemini model not initialized."
        return

    prompt = f"""
    You are a professional book explainer. Expand the following summary
    into a detailed, engaging, and well-structured explanation.

    Book Title: {title}
    Short Summary: {summary}
    """
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk and hasattr(chunk, "text"):
            yield chunk.text


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="📚 AI Library Assistant", page_icon="📖")

st.title("📚 AI Library Assistant")
st.write("Use *ML (TextRank)* to summarize books and *Google Gemini* to expand them!")

# API key input
api_key = st.text_input("🔑 Enter your Google Gemini API Key:", type="password")

if api_key:
    setup_gemini(api_key)

    # Upload dataset
    uploaded_file = st.file_uploader("📂 Upload your books.csv (must have 'title' and 'description' columns)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Validate dataset
        if "title" not in df.columns or "description" not in df.columns:
            st.error("❌ Your CSV must contain 'title' and 'description' columns.")
        else:
            st.success("✅ Dataset loaded successfully!")
            st.write("### 📖 Available Books:")
            selected_title = st.selectbox("Choose a book:", df["title"].tolist())

            if selected_title:
                desc = df.loc[df["title"] == selected_title, "description"].values[0]

                # Summarize
                st.subheader("🧠 ML Summary:")
                summary = summarize_text(desc)
                st.write(summary)

                # Expand
                if st.button("✨ Expand with Gemini"):
                    with st.spinner("Expanding with Gemini..."):
                        expanded_text_container = st.empty()
                        expanded_text = ""
                        for chunk in expand_summary(summary, selected_title):
                            expanded_text += chunk
                            expanded_text_container.markdown(expanded_text)
else:
    st.warning("Please enter your Google Gemini API Key to continue.")