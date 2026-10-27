import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)

from dotenv import load_dotenv

load_dotenv()


# Streamlit Configuration
st.set_page_config(
    page_title="LangChain: YouTube & Website Summarizer",
    page_icon="🦜"
)

st.title("🦜 LangChain: YouTube & Website Summarizer")
st.subheader("Summarize YouTube Videos or Website Content")


# Sidebar
with st.sidebar:
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password"
    )


# URL Input
generic_url = st.text_input(
    "Enter YouTube or Website URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


# Prompt Template
prompt_template = """
Provide a clear and detailed summary of the following content
in approximately 300 words.

Content:
{text}

Summary:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# Summarize Button
if st.button("Summarize Content"):

    # Validate Inputs
    if not groq_api_key.strip() or not generic_url.strip():

        st.error("Please enter your Groq API key and URL.")

    elif not validators.url(generic_url):

        st.error("Please enter a valid YouTube or Website URL.")

    else:

        try:

            with st.spinner("Loading and summarizing content..."):

                # Initialize Groq Model
                llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    groq_api_key=groq_api_key
                )

                # Load YouTube or Website Content
                if (
                    "youtube.com/" in generic_url.lower()
                    or "youtu.be/" in generic_url.lower()
                ):

                    st.info("Loading YouTube video transcript...")

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False,
                        language=["en-US", "en", "hi"],
                        translation="en",
                        continue_on_failure=False
                    )

                else:

                    st.info("Loading website content...")

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )

                # Load Documents
                docs = loader.load()

                if not docs or not any(
                    doc.page_content.strip() for doc in docs
                ):
                    st.error("No content found to summarize.")
                    st.stop()

                # Summarization Chain
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                # Generate Summary
                output_summary = chain.invoke({
                    "input_documents": docs
                })

                # Display Summary
                st.success("Summary Generated Successfully!")

                st.write(output_summary["output_text"])

        except Exception as e:

            st.error("Unable to summarize the content.")
            st.exception(e)