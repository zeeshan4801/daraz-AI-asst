import streamlit as st
import faiss
import pickle
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from groq import Groq


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Daraz AI Support Assistant",
    page_icon="🛒",
    layout="wide"
)


# ==============================
# DARAZ UI STYLE
# ==============================

st.markdown(
"""
<style>

.stApp {
    background-color:#f7f7f7;
}

h1 {
    color:#f85606;
}

[data-testid="stSidebar"] {
    background-color:#ffffff;
}

</style>
""",
unsafe_allow_html=True
)



# ==============================
# LOAD FAISS DATABASE
# ==============================

@st.cache_resource
def load_database():

    index_path = "faiss_index/index.faiss"
    metadata_path = "faiss_index/metadata.pkl"


    # Debug check

    if not os.path.exists(index_path):

        st.error(
            """
            FAISS index not found.

            Required file:
            faiss_index/index.faiss

            Please upload your FAISS folder to GitHub.
            """
        )

        st.stop()



    if not os.path.exists(metadata_path):

        st.error(
            """
            Metadata file not found.

            Required file:
            faiss_index/metadata.pkl
            """
        )

        st.stop()



    # Load FAISS

    index = faiss.read_index(
        index_path
    )


    # Load metadata

    with open(
        metadata_path,
        "rb"
    ) as f:

        metadata = pickle.load(f)



    # Load embedding model only
    # No PDF processing happens here

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index, metadata, model



index, metadata, embedding_model = load_database()



# ==============================
# GROQ CLIENT
# ==============================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)



# ==============================
# SIDEBAR
# ==============================

st.sidebar.title(
    "🛒 Daraz Knowledge Base"
)


departments = [

    "All Sections",
    "returns",
    "delivery",
    "refunds",
    "sellers",
    "payments",
    "customer_support"

]


selected_section = st.sidebar.selectbox(
    "Search Section",
    departments
)



st.sidebar.markdown("---")


st.sidebar.success(
f"""
FAISS Database Loaded

Vectors:
{index.ntotal}

Section:
{selected_section}
"""
)



# ==============================
# SEARCH FUNCTION
# ==============================

def retrieve_chunks(
        query,
        department,
        k=3
):


    query_vector = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )


    scores, ids = index.search(
        np.array(query_vector),
        len(metadata)
    )


    results=[]


    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        item = metadata[idx]


        if department != "All Sections":

            if item["department"] != department:
                continue



        results.append(
            {
                "text": item["text"],
                "department": item["department"],
                "source": item["source_file"],
                "score": float(score)
            }
        )



        if len(results) >= k:
            break



    return results



# ==============================
# GROQ ANSWER GENERATION
# ==============================

def generate_answer(
        question,
        context
):


    prompt = f"""

You are Daraz Customer Support Operations Assistant.

Answer the customer question using ONLY the provided knowledge base.

If information is missing, clearly say:
"I could not find this information in the Daraz knowledge base."


Knowledge Base:

{context}


Customer Question:

{question}


Give a professional and helpful support response.

"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
                "role":"system",
                "content":
                "You are a Daraz customer support expert."
            },

            {
                "role":"user",
                "content":prompt
            }

        ],

        temperature=0.2,

        max_tokens=700

    )


    return response.choices[0].message.content



# ==============================
# CHAT UI
# ==============================


st.title(
"🛒 Daraz Customer Support Operations Assistant"
)


st.caption(
"AI powered support assistant using FAISS Knowledge Base + Groq LLM"
)



if "messages" not in st.session_state:

    st.session_state.messages=[]



for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )



question = st.chat_input(
"Ask a Daraz operations question..."
)



if question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )



    with st.chat_message("user"):

        st.write(question)



    documents = retrieve_chunks(
        question,
        selected_section
    )


    if documents:

        context="\n\n".join(
            [
                doc["text"]
                for doc in documents
            ]
        )


        answer = generate_answer(
            question,
            context
        )


    else:

        answer = (
            "I could not find relevant information "
            "in the selected knowledge base section."
        )



    with st.chat_message("assistant"):

        st.write(answer)


        with st.expander(
            "📚 Sources Used"
        ):

            for doc in documents:

                st.write(
f"""
**Department:** {doc['department']}

**File:** {doc['source']}

**Similarity:** {doc['score']:.3f}
"""
                )



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )
