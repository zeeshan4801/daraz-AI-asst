import streamlit as st
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Daraz Customer Support Assistant",
    page_icon="🛒",
    layout="wide"
)


# ==========================
# DARAZ STYLE CSS
# ==========================

st.markdown(
"""
<style>

.main {
    background-color:#f7f7f7;
}


h1 {
    color:#f85606;
}


.sidebar .sidebar-content {
    background-color:#ffffff;
}


.chat-message {
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
}


</style>
""",
unsafe_allow_html=True
)



# ==========================
# LOAD FAISS
# ==========================

@st.cache_resource
def load_database():

    index = faiss.read_index(
        "faiss_index/index.faiss"
    )

    with open(
        "faiss_index/metadata.pkl",
        "rb"
    ) as f:

        metadata = pickle.load(f)


    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index, metadata, model



index, metadata, embedding_model = load_database()



# ==========================
# GROQ CLIENT
# ==========================


client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)



# ==========================
# SIDEBAR
# ==========================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/7/7e/Daraz_Logo.png",
    width=150
)


st.sidebar.title(
    "Knowledge Base"
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


selected_department = st.sidebar.selectbox(
    "Choose Section",
    departments
)



st.sidebar.markdown("---")

st.sidebar.info(
"""
🛒 Daraz Customer Support Operations Assistant

Ask questions about:
- Orders
- Payments
- Delivery
- Returns
- Refunds
- Sellers
"""
)



# ==========================
# RETRIEVAL FUNCTION
# ==========================

def search_documents(query, department, k=3):


    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )


    scores, ids = index.search(
        np.array(query_embedding),
        len(metadata)
    )


    results=[]


    for score, idx in zip(scores[0], ids[0]):

        item = metadata[idx]


        if department != "All Sections":

            if item["department"] != department:
                continue


        results.append(
            {
                "score":float(score),
                "text":item["text"],
                "department":item["department"],
                "source":item["source_file"]
            }
        )


        if len(results)==k:
            break


    return results



# ==========================
# GROQ RESPONSE
# ==========================

def generate_answer(question, context):


    prompt=f"""
You are Daraz Customer Support Operations Assistant.

Answer ONLY using the provided knowledge base.

If the answer is not available in the context,
say:
"I could not find this information in the Daraz knowledge base."

Knowledge Base:

{context}


Customer Question:

{question}

Provide a clear professional support answer.
"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role":"system",
                "content":
                "You are a helpful Daraz support assistant."
            },
            {
                "role":"user",
                "content":prompt
            }
        ],

        temperature=0.2,

        max_tokens=800
    )


    return response.choices[0].message.content



# ==========================
# MAIN UI
# ==========================


st.title(
"🛒 Daraz Customer Support Operations Assistant"
)


st.caption(
"AI assistant powered by Daraz Knowledge Base + FAISS + Groq"
)



if "messages" not in st.session_state:

    st.session_state.messages=[]



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])



question = st.chat_input(
"Ask your Daraz support question..."
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



    results = search_documents(
        question,
        selected_department
    )


    context="\n\n".join(
        [
            r["text"]
            for r in results
        ]
    )


    answer = generate_answer(
        question,
        context
    )



    with st.chat_message("assistant"):

        st.write(answer)


        with st.expander(
            "📚 Retrieved Sources"
        ):

            for r in results:

                st.write(
                f"""
                **Department:** {r['department']}

                **Source:** {r['source']}

                **Similarity:** {r['score']:.3f}
                """
                )


    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )
