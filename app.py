import streamlit as st
import faiss
import pickle
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from groq import Groq


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Daraz AI Support Assistant",
    page_icon="🛒",
    layout="wide"
)


# =====================================
# CUSTOM CSS
# =====================================

st.markdown(
"""
<style>

.stApp {
    background:#f7f7f7;
}


/* Main title */

h1 {
    color:#f85606;
    font-weight:700;
}


/* Sidebar */

[data-testid="stSidebar"] {

    background:white;

}


/* Chat boxes */

.stChatMessage {

    border-radius:15px;

}


/* Buttons */

.stButton button {

    border-radius:20px;
    border:none;
    background:#f85606;
    color:white;

}


</style>

""",
unsafe_allow_html=True
)



# =====================================
# LOAD FAISS DATABASE
# =====================================


@st.cache_resource
def load_database():

    index_path = "faiss_index/index.faiss"
    metadata_path = "faiss_index/metadata.pkl"


    if not os.path.exists(index_path):

        st.error(
        """
        ❌ FAISS index not found.

        Please upload:
        faiss_index/index.faiss
        """
        )

        st.stop()



    if not os.path.exists(metadata_path):

        st.error(
        """
        ❌ Metadata file missing.

        Please upload:
        faiss_index/metadata.pkl
        """
        )

        st.stop()



    index = faiss.read_index(
        index_path
    )


    with open(
        metadata_path,
        "rb"
    ) as f:

        metadata = pickle.load(f)



    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    return index, metadata, model




index, metadata, embedding_model = load_database()



# =====================================
# GROQ
# =====================================


client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)



# =====================================
# SIDEBAR
# =====================================


st.sidebar.markdown(
"""
# 🛒 Daraz AI Assistant

Your intelligent customer support operations assistant.

---
"""
)



sections = [

"All Sections",
"returns",
"delivery",
"refunds",
"sellers",
"payments",
"customer_support"

]


selected_section = st.sidebar.radio(
    "📚 Select Knowledge Section",
    sections
)



st.sidebar.markdown("---")


st.sidebar.success(
f"""
🟢 System Online


FAISS Database:
Loaded


Knowledge Vectors:
{index.ntotal}


Active Section:
{selected_section}


AI Model:
GPT OSS 120B
"""
)



st.sidebar.markdown("---")


st.sidebar.markdown(
"""
### 💡 Example Questions

• How can I track my order?

• What payment methods are available?

• How can I request refund?

• How to return product?

• Seller requirements?
"""
)




# =====================================
# RETRIEVAL
# =====================================


def retrieve_chunks(
    query,
    department,
    k=3
):


    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )


    scores, ids = index.search(
        np.array(query_embedding),
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

            "text":item["text"],

            "department":item["department"],

            "source":item["source_file"],

            "score":float(score)

            }

        )



        if len(results)>=k:

            break



    return results




# =====================================
# GROQ ANSWER
# =====================================


def generate_answer(
    question,
    context
):


    prompt=f"""

You are Daraz Customer Support Operations Assistant.

Answer only from the provided knowledge base.

If information is unavailable, explain politely.

Knowledge Base:

{context}


Customer Question:

{question}


Provide a concise professional support answer.

"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {
            "role":"system",
            "content":
            "You are an expert Daraz support agent."
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




# =====================================
# MAIN UI
# =====================================


st.title(
"🛒 Daraz Customer Support Operations Assistant"
)



st.markdown(
"""
### 🤖 AI-powered support assistant

Ask questions related to:

📦 Orders  
🚚 Delivery  
💳 Payments  
🔄 Returns  
💰 Refunds  
🏪 Sellers  

"""
)



# Quick questions


st.subheader(
"⚡ Quick Questions"
)



quick_questions=[

"How can I track my order?",

"What payment options are available?",

"How do I request a refund?",

"What are seller requirements?"

]



cols=st.columns(4)



for col,q in zip(cols,quick_questions):

    if col.button(q):

        st.session_state.question=q





if "messages" not in st.session_state:

    st.session_state.messages=[]




for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )




question = st.chat_input(
"Ask your Daraz question..."
)



if "question" in st.session_state:

    question=st.session_state.question

    del st.session_state.question




if question:



    st.session_state.messages.append(

        {
        "role":"user",
        "content":question
        }

    )



    with st.chat_message("user"):

        st.write(question)



    results = retrieve_chunks(

        question,

        selected_section

    )



    if results:


        context="\n\n".join(

            [
            r["text"]

            for r in results

            ]

        )


        answer=generate_answer(

            question,

            context

        )


    else:


        answer="""

I could not find this information in the current Daraz knowledge base.

Available sections:

• Returns
• Delivery
• Refunds
• Payments
• Sellers
• Customer Support

Please ask a question related to these topics.

"""




    with st.chat_message("assistant"):


        st.write(answer)



        if results:


            with st.expander(
                "📚 Knowledge Sources"
            ):


                for r in results:


                    st.markdown(

f"""
**Department:** {r['department']}

**Source:** {r['source']}

**Similarity Score:** {r['score']:.3f}

---
"""
                    )



    st.session_state.messages.append(

        {
        "role":"assistant",
        "content":answer
        }

    )
