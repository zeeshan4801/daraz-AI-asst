import streamlit as st
import faiss
import pickle
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Daraz AI Support Assistant",
    page_icon="🛒",
    layout="wide"
)


# ==========================================
# UI STYLE
# ==========================================

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

    background:white;

}


.stChatMessage {

    border-radius:15px;

}


button {

    border-radius:20px !important;

}

</style>

""",
unsafe_allow_html=True
)



# ==========================================
# LOAD FAISS DATABASE
# ==========================================


@st.cache_resource
def load_database():


    index_path="faiss_index/index.faiss"

    metadata_path="faiss_index/metadata.pkl"



    if not os.path.exists(index_path):

        st.error(
        "FAISS index missing. Upload faiss_index/index.faiss"
        )

        st.stop()



    if not os.path.exists(metadata_path):

        st.error(
        "Metadata missing. Upload faiss_index/metadata.pkl"
        )

        st.stop()



    index=faiss.read_index(
        index_path
    )



    with open(
        metadata_path,
        "rb"
    ) as f:

        metadata=pickle.load(f)



    model=SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )



    return index, metadata, model




index, metadata, embedding_model = load_database()



# ==========================================
# GROQ
# ==========================================


client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)



# ==========================================
# SIDEBAR
# ==========================================


st.sidebar.markdown(
"""
# 🛒 Daraz AI Assistant

Customer Support + Marketplace Knowledge Assistant

---
"""
)



sections=[

"All Sections",
"returns",
"delivery",
"refunds",
"sellers",
"payments",
"customer_support"

]



selected_section=st.sidebar.radio(

    "📚 Knowledge Section",

    sections

)



st.sidebar.markdown("---")



st.sidebar.success(

f"""
🟢 System Online


Vectors:
{index.ntotal}


Mode:
Hybrid RAG + AI


Section:
{selected_section}

"""

)



st.sidebar.markdown("---")


st.sidebar.markdown(
"""
### Example Questions

📦 How do I return an order?

🚚 How can I track delivery?

💳 Payment failed

🏪 How to become seller?

🛒 Daraz vs Amazon?
"""
)



# ==========================================
# RETRIEVAL
# ==========================================


def retrieve_chunks(
        query,
        department,
        k=5
):


    query_vector=embedding_model.encode(

        [query],

        normalize_embeddings=True

    )


    scores, ids=index.search(

        np.array(query_vector),

        len(metadata)

    )



    results=[]



    for score, idx in zip(
        scores[0],
        ids[0]
    ):


        item=metadata[idx]



        if department!="All Sections":

            if item["department"]!=department:

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




# ==========================================
# AI GENERATION
# ==========================================


def generate_answer(

        question,

        context=None,

        use_rag=True

):



    if use_rag:


        prompt=f"""

You are Daraz Customer Support Operations Assistant.

Answer using the provided internal knowledge base.


Knowledge Base:

{context}



Customer Question:

{question}



Rules:

- Use only relevant information.
- Do not invent policies.
- Be professional and concise.

"""


    else:


        prompt=f"""

You are a helpful ecommerce assistant.

Answer this general Daraz question.


Question:

{question}



Rules:

- Provide useful general information.
- Do not make up exact commission rates,
fees, or policies.
- Mention when official seller documentation is required.

"""




    response=client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[

            {

            "role":"system",

            "content":
            "You are an expert ecommerce assistant."

            },

            {

            "role":"user",

            "content":prompt

            }

        ],

        temperature=0.3,

        max_tokens=800

    )



    return response.choices[0].message.content





# ==========================================
# MAIN PAGE
# ==========================================


st.title(
"🛒 Daraz Customer Support Operations Assistant"
)



st.caption(
"""
AI assistant for Daraz operations, sellers, payments, delivery and marketplace questions.
"""
)



st.info(
"""
Try asking:

• How do I return my order?

• What payment options are available?

• What is Daraz commission?

• Difference between Daraz and Amazon?

"""
)




# ==========================================
# CHAT MEMORY
# ==========================================


if "messages" not in st.session_state:

    st.session_state.messages=[]




for msg in st.session_state.messages:


    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )



question=st.chat_input(

"Ask your Daraz question..."

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



    results=retrieve_chunks(

        question,

        selected_section

    )



    # ==============================
    # ROUTING LOGIC
    # ==============================


    if results and results[0]["score"] >= 0.35:


        context="\n\n".join(

            [

            r["text"]

            for r in results

            ]

        )


        answer=generate_answer(

            question,

            context,

            True

        )


        used_sources=True



    else:


        answer=generate_answer(

            question,

            None,

            False

        )


        used_sources=False





    with st.chat_message("assistant"):


        st.write(answer)



        if used_sources:


            with st.expander(
                "📚 Knowledge Sources"
            ):


                for r in results:


                    st.markdown(

f"""
**Department:** {r['department']}

**Source:** {r['source']}

**Similarity:** {r['score']:.3f}

---
"""
                    )



    st.session_state.messages.append(

        {

        "role":"assistant",

        "content":answer

        }

    )
