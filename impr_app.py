import streamlit as st
import os
import uuid
import tempfile
import numpy as np
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Reranker
from sentence_transformers import CrossEncoder

# 1. Carico le variabili d'ambiente (il token di Hugging Face)
load_dotenv()

# ─────────────────────────────────────────────
# Configurazione della pagina Streamlit
# ─────────────────────────────────────────────
st.set_page_config(page_title="Corporate RAG Assistant", page_icon="📚", layout="centered")
st.title("📚 Corporate Assistant (RAG)")
st.write("Carica un documento aziendale (PDF) e fammi qualsiasi domanda al riguardo!")

# ─────────────────────────────────────────────
# Stato della sessione
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "splits" not in st.session_state:
    st.session_state.splits = None

# ─────────────────────────────────────────────
# MIGLIORAMENTO 2 — Reranker cross-encoder
# Caricato una volta sola e messo in cache
# ─────────────────────────────────────────────
@st.cache_resource
def load_reranker():
    """
    Cross-encoder multilingua per riordinare i chunk recuperati.
    mmarco-mMiniLMv2-L12-H384-v1 è addestrato su mMARCO (MS MARCO
    tradotto in 13 lingue, italiano incluso).
    Il precedente ms-marco-MiniLM-L-6-v2 era solo inglese e
    riordinava i chunk italiani in modo errato, peggiorando i risultati.
    """
    return CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device="cpu",
    )

def rerank_docs(query: str, docs: list, top_k: int = 4) -> list:
    """Riordina i documenti per rilevanza rispetto alla query."""
    if not docs:
        return []
    reranker = load_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# ─────────────────────────────────────────────
# MIGLIORAMENTO 1 — Embedding model migliore
# Caricato una volta sola e messo in cache
# ─────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    """
    intfloat/multilingual-e5-large forzato su CPU:
    - Evita OOM su MPS (Apple Silicon con RAM GPU limitata)
    - batch_size=16 → processa i chunk a gruppi per contenere
      il picco di memoria durante l'indicizzazione
    - normalize_embeddings=True → cosine similarity più stabile
    
    Se la RAM è ancora insufficiente, sostituisci model_name con
    "intfloat/multilingual-e5-base" (dimensioni dimezzate, ~560 MB).
    """
class E5Embeddings(HuggingFaceEmbeddings):
    """
    Wrapper di HuggingFaceEmbeddings che aggiunge i prefissi obbligatori
    per la famiglia intfloat/multilingual-e5-*.

    Il modello è stato addestrato con:
      - "query: <testo>"   → per le domande in retrieval
      - "passage: <testo>" → per i chunk indicizzati
    Senza questi prefissi i vettori prodotti sono incompatibili tra loro
    e il retrieval diventa peggiore del MiniLM originale.

    La versione installata di langchain_huggingface non accetta
    query_instruction/embed_instruction nel costruttore, quindi
    sovrascriviamo i due metodi direttamente.
    """
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = ["passage: " + t for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query("query: " + text)


@st.cache_resource
def load_embeddings():
    return E5Embeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,
        },
    )

# ─────────────────────────────────────────────
# Sidebar — caricamento e indicizzazione PDF
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Gestione Documenti")

    uploaded_files = st.file_uploader(
        "Carica uno o più file PDF", type="pdf", accept_multiple_files=True
    )

    if st.button("Elabora Documenti"):
        if uploaded_files:
            # Svuota il database precedente per evitare cloni
            st.session_state.vector_store = None
            st.session_state.bm25_retriever = None
            st.session_state.splits = None

            with st.spinner("Elaborazione e unione dei documenti in corso..."):
                all_docs = []

                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    os.unlink(tmp_file_path)

                # Chunk di dimensione bilanciata: 800 chars è un buon
                # compromesso tra retrieval preciso (chunk piccoli) e
                # contesto sufficiente al modello (chunk grandi).
                # Il precedente 400 era troppo piccolo e produceva chunk
                # privi di contesto autonomo.
                child_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", ".", " ", ""],
                )
                splits = child_splitter.split_documents(all_docs)

                embeddings = load_embeddings()

                # Nome univoco per evitare collisioni tra sessioni
                nome_univoco = f"db_{uuid.uuid4().hex}"
                st.session_state.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    collection_name=nome_univoco,
                )

                # ── MIGLIORAMENTO 5 — BM25 retriever (keyword) ─────────
                # Salvo i splits in sessione per costruire il
                # BM25Retriever (non serializzabile in Chroma)
                st.session_state.bm25_retriever = BM25Retriever.from_documents(splits)
                st.session_state.bm25_retriever.k = 6
                st.session_state.splits = splits

            st.success(
                f"Tutti i {len(uploaded_files)} documenti sono stati indicizzati con successo! ✅"
            )
        else:
            st.warning("Per favore, carica almeno un PDF prima di cliccare su Elabora.")

# ─────────────────────────────────────────────
# Area principale — chat
# ─────────────────────────────────────────────
if st.session_state.vector_store is not None:

    # ── LLM ────────────────────────────────────────────────────────────
    base_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )
    llm = ChatHuggingFace(llm=base_llm)

    # ── MIGLIORAMENTO 3 — Prompt con chat history ───────────────────────
    system_prompt = (
        "Sei un assistente aziendale severo, preciso e ultra-conciso. "
        "Devi rispondere alla domanda dell'utente basandoti ESCLUSIVAMENTE "
        "sui frammenti di contesto forniti qui sotto. "
        "Segui rigorosamente queste REGOLE:\n"
        "1. Vai dritto al punto: rispondi in massimo 3 o 4 frasi brevi.\n"
        "2. NON INVENTARE NULLA: Se la risposta non è chiaramente scritta "
        "nel contesto, rispondi ESATTAMENTE: "
        "'Mi dispiace, ma questa informazione non è presente nei documenti caricati.'\n"
        "3. Non aggiungere convenevoli, opinioni personali o frasi introduttive.\n"
        "4. Se la domanda fa riferimento a messaggi precedenti, usali per "
        "contestualizzare, ma rispondi sempre dal contesto documentale.\n\n"
        "Contesto fornito:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),   # ← chat history iniettata qui
        ("human", "{input}"),
    ])

    # ── MIGLIORAMENTO 5 — Ensemble Retriever (dense + keyword) ─────────
    dense_retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.8},
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[st.session_state.bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],  # 60% semantico, 40% keyword
    )

    # ── Cronologia chat ────────────────────────────────────────────────
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Input utente ───────────────────────────────────────────────────
    user_query = st.chat_input("Chiedi qualcosa sul documento...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Cerco nel documento..."):

                # ── Step 1: recupera candidati con Ensemble ─────────────
                raw_docs = ensemble_retriever.invoke(user_query)

                # ── Step 2: MIGLIORAMENTO 2 — rerank cross-encoder ──────
                docs_reranked = rerank_docs(user_query, raw_docs, top_k=4)

                # ── Step 3: costruisci contesto dal top-k reranked ───────
                context = "\n\n---\n\n".join(
                    [doc.page_content for doc in docs_reranked]
                )

                # ── Step 4: MIGLIORAMENTO 3 — costruisci chat history ───
                # Converti la cronologia Streamlit in messaggi LangChain
                lc_history = []
                # Escludi l'ultimo messaggio (è la domanda corrente)
                for msg in st.session_state.chat_history[:-1]:
                    if msg["role"] == "user":
                        lc_history.append(HumanMessage(content=msg["content"]))
                    else:
                        lc_history.append(AIMessage(content=msg["content"]))

                # ── Step 5: invoca il modello ────────────────────────────
                formatted_prompt = prompt.format_messages(
                    input=user_query,
                    context=context,
                    chat_history=lc_history,
                )
                response = llm.invoke(formatted_prompt)
                answer = response.content

                # Pulizia residui da alcuni modelli HF
                if "Helpful Answer:" in answer:
                    answer = answer.split("Helpful Answer:")[-1].strip()

                st.markdown(answer)

                # ── Modalità Raggi X — chunk usati ──────────────────────
                with st.expander("🔍 Clicca qui per vedere cosa ha letto l'AI (Modalità Raggi X)"):
                    if not docs_reranked:
                        st.warning("Il database non ha trovato nessun paragrafo rilevante!")
                    for i, doc in enumerate(docs_reranked):
                        source = doc.metadata.get("source", "N/A")
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(
                            f"**Frammento {i+1}** — "
                            f"sorgente: `{os.path.basename(str(source))}`, "
                            f"pagina: `{page}`"
                        )
                        st.info(doc.page_content)

                # Salva in cronologia
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

else:
    st.info(
        "👈 Per iniziare, carica un file PDF aziendale "
        "(es. una policy, un manuale o un report) dalla barra laterale."
    )