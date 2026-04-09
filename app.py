import streamlit as st
import os
import uuid
import tempfile
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carico le variabili d'ambiente (il token di Hugging Face)
load_dotenv()

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Corporate RAG Assistant", page_icon="📚", layout="centered")
st.title("📚 Corporate Assistant (RAG)")
st.write("Carica un documento aziendale (PDF) e fammi qualsiasi domanda al riguardo!")

# Inizializzo lo stato della sessione per la chat e il database vettoriale
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar per il caricamento del documento
with st.sidebar:
    st.header("📄 Gestione Documenti")
    
    uploaded_files = st.file_uploader("Carica uno o più file PDF", type="pdf", accept_multiple_files=True)
    
    if st.button("Elabora Documenti"):
        if uploaded_files:
            # NOVITÀ: Svuoto esplicitamente il database vecchio per evitare cloni!
            st.session_state.vector_store = None 
            
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

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(all_docs)

                # NOVITÀ: Modello Multilingua perfetto per l'Italiano!
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                
                # Do un nome casuale al database ogni volta che clicco Elabora
                nome_univoco = f"db_{uuid.uuid4().hex}"

                st.session_state.vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name=nome_univoco)
                
            #st.success(f"✅ Letti {len(splits)} frammenti puliti da {len(uploaded_files)} documenti!")
            st.success(f"Tutti i {len(uploaded_files)} documenti sono stati indicizzati con successo! ✅")
        else:
            st.warning("Per favore, carica almeno un PDF prima di cliccare su Elabora.")

# Se il documento è stato caricato, mostro la chat
if st.session_state.vector_store is not None:

    # Configurazione del Modello Linguistico (LLM) tramite Hugging Face
    # Aggiornato a un modello molto più capace per il ragionamento
    base_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct", # Ottima alternativa: "mistralai/Mistral-7B-Instruct-v0.3"
        max_new_tokens=512,
        temperature=0.1, # Quasi a zero per renderlo freddo, analitico e preciso
        do_sample=False
    )
    llm = ChatHuggingFace(llm=base_llm)

    # Configurazione del Prompt per il RAG "Militare"
    system_prompt = (
        "Sei un assistente aziendale severo, preciso e ultra-conciso. "
        "Devi rispondere alla domanda dell'utente basandoti ESCLUSIVAMENTE sui frammenti di contesto forniti qui sotto. "
        "Segui rigorosamente queste REGOLE:\n"
        "1. Vai dritto al punto: rispondi in massimo 3 o 4 frasi brevi.\n"
        "2. NON INVENTARE NULLA: Se la risposta non è chiaramente scritta nel contesto, devi rispondere ESATTAMENTE così: 'Mi dispiace, ma questa informazione non è presente nei documenti caricati.'\n"
        "3. Non aggiungere convenevoli, opinioni personali o frasi introduttive.\n\n"
        "Contesto fornito:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Creazione della "Catena" RAG (Collego il database, il prompt e il modello)
    #retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}) # Prende i 3 paragrafi più rilevanti
    # Uso MMR per forzare la diversità dei risultati 

    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.8}
    )

    #retriever = st.session_state.vector_store.as_retriever(
    #    search_type="similarity",
    #    search_kwargs={"k": 5}
    #)   

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Mostro la cronologia della chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input dell'utente
    user_query = st.chat_input("Chiedi qualcosa sul documento...")

    if user_query:

        # Mostro la domanda dell'utente
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        
        # Genero la risposta
        #with st.chat_message("assistant"):
        #    with st.spinner("Cerco nel documento..."):
        #        response = rag_chain.invoke({"input": user_query})
        #        # Zephyr a volte ripete la domanda nell'output, formatto per pulizia
        #        answer = response["answer"]
        #        if "Helpful Answer:" in answer:
        #            answer = answer.split("Helpful Answer:")[-1].strip()
        #            
        #        st.markdown(answer)
        #        
        #        # Salvo la risposta nella cronologia
        #        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        

        # Genero la risposta
        with st.chat_message("assistant"):
            with st.spinner("Cerco nel documento..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                if "Helpful Answer:" in answer:
                    answer = answer.split("Helpful Answer:")[-1].strip()
                    
                st.markdown(answer)
                
                # tendina a comparsa per vedere i documenti pescati!
                with st.expander("🔍 Clicca qui per vedere cosa ha letto l'AI (Modalità Raggi X)"):
                    docs_trovati = retriever.invoke(user_query)
                    if not docs_trovati:
                        st.warning("Il database non ha trovato nessun paragrafo rilevante!")
                    for i, doc in enumerate(docs_trovati):
                        st.markdown(f"**Frammento {i+1}:**")
                        st.info(doc.page_content)
                
                # Salvo la risposta nella cronologia
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Per iniziare, carica un file PDF aziendale (es. una policy, un manuale o un report) dalla barra laterale.")