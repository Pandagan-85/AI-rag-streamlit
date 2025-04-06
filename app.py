"""Streamlit application combining Meal Planner and Document Q&A using LangChain and RAG."""
import json
import os

import streamlit as st

# Updated imports for modern LangChain
try:
    # Try modern imports first
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
    import chromadb
except ImportError:
    # Fallback to older imports if needed
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chat_models import ChatOpenAI
    import chromadb
    # Note: streamlit_chromadb_connection might still be needed

# Core LangChain imports with fallbacks
try:
    from langchain.document_loaders.base import BaseLoader
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chains import LLMChain
except ImportError:
    # Modern imports
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    from langchain.chains.llm import LLMChain
    from langchain.memory.buffer import ConversationBufferMemory

import re

# Customize the page layout
st.set_page_config(page_title="AI Assistant with RAG",
                   page_icon="🤖", layout="wide")

# Initial session state setup for app selection
if "current_app" not in st.session_state:
    st.session_state.current_app = "📄 Document Q&A"

# -------------------------------------------------------------------------------
# DOCUMENT Q&A FUNCTIONS
# -------------------------------------------------------------------------------


def load_document(file):
    """Load document from file based on its extension."""
    import os
    name, extension = os.path.splitext(file)

    if extension.lower() == '.pdf':
        try:
            from langchain_community.document_loaders import PyPDFLoader
            print(f'Loading {file}')
            loader = PyPDFLoader(file)
        except ImportError:
            from langchain.document_loaders import PyPDFLoader
            print(f'Loading {file}')
            loader = PyPDFLoader(file)
    elif extension.lower() == '.docx':
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            print(f'Loading {file}')
            loader = Docx2txtLoader(file)
        except ImportError:
            from langchain.document_loaders import Docx2txtLoader
            print(f'Loading {file}')
            loader = Docx2txtLoader(file)
    elif extension.lower() == '.txt':
        try:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file)
        except ImportError:
            from langchain.document_loaders import TextLoader
            loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def chunk_document_data(data, chunk_size=512, chunk_overlap=100):  # Aumento overlap
    """Split document into chunks with specified size and overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Separatori espliciti per un chunking migliore
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def create_document_embeddings(chunks, api_key=None):
    """Create embeddings for document chunks using ChromaDB connection."""
    if not api_key:
        return None

    # Ensure the directory exists
    chroma_path = "/tmp/.chroma"
    os.makedirs(chroma_path, exist_ok=True)

    # Configuration for ChromaDB connection
    configuration = {
        "client": "PersistentClient",
        "path": chroma_path
    }

    # Create Streamlit connection
    conn = st.connection("chromadb", type=ChromadbConnection, **configuration)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Collection name
    collection_name = "document_collection"

    # Create or get the collection
    try:
        # Use chromadb client directly
        chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Create a new collection
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=collection_name,
            client=chroma_client
        )

        return vector_store
    except Exception as e:
        st.error(f"Error creating document embeddings: {e}")
        return None


def normalize_query(query):
    """Preprocess query to improve retrieval quality."""
    # Convert to lowercase for case-insensitive search
    query = query.lower()
    # Remove excessive whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    return query


def ask_document(vector_store, q, k=3, api_key=None):
    """Get answer to question using vector store and LLM. Handles summary requests."""
    # Verifica se la chiave API è stata fornita
    if not api_key:
        return "Please provide an OpenAI API key in the sidebar to use this feature."

    # Verifica se la domanda è una richiesta di riassunto
    summary_keywords = ["riassumi", "riassunto", "riepilogo", "sintetizza", "sintesi",
                        "summary", "summarize", "summarise", "overview", "recap"]

    lower_q = q.lower()
    is_summary_request = any(
        keyword in lower_q for keyword in summary_keywords)

    if is_summary_request:
        # Se è una richiesta di riassunto, genera un riassunto completo
        # Usa un k più alto per i riassunti
        return generate_document_summary(vector_store, k=10, api_key=api_key)
    else:
        # Altrimenti, procedi con la normale domanda e risposta
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, api_key=api_key)
        retriever = vector_store.as_retriever(
            search_type='similarity', search_kwargs={'k': k})
        chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever)

        answer = chain.invoke(q)
        return answer['result']


# Implementazione della funzione di riassunto
def generate_document_summary(vector_store, k=10, api_key=None):
    """Generate a document summary using document chunks."""
    # Verifica se la chiave API è stata fornita
    if not api_key:
        return "Please provide an OpenAI API key in the sidebar to use this feature."

    llm = ChatOpenAI(model='gpt-3.5-turbo',
                     # Lower temperature for more consistent summaries
                     temperature=0.3, api_key=api_key)
    retriever = vector_store.as_retriever(
        search_type='mmr',  # Cambiato a MMR per una migliore diversità nei risultati
        # Recupera più documenti per selezione
        search_kwargs={"k": k, "fetch_k": k*2}
    )

    # Create a custom chain for summarization
    summary_prompt = PromptTemplate(
        input_variables=["context"],
        template="""You are an expert at summarizing documents. 
Please create a comprehensive summary of the following document content.
Focus on the main topics, key points, and overall structure.
Make sure to capture the essence of the document.
Be thorough and informative.

DOCUMENT CONTENT:
{context}

COMPREHENSIVE SUMMARY:"""
    )

    # Strategia migliorata per ottenere parti rappresentative del documento
    docs_beginning = retriever.get_relevant_documents(
        "beginning of document introduction")
    docs_middle = retriever.get_relevant_documents(
        "main content central information")
    docs_end = retriever.get_relevant_documents("conclusion summary end")
    docs_key = retriever.get_relevant_documents(
        "key points important information")

    # Combina i documenti evitando duplicati
    all_docs = []
    seen_content = set()

    for doc_list in [docs_beginning, docs_middle, docs_end, docs_key]:
        for doc in doc_list:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)

    # Limita il numero di documenti se necessario
    if len(all_docs) > 20:  # Aumentato a 20 per catturare più contenuti
        # Ordina i documenti per mantenere la struttura del documento originale
        try:
            all_docs.sort(key=lambda x: x.metadata.get('page', 0)
                          * 1000 + x.metadata.get('order', 0))
            all_docs = all_docs[:20]
        except:
            # Se non funziona, seleziona semplicemente un sottoinsieme
            import random
            random.shuffle(all_docs)
            all_docs = all_docs[:20]

    # Unisci il contenuto dei documenti
    context = "\n\n".join([doc.page_content for doc in all_docs])

    # Generate summary
    chain = LLMChain(llm=llm, prompt=summary_prompt)
    try:
        result = chain.run(context=context)
        return result
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "I couldn't generate a complete summary. The document might be too large or complex. Try breaking it into smaller sections or adjusting the chunk size."


def calculate_tokens(texts):
    """Calculate number of tokens in texts using tiktoken."""
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens


def clear_document_history():
    """Clear document chat history from streamlit session state."""
    if 'document_history' in st.session_state:
        del st.session_state['document_history']


# -------------------------------------------------------------------------------
# MEAL PLANNER FUNCTIONS
# -------------------------------------------------------------------------------


class RecipeJSONLoader(BaseLoader):
    """Custom loader for recipe JSON files."""

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        """Load recipes from JSON file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        for recipe in data["recipes"]:
            # Format recipe text - convert to lowercase for case-insensitive search
            recipe_text = f"Title: {recipe['title']}\n"
            recipe_text += f"Cuisine: {recipe['cuisine']}\n"
            recipe_text += f"Meal Type: {recipe['mealType']}\n"
            recipe_text += f"Preparation Time: {recipe['prepTime']}\n"
            recipe_text += f"Cooking Time: {recipe['cookTime']}\n"
            recipe_text += f"Servings: {recipe['servings']}\n"
            recipe_text += f"Dietary Info: {', '.join(recipe['dietaryInfo'])}\n\n"

            recipe_text += f"Ingredients:\n"
            for ingredient in recipe['ingredients']:
                recipe_text += f"- {ingredient}\n"

            recipe_text += f"\nInstructions:\n{recipe['instructions']}"

            # Recipe-specific metadata
            metadata = {
                "id": recipe["id"],
                "title": recipe["title"],
                "cuisine": recipe["cuisine"],
                "mealType": recipe["mealType"],
                "source": self.file_path
            }

            # Create document
            doc = Document(page_content=recipe_text, metadata=metadata)
            documents.append(doc)

        return documents


def load_recipes(file_path):
    """Function to load recipes using the custom loader."""
    try:
        loader = RecipeJSONLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading recipes: {e}")
        return []


def chunk_recipe_data(documents, chunk_size=1000, chunk_overlap=200):
    """Split recipe documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_recipe_vector_store(chunks, persist_directory=None, api_key=None):
    """Create vector store from recipe chunks using ChromaDB connection."""
    if not api_key:
        return None

    # Ensure the directory exists
    chroma_path = "/tmp/.chroma_recipes"
    os.makedirs(chroma_path, exist_ok=True)

    # Configuration for ChromaDB connection
    configuration = {
        "client": "PersistentClient",
        "path": chroma_path
    }

    # Create Streamlit connection
    conn = st.connection("chromadb_recipes",
                         type=ChromadbConnection, **configuration)

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Collection name
    collection_name = "recipe_collection"

    # Create or get the collection
    try:
        # Use chromadb client directly
        chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Create a new collection
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            client=chroma_client
        )

        return vector_store
    except Exception as e:
        st.error(f"Error creating recipe embeddings: {e}")
        return None


def search_recipes(vector_store, query, top_k=5):
    """Search for similar recipes in the vector store."""
    # Normalize query for case-insensitive search
    normalized_query = normalize_query(query)

    # Semantic search
    results = vector_store.similarity_search_with_score(
        normalized_query, k=top_k)

    for i, (doc, score) in enumerate(results):
        st.write(f"**Result {i+1} (Similarity: {1/(1+score):.4f}):**")
        st.write(f"**Title:** {doc.metadata.get('title')}")
        st.write(f"**Cuisine:** {doc.metadata.get('cuisine')}")
        st.write(f"**Meal Type:** {doc.metadata.get('mealType')}")
        st.write(f"**Content preview:** {doc.page_content[:300]}...")
        st.write("---")


def create_conversational_chain(vector_store, api_key=None):
    """Create a conversational chain with memory."""
    # Verifica se la chiave API è stata fornita
    if not api_key:
        st.error(
            "Please provide an OpenAI API key in the sidebar to use this feature.")
        return None

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create a conversational retrieval chain with a system message for case-insensitivity
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )

    return conversation_chain


def create_meal_planner_chain(vector_store, api_key=None):
    """Create a specialized chain for meal planning."""
    # Verifica se la chiave API è stata fornita
    if not api_key:
        st.error(
            "Please provide an OpenAI API key in the sidebar to use this feature.")
        return None

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Define specialized prompt for meal planning
    meal_planner_template = """
    You are a helpful meal planning assistant. Based on the recipe information provided and the user's preferences,
    create a meal plan as requested. Use only the recipes mentioned in the context or reasonable variations of them.
    Be case-insensitive in your search, meaning treat 'chicken' and 'Chicken' as the same ingredient.

    Provide a detailed meal plan with:
    - Recipe suggestions for each meal
    - Preparation tips from the instructions
    - Modifications needed to meet the user's requirements

    Format your response in a clear, organized way with headings and bullet points as appropriate.

    Context (Recipe Information):
    {context}

    User Request: {question}
    """

    meal_planner_prompt = ChatPromptTemplate.from_template(
        meal_planner_template)

    # Define function to format documents
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Build the meal planner RAG chain
    meal_planner_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | meal_planner_prompt
        | llm
        | StrOutputParser()
    )

    return meal_planner_chain


def create_shopping_list_chain(api_key=None):
    """Create a chain for generating shopping lists from meal plans."""
    # Verifica se la chiave API è stata fornita
    if not api_key:
        st.error(
            "Please provide an OpenAI API key in the sidebar to use this feature.")
        return None

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

    # Define template for shopping list generation
    shopping_list_template = """
    You are an assistant specialized in creating shopping lists. Based on the provided meal plan,
    create a complete and organized shopping list.

    Carefully analyze the following meal plan and identify all necessary ingredients:

    {meal_plan}

    Generate a shopping list organized by categories (e.g., Proteins, Vegetables, Fruits, Dairy, Condiments, etc.).
    Group similar ingredients and indicate the approximate quantity needed to cover the period of the meal plan.
    Consider that some recipes may use the same ingredients.

    Format your response clearly and neatly, using bullet points to facilitate use during shopping.
    """

    # Create prompt from template
    shopping_list_prompt = ChatPromptTemplate.from_template(
        shopping_list_template)

    # Build the chain for shopping list
    shopping_list_chain = (
        {"meal_plan": RunnablePassthrough()}
        | shopping_list_prompt
        | llm
        | StrOutputParser()
    )

    return shopping_list_chain


# -------------------------------------------------------------------------------
# MAIN APP FUNCTIONS
# -------------------------------------------------------------------------------


def document_qa_app():
    """Document Q&A Application"""
    st.title("📄 Document Q&A Application")

    # Initialize session state for document Q&A (only if not already initialized)
    if 'document_vs' not in st.session_state:
        st.session_state.document_vs = None
    if 'document_history' not in st.session_state:
        st.session_state.document_history = ''

    # File uploader and settings in sidebar
    with st.sidebar:
        st.header("Document Settings")

        # API key input
        api_key = st.text_input(
            'OpenAI API Key (required):', type='password', key="doc_api_key",
            help="Enter your OpenAI API key to use the document Q&A features")

        # Check if API key is valid format (simple check)
        if api_key and not api_key.startswith("sk-"):
            st.warning(
                "The API key format doesn't look correct. OpenAI API keys typically start with 'sk-'")

        # Document file uploader
        uploaded_file = st.file_uploader('Upload a document:', type=[
                                         'pdf', 'docx', 'txt'], key="document_uploader")

        # Chunk size configuration
        chunk_size = st.number_input(
            'Chunk size:', min_value=100, max_value=2048, value=512,
            on_change=clear_document_history, key="doc_chunk_size")

        # K value for retrieval
        k = st.number_input('k (number of chunks to retrieve):',
                            min_value=1, max_value=20, value=3,
                            on_change=clear_document_history, key="doc_k_value")

        # Process document button - disabilitato se non c'è API key
        process_doc = st.button(
            'Process Document',
            on_click=clear_document_history,
            key="process_doc_button",
            disabled=not api_key  # Disabilita il pulsante se non c'è l'API key
        )

        # Messaggio di avviso se non c'è API key
        if not api_key:
            st.warning(
                "⚠️ Please enter your OpenAI API key to use this application.")

        # Process uploaded document
        if uploaded_file and process_doc and api_key:
            with st.spinner('Processing document...'):
                # Save file temporarily
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./temp', uploaded_file.name)

                # Create temp directory if it doesn't exist
                if not os.path.exists('./temp'):
                    os.makedirs('./temp')

                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                # Load and process document
                data = load_document(file_name)
                if data:
                    chunks = chunk_document_data(
                        data, chunk_size=chunk_size, chunk_overlap=100)
                    st.write(
                        f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                    tokens = calculate_tokens(chunks)
                    st.write(f'Total tokens: {tokens}')

                    # Create embeddings
                    vector_store = create_document_embeddings(
                        chunks, api_key=api_key)
                    st.session_state.document_vs = vector_store

                    st.success('Document processed successfully!')
                else:
                    st.error('Failed to load document. Check the file format.')

        # Aggiungi una nota informativa sulla possibilità di chiedere un riassunto
        if st.session_state.document_vs is not None:
            st.info("💡 Tip: You can ask for a document summary by typing 'summarize the document' or 'give me a summary' in the Q&A section.")

    # Main document Q&A interface
    if st.session_state.document_vs:
        # Controlla se c'è una chiave API
        if not api_key:
            st.warning(
                "⚠️ Please enter your OpenAI API key in the sidebar to ask questions about the document.")
        else:
            # Document question input
            doc_question = st.text_input(
                'Ask a question about the document:',
                help="You can also ask for a summary by typing 'summarize the document'",
                key="doc_question_input")

            if doc_question:
                # Normalize the query for case insensitivity
                normalized_question = normalize_query(doc_question)

                # Add context directive to question (escludiamo le richieste di riassunto)
                is_summary_request = any(keyword in normalized_question for keyword in
                                         ["riassumi", "riassunto", "riepilogo", "sintetizza", "sintesi",
                                          "summary", "summarize", "summarise", "overview", "recap"])

                if not is_summary_request:
                    standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                        "If you can't answer then return `I DONT KNOW`."
                    full_question = f"{normalized_question} {standard_answer}"
                else:
                    full_question = normalized_question

                # Get answer
                vector_store = st.session_state.document_vs
                with st.spinner('Processing your question...' if not is_summary_request else 'Generating document summary...'):
                    answer = ask_document(
                        vector_store, full_question, k, api_key=api_key)

                # Display answer
                st.markdown("### Answer:")
                st.markdown(answer)

                # Se è un riassunto, offri la possibilità di scaricarlo
                if is_summary_request:
                    if st.download_button(
                        label="Download Summary",
                        data=answer,
                        file_name="document_summary.md",
                        mime="text/markdown",
                        key="download_answer_summary"
                    ):
                        st.success("Summary downloaded!")

                st.divider()

                # Update chat history
                value = f'Q: {doc_question} \nA: {answer}'
                st.session_state.document_history = f'{value} \n {"-" * 100} \n {st.session_state.document_history}'

                # Display chat history
                st.text_area(label='Chat History', value=st.session_state.document_history,
                             key='document_history_display', height=400)
    else:
        st.info("Please upload and process a document to start asking questions.")


def meal_planner_app():
    """Meal Planner Application"""
    st.title("🍲 Meal Planner with RAG")

    # Initialize session state for meal planner (only if not already initialized)
    if "recipe_vs" not in st.session_state:
        st.session_state.recipe_vs = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "meal_plan" not in st.session_state:
        st.session_state.meal_plan = None
    if "recipe_titles" not in st.session_state:
        st.session_state.recipe_titles = []

    # Recipe settings in sidebar
    with st.sidebar:
        st.header("Recipe Settings")

        # API key input
        api_key = st.text_input(
            'OpenAI API Key (required):', type='password', key="meal_api_key",
            help="Enter your OpenAI API key to use the meal planner features")

        # Check if API key is valid format
        if api_key and not api_key.startswith("sk-"):
            st.warning(
                "The API key format doesn't look correct. OpenAI API keys typically start with 'sk-'")

        # Messaggio di avviso se non c'è API key
        if not api_key:
            st.warning(
                "⚠️ Please enter your OpenAI API key to use this application.")

        # JSON Template guidance
        st.markdown("### Recipe JSON Format")
        with st.expander("How to Format Recipe JSON", expanded=False):
            st.markdown("""
            Your recipe JSON file should follow this structure:
            ```json
            {
              "recipes": [
                {
                  "id": 1,
                  "title": "Recipe Name",
                  "cuisine": "Italian/Mexican/etc",
                  "mealType": "Breakfast/Lunch/Dinner/Snack",
                  "prepTime": "30 minutes",
                  "cookTime": "45 minutes",
                  "servings": 4,
                  "dietaryInfo": ["Vegetarian", "Gluten-free", "etc"],
                  "ingredients": [
                    "200g ingredient one",
                    "2 tbsp ingredient two",
                    "etc"
                  ],
                  "instructions": "Step by step cooking instructions..."
                }
              ]
            }
            ```
            """)

        # Template download
        template_json = """
{
  "recipes": [
    {
      "id": 1,
      "title": "Recipe Name",
      "cuisine": "Cuisine Type",
      "mealType": "Breakfast/Lunch/Dinner/Snack",
      "prepTime": "30 minutes",
      "cookTime": "45 minutes",
      "servings": 4,
      "dietaryInfo": ["Vegetarian", "Gluten-free"],
      "ingredients": [
        "200g ingredient one",
        "2 tbsp ingredient two",
        "3 pieces ingredient three"
      ],
      "instructions": "Step by step cooking instructions. Be as detailed as possible."
    }
  ]
}
        """
        st.download_button(
            label="Download JSON Template",
            data=template_json,
            file_name="recipe_template.json",
            mime="application/json",
            key="recipe_template_download"
        )

        # Recipe file uploader
        st.markdown("### Upload Your Recipes")
        recipe_file = st.file_uploader("Upload Recipe JSON:", type=[
                                       "json"], key="recipe_uploader")

        # Process recipes button - disabilitato se non c'è API key
        process_recipes = st.button(
            "Process Recipes",
            key="process_recipes_button",
            disabled=not api_key  # Disabilita il pulsante se non c'è l'API key
        )

        # Process uploaded recipe file
        if recipe_file and process_recipes and api_key:
            with st.spinner("Processing recipes..."):
                try:
                    # Create data directory if needed
                    if not os.path.exists("./data"):
                        os.makedirs("./data")

                    # Save file to disk
                    file_path = os.path.join("./data", recipe_file.name)
                    with open(file_path, "wb") as f:
                        f.write(recipe_file.getbuffer())

                    # Validate JSON format
                    with open(file_path, 'r') as f:
                        try:
                            recipe_data = json.load(f)
                            if "recipes" not in recipe_data or not isinstance(recipe_data["recipes"], list):
                                st.error(
                                    "Invalid JSON format. The file must contain a 'recipes' array.")
                                st.stop()
                        except json.JSONDecodeError:
                            st.error(
                                "Invalid JSON file. Please check the format.")
                            st.stop()

                    # Load and process recipes
                    documents = load_recipes(file_path)
                    if documents:
                        st.success(f"✅ Loaded {len(documents)} recipes.")

                        # Create chunks
                        chunks = chunk_recipe_data(documents)
                        st.info(
                            f"📄 Created {len(chunks)} chunks from {len(documents)} recipes.")

                        # Extract recipe titles
                        recipe_titles = [doc.metadata.get(
                            'title') for doc in documents]
                        st.session_state.recipe_titles = recipe_titles

                        # Create vector store
                        vector_store = create_recipe_vector_store(
                            chunks,
                            persist_directory="./data/recipe_chroma_db",
                            api_key=api_key)
                        st.session_state.recipe_vs = vector_store
                        st.success(
                            "🎉 Recipes processed and embeddings created successfully!")
                    else:
                        st.warning("No recipes found in the uploaded file.")
                except Exception as e:
                    st.error(f"Error processing recipes: {str(e)}")
                    st.stop()

    # Mostra i titoli delle ricette nella pagina principale
    if st.session_state.recipe_titles:
        st.header("Available Recipes")

        # Mostra le ricette in colonne per un'interfaccia più ordinata
        recipe_cols = st.columns(3)
        for i, title in enumerate(st.session_state.recipe_titles):
            col_idx = i % 3
            with recipe_cols[col_idx]:
                st.markdown(f"- **{title}**")

        st.divider()  # Aggiungi un separatore tra le ricette e le tabs

    # Create tabs for different meal planner functionalities (senza "Recipe Search")
    tabs = st.tabs([
        "Recipe Q&A",
        "Meal Planning",
        "Shopping List"
    ])

    # Tab 1: Recipe Q&A
    with tabs[0]:
        st.header("Recipe Q&A")
        if st.session_state.recipe_vs:
            # Verifica se c'è l'API key
            if not api_key:
                st.warning(
                    "⚠️ Please enter your OpenAI API key in the sidebar to use this feature.")
            else:
                # Create conversation chain if not already in session state
                if "conversation_chain" not in st.session_state or st.session_state.last_api_key != api_key:
                    st.session_state.conversation_chain = create_conversational_chain(
                        st.session_state.recipe_vs, api_key=api_key)
                    st.session_state.last_api_key = api_key

                # Display conversation history
                for message in st.session_state.conversation_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

                # User input
                user_question = st.chat_input(
                    "Ask a question about recipes:", key="recipe_qa_input")

                if user_question:
                    # Normalize the query for case-insensitive search
                    normalized_question = normalize_query(user_question)

                    # Add user message to chat history
                    st.session_state.conversation_history.append(
                        {"role": "user", "content": user_question})
                    with st.chat_message("user"):
                        st.write(user_question)

                    # Get AI response using normalized question
                    with st.spinner("Thinking..."):
                        if st.session_state.conversation_chain:
                            response = st.session_state.conversation_chain.invoke(
                                {"question": normalized_question})
                            assistant_response = response["answer"]
                        else:
                            assistant_response = "There was an error connecting to the OpenAI API. Please check your API key."

                    # Add assistant response to chat history
                    st.session_state.conversation_history.append(
                        {"role": "assistant", "content": assistant_response})
                    with st.chat_message("assistant"):
                        st.write(assistant_response)

                    # Force UI to refresh
                    st.rerun()
        else:
            st.warning("Please upload and process recipe data first.")

    # Tab 2: Meal Planning
    with tabs[1]:
        st.header("Meal Planning")
        if st.session_state.recipe_vs:
            # Verifica se c'è l'API key
            if not api_key:
                st.warning(
                    "⚠️ Please enter your OpenAI API key in the sidebar to use this feature.")
            else:
                # Create meal planner chain if not already in session state
                if "meal_planner_chain" not in st.session_state or st.session_state.last_meal_planner_api_key != api_key:
                    st.session_state.meal_planner_chain = create_meal_planner_chain(
                        st.session_state.recipe_vs, api_key=api_key)
                    st.session_state.last_meal_planner_api_key = api_key

                # Meal planning form
                with st.form("meal_plan_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        duration = st.selectbox(
                            "Duration:", ["1 day", "3 days", "1 week"], index=1, key="plan_duration")
                        meals_per_day = st.multiselect(
                            "Meals per day:",
                            ["Breakfast", "Lunch", "Dinner", "Snacks"],
                            default=["Breakfast", "Lunch", "Dinner"],
                            key="plan_meals_per_day"
                        )

                    with col2:
                        dietary_pref = st.multiselect(
                            "Dietary preferences:",
                            ["Vegetarian", "Vegan", "Low-carb",
                                "High-protein", "Gluten-free", "Dairy-free"],
                            key="plan_dietary_pref"
                        )
                        allergies = st.multiselect(
                            "Allergies/Restrictions:",
                            ["Nuts", "Seafood", "Eggs", "Soy", "Wheat", "None"],
                            default=["None"],
                            key="plan_allergies"
                        )

                    additional_reqs = st.text_area(
                        "Additional requirements or ingredients:", key="plan_additional_reqs")
                    submit_button = st.form_submit_button("Generate Meal Plan")

                if submit_button:
                    with st.spinner("Generating meal plan..."):
                        # Construct meal plan request
                        dietary_info = ", ".join(
                            dietary_pref) if dietary_pref else "None"
                        allergy_info = ", ".join(
                            [a for a in allergies if a != "None"]) if "None" not in allergies else "None"

                        meal_plan_request = f"""
                        Please create a {duration} meal plan with {', '.join([meal.lower() for meal in meals_per_day])} each day.
                        Dietary preferences: {dietary_info}
                        Allergies/Restrictions: {allergy_info}
                        Additional requirements: {additional_reqs}
                        """

                        # Generate meal plan
                        meal_plan = st.session_state.meal_planner_chain.invoke(
                            meal_plan_request)
                        st.session_state.meal_plan = meal_plan

                    # Force UI to refresh
                    st.rerun()

                # Display meal plan if available
                if st.session_state.meal_plan:
                    st.subheader("Your Personalized Meal Plan")
                    st.markdown(st.session_state.meal_plan)

                    # Option to save meal plan
                    if st.download_button(
                        label="Download Meal Plan",
                        data=st.session_state.meal_plan,
                        file_name="meal_plan.md",
                        mime="text/markdown",
                        key="download_meal_plan"
                    ):
                        st.success("Meal plan downloaded!")
        else:
            st.warning("Please upload and process recipe data first.")

    # Tab 3: Shopping List
    with tabs[2]:
        st.header("Shopping List Generator")
        if st.session_state.meal_plan:
            # Verifica se c'è l'API key
            if not api_key:
                st.warning(
                    "⚠️ Please enter your OpenAI API key in the sidebar to use this feature.")
            else:
                # Create shopping list chain if not already in session state
                if "shopping_list_chain" not in st.session_state or st.session_state.last_shopping_list_api_key != api_key:
                    st.session_state.shopping_list_chain = create_shopping_list_chain(
                        api_key=api_key)
                    st.session_state.last_shopping_list_api_key = api_key

                if st.button("Generate Shopping List", key="generate_shopping_list"):
                    with st.spinner("Generating shopping list..."):
                        # Generate shopping list from meal plan
                        shopping_list = st.session_state.shopping_list_chain.invoke(
                            st.session_state.meal_plan)
                        st.session_state.shopping_list = shopping_list

                    # Force UI to refresh
                    st.rerun()

                # Display shopping list if available
                if "shopping_list" in st.session_state:
                    st.subheader("Your Shopping List")
                    st.markdown(st.session_state.shopping_list)

                    # Option to save shopping list
                    if st.download_button(
                        label="Download Shopping List",
                        data=st.session_state.shopping_list,
                        file_name="shopping_list.md",
                        mime="text/markdown",
                        key="download_shopping_list"
                    ):
                        st.success("Shopping list downloaded!")
        else:
            st.warning("Please generate a meal plan first.")


def main():
    """Main function to run the combined application."""
    # Add app selection to the sidebar
    with st.sidebar:
        st.title("AI Assistant with RAG")

        # Messaggio principale sull'API key
        st.markdown("""
        ### 🔑 API Key Required
        This application requires an OpenAI API key to function.
        You'll need to provide your API key in each section you want to use.
        """)

        st.markdown("---")

        # App selection
        st.header("Choose Application")
        app_selection = st.radio(
            "Select which application to use:",
            ["📄 Document Q&A", "🍲 Meal Planner"],
            key="app_selector"
        )

        # App descriptions
        if app_selection == "📄 Document Q&A":
            st.info("""
            **Document Q&A** lets you upload documents (PDF, DOCX, TXT) and ask questions about their content.
            
            The app uses RAG (Retrieval-Augmented Generation) to find the most relevant parts of your document and generate accurate answers.
            
            You can also generate a comprehensive summary of your document.
            """)
        else:
            st.info("""
            **Meal Planner** helps you plan meals based on available recipes.
            
            You can upload recipes(follow the JSON), ask questions about cooking, generate personalized meal plans, and create shopping lists.
            
            
            """)

        # Update current app in session state when changed
        if app_selection != st.session_state.current_app:
            st.session_state.current_app = app_selection

        # Create directories if they don't exist
        for directory in ["./data", "./temp"]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    # Run the selected application
    if st.session_state.current_app == "📄 Document Q&A":
        document_qa_app()
    else:
        meal_planner_app()

    # About Me section - now at the bottom of the page
    with st.sidebar:
        st.markdown("---")
        st.title("About me!")
        # Profile image
        st.image(
            "https://avatars.githubusercontent.com/u/72889405?v=4",
            width=120,
            caption="Veronica Schembri",
            output_format="auto",
        )

        # Name and description
        st.write("## Veronica Schembri")
        st.write("Junior AI Engineer | Front-end Developer")

        # Social Media section
        st.write("### Social Media")
        st.markdown(
            """
            - [🌐 Sito](https://www.veronicaschembri.com)
            - [🐙 GitHub](https://github.com/Pandagan-85)
            - [🔗 LinkedIn](https://www.linkedin.com/in/veronicaschembri/)
            - [📸 Instagram](https://www.instagram.com/schembriveronica/)
            """
        )


if __name__ == "__main__":
    main()