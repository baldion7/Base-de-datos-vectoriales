import streamlit as st
import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================== Configuración inicial ====================
def initialize_chromadb(collection_name="enterprise_data"):
    """Inicializa la conexión con ChromaDB"""
    client = chromadb.Client()
    return client.get_or_create_collection(name=collection_name)


def initialize_transformer_model():
    """Carga el modelo de transformers para embeddings"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return (
        AutoTokenizer.from_pretrained(model_name),
        AutoModel.from_pretrained(model_name)
    )


# ==================== Funciones principales ====================
def generate_embedding(text, tokenizer, model):
    """Genera embeddings para el texto"""
    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**tokens)
        embeddings = output.last_hidden_state
        embeddings = torch.mean(embeddings, dim=1)
    return embeddings.squeeze().numpy()


def add_initial_data(collection, tokenizer, model):
    """Agrega datos iniciales de ejemplo"""
    company_data = {
        "documents": [
            "Política de vacaciones: 20 días anuales pagados",
            "Horario de oficina: L-V 8:00 AM a 6:00 PM",
            "Soporte técnico disponible 24/7"
        ],
        "metadatas": [
            {"category": "RH", "department": "Recursos Humanos"},
            {"category": "Operaciones", "department": "Administración"},
            {"category": "TI", "department": "Soporte"}
        ],
        "ids": ["doc_001", "doc_002", "doc_003"]
    }

    company_data["embeddings"] = [
        generate_embedding(doc, tokenizer, model)
        for doc in company_data["documents"]
    ]

    collection.add(**company_data)
    return "Datos iniciales agregados!"


def update_document(collection, tokenizer, model, doc_id, new_text, new_metadata):
    """Actualiza un documento existente"""
    collection.delete(ids=[doc_id])
    collection.add(
        documents=[new_text],
        metadatas=[new_metadata],
        ids=[doc_id],
        embeddings=[generate_embedding(new_text, tokenizer, model)]
    )
    return f"Documento {doc_id} actualizado"


def delete_document(collection, doc_id):
    """Elimina un documento"""
    collection.delete(ids=[doc_id])
    return f"Documento {doc_id} eliminado"


def search_documents(collection, query, tokenizer, model, n_results=5):
    """Búsqueda semántica de documentos"""
    query_embedding = generate_embedding(query, tokenizer, model)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return format_search_results(results, query)


def format_search_results(results, query):
    """Formatea los resultados de búsqueda"""
    formatted_results = []
    for doc_id, doc, meta, distance in zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
    ):
        similarity = calculate_similarity(query, doc)
        formatted_results.append({
            'id': doc_id,
            'document': doc,
            'metadata': meta,
            'similarity_score': round(similarity, 3),
            'vector_distance': round(float(distance), 3)
        })
    return formatted_results


def calculate_similarity(query, text):
    """Calcula similitud coseno"""
    vectorizer = TfidfVectorizer().fit([query, text])
    vectors = vectorizer.transform([query, text])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


# ==================== Interfaz de usuario ====================
def main():
    # Configuración inicial
    st.set_page_config(page_title="Gestor de Conocimiento", layout="wide")
    collection = initialize_chromadb()
    tokenizer, model = initialize_transformer_model()

    # Inicializar estado de conversaciones
    if 'current_conv' not in st.session_state:
        st.session_state.current_conv = None
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}

    # Sidebar para gestión de documentos
    with st.sidebar:
        st.header("📂 Gestión de Documentos")

        # Cargar datos iniciales
        if st.button("🔄 Cargar datos de ejemplo"):
            add_initial_data(collection, tokenizer, model)
            st.rerun()

        # Nuevo documento
        with st.expander("➕ Nuevo Documento"):
            new_id = st.text_input("ID del Documento")
            new_text = st.text_area("Contenido")
            new_category = st.text_input("Categoría")
            new_department = st.text_input("Departamento")
            if st.button("💾 Guardar Documento"):
                update_document(collection, tokenizer, model, new_id, new_text,
                                {"category": new_category, "department": new_department})
                st.success("¡Documento guardado!")
                st.rerun()

        # Editar/Eliminar documentos
        with st.expander("✏️ Editar Documentos"):
            docs = collection.get()['ids']
            selected_doc = st.selectbox("Seleccionar Documento", docs)

            if st.button("🗑️ Eliminar Documento"):
                delete_document(collection, selected_doc)
                st.rerun()

            if selected_doc:
                doc_info = collection.get(ids=[selected_doc])
                new_content = st.text_area("Editar Contenido", value=doc_info['documents'][0])
                new_meta = {
                    "category": st.text_input("Categoría", value=doc_info['metadatas'][0]['category']),
                    "department": st.text_input("Departamento", value=doc_info['metadatas'][0]['department'])
                }
                if st.button("💾 Guardar Cambios"):
                    update_document(collection, tokenizer, model, selected_doc, new_content, new_meta)
                    st.success("¡Cambios guardados!")
                    st.rerun()

    # Contenido principal
    st.title("🧠 Gestor de Conocimiento Empresarial")

    # Estadísticas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_docs = len(collection.get()['ids'])
        st.metric("📄 Documentos Totales", total_docs)
    with col2:
        categories = len(set([m['category'] for m in collection.get()['metadatas']])) if total_docs > 0 else 0
        st.metric("📂 Categorías", categories)
    with col3:
        departments = len(set([m['department'] for m in collection.get()['metadatas']])) if total_docs > 0 else 0
        st.metric("🏢 Departamentos", departments)
    with col4:
        last_update = collection.get()['ids'][-1] if total_docs > 0 else "N/A"
        st.metric("🕒 Última Actualización", last_update)

    # Buscador principal
    search_query = st.text_input("🔍 Buscar en todos los documentos...")

    # Tabla de documentos
    if search_query:
        results = search_documents(collection, search_query, tokenizer, model)
        display_data = [
            {
                "ID": result['id'],
                "Categoría": result['metadata']['category'],
                "Departamento": result['metadata']['department'],
                "Contenido": result['document'][:100] + "...",
                "Similitud": f"{result['similarity_score']:.0%}"
            } for result in results
        ]
    else:
        all_docs = collection.get()
        display_data = [
            {
                "ID": doc_id,
                "Categoría": meta['category'],
                "Departamento": meta['department'],
                "Contenido": doc[:100] + "...",
                "Similitud": "N/A"
            } for doc_id, doc, meta in zip(
                all_docs['ids'],
                all_docs['documents'],
                all_docs['metadatas']
            )
        ] if total_docs > 0 else []

    st.dataframe(
        display_data,
        column_config={
            "Similitud": st.column_config.ProgressColumn(
                format="%.0f%%",
                min_value=0,
                max_value=1,
            ) if search_query else None
        },
        use_container_width=True,
        hide_index=True
    )

    # Chat interactivo
    st.header("💬 Chat Inteligente")

    # Gestión de conversaciones
    col1, col2 = st.columns([4, 1])
    with col1:
        conv_name = st.text_input("Nombre de la conversación", key="new_conv_name")
    with col2:
        if st.button("Nueva Conversación"):
            if conv_name:
                st.session_state.conversations[conv_name] = []
                st.session_state.current_conv = conv_name
                st.rerun()

    selected_conv = st.selectbox(
        "Conversación Actual",
        options=list(st.session_state.conversations.keys()),
        key="conv_selector"
    )

    # Historial del chat
    if selected_conv:
        # Mostrar historial
        for msg in st.session_state.conversations[selected_conv]:
            role_icon = "🤖" if msg['role'] == 'assistant' else "👤"
            st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")

        input_key = f"input_{selected_conv}"

        # Procesar entrada antes de crear el widget
        if input_key in st.session_state and st.session_state[input_key].strip():
            user_input = st.session_state[input_key]

            # Búsqueda de documentos relevantes
            results = search_documents(collection, user_input, tokenizer, model)
            response = "**Información relevante:**\n\n"
            response += "\n\n".join([f"- {res['document'][:150]}..." for res in results])

            # Actualizar historial
            st.session_state.conversations[selected_conv].extend([
                {'role': 'user', 'content': user_input},
                {'role': 'assistant', 'content': response}
            ])

            # Limpiar input y forzar rerun
            del st.session_state[input_key]
            st.rerun()

        # Crear el widget después de procesar
        user_input = st.text_input(
            "Escribe tu pregunta...",
            key=input_key,
            value=""
        )


if __name__ == "__main__":
    main()