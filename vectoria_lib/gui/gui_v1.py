# https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart
from pathlib import Path
import streamlit as st

from vectoria_lib.tasks.build_index import build_index_from_files
from vectoria_lib.llm.application_builder import ApplicationBuilder
import tempfile
import time

indexes_dir = Path(__file__).parent  / "indexes"
indexes_dir.mkdir(parents=True, exist_ok=True)

def init_globals():
    if "qa_app" not in st.session_state:
        st.session_state.qa_app = None
    if "index_path" not in st.session_state:
        st.session_state.index_path = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
            
def show_info_messages():
    # if st.session_state.vector_store_pickle_path is None:
    #     st.warning("FAISS index is not created yet.")
    if st.session_state.qa_app is None:
        st.warning("ChatBot is not loaded yet.")

def dump_files(files, output_dir):
    generated_files = []
    for file in files:
        print("File: ", file)
        with open(output_dir / file.name, "wb") as f:
            f.write(file.getvalue())
        generated_files.append(output_dir / file.name)
    return generated_files
    

def sidebar():
    st.sidebar.title("Navigation")

    with st.sidebar:

        with st.expander("Create Index"):
            with st.form(key='index_creation_form'):
                st.write("Specify the folder and embedder model to create the index.")
                input_docs = st.file_uploader("Input Documents", type=["pdf", "docx"], accept_multiple_files=True)
                output_index_dir = st.text_input("Output Index Directory",  str(indexes_dir))
                submit_button = st.form_submit_button(label='Generate Index')                
                if submit_button:
                    with st.spinner('Generating index...'):
                        with tempfile.TemporaryDirectory() as temp_dir:
                            generated_files = dump_files(input_docs, Path(temp_dir))
                            vector_store_pickle_path, _ = build_index_from_files(generated_files, output_index_dir)
                            if vector_store_pickle_path.exists():
                                st.session_state.index_path = vector_store_pickle_path
                                st.success(f"Index created successfully in {vector_store_pickle_path}.")
                            else:
                                st.error("Failed to create FAISS index.")

        with st.expander("Load ChatBot"):
            with st.form(key='chatbot_load_form'):
                if st.session_state.index_path is not None:
                    index_path = st.text_input("Index folder path", value=st.session_state.index_path)
                else:
                    index_path = st.text_input("Index folder path")
                submit_button = st.form_submit_button(label='Load ChatBot')
                if submit_button:
                    with st.spinner('Loading ChatBot...'):
                        st.session_state.qa_app = ApplicationBuilder.build_qa(
                            index_path = index_path
                        )
                        if st.session_state.qa_app:
                            st.success("ChatBot loaded successfully.")
                        else:
                            st.error("Failed to load ChatBot.")

def ask_question():
    with st.form(key='chatbot_qa'):
        query = st.text_input("Fai una domanda!", key='query')
        print(query)
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            with st.spinner('Generating answer...'):
                s = time.time()
                answer = st.session_state.qa_app.ask(query)
                e = time.time()
                st.write(f"{answer['answer']}")
                st.write(f"Time taken: {e-s:.2f} seconds")

def main(): 
    # st.info(f"manage_session_state {st.session_state}")
    logo_url = "https://euroccitaly.it/wp-content/uploads/2023/12/logo-eurocc-italy-white.svg"
    st.image(logo_url, width=250)

    st.title("Vectoria")
    
    init_globals()
    show_info_messages()    
    sidebar()
    
    if st.session_state.qa_app is not None:
        ask_question()

if __name__ == "__main__":
    main()
