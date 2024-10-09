# https://docs.streamlit.io/develop/tutorials/llms/llm-quickstart
import threading
from pathlib import Path
import streamlit as st

from eurocc_v1.lib.api.v1 import (
    create_and_write_index,
    create_qa_agent
)
from eurocc_v1.paths import DATA_DIR

import GPUtil
import plotly.graph_objs as go
import time

def create_index(input_docs_dir, output_index_dir, embedder_model):
    return create_and_write_index(input_docs_dir, output_index_dir, embedder_model)

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    usage = [gpu.load * 100 for gpu in gpus]
    st.info(f"GPU Usage: {usage}")
    return usage[0] if usage else 0

def collect_gpu_data():
    current_time = time.time()
    current_time_label = time.strftime("%H:%M:%S", time.localtime(current_time))
    current_gpu_usage = get_gpu_usage()
    
    if 'gpu_usage' not in st.session_state:
        st.session_state.gpu_usage = []
    if 'time_labels' not in st.session_state:
        st.session_state.time_labels = []
    if 'time' not in st.session_state:
        st.session_state.time = []
    
    st.session_state.gpu_usage.append(current_gpu_usage)
    st.session_state.time.append(current_time)
    st.session_state.time_labels.append(current_time_label)
    
    if len(st.session_state.gpu_usage) > 10:
        st.session_state.gpu_usage = st.session_state.gpu_usage[-10:]
        st.session_state.time = st.session_state.time[-10:]
        st.session_state.time_labels = st.session_state.time_labels[-10:]


def main(): 
    # st.info(f"manage_session_state {st.session_state}")

    logo_url = "https://upload.wikimedia.org/wikipedia/commons/d/da/Logo_Leonardo.png"
    st.image(logo_url, caption='Leonardo SpA', use_column_width=True, width=50)

    st.title("Electronics QA Chatbot")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Chatbot", "GPU Usage"])
    
    if page == "Home":
        st.header("Loading ChatBot")
        st.write("Choose an option below:")

        # Info messages
        if 'faiss_index' in st.session_state:
            st.success("FAISS index creation status: Success")
        else:
            st.warning("FAISS index is not created yet.")

        if 'qa_agent' in st.session_state:
            st.success("ChatBot load status: Success")
        else:
            st.warning("ChatBot is not loaded yet.")

        # Manage widgets
        if st.button("Create FAISS Index"):
            st.session_state.show_index_form = True

        if st.button("Load ChatBot"):
            st.session_state.show_load_chatbot_form = True



        # Index creation form
        if 'show_index_form' in st.session_state and st.session_state.show_index_form:
            
            with st.form(key='index_creation_form'):
                st.write("Specify the folder and embedder model to create FAISS index.")
                input_docs_dir = st.text_input("Input Documents Directory", str(DATA_DIR / "raw"))
                output_index_dir = st.text_input("Output Index Directory",  str(Path(__file__).parent))
                embedder_model = st.selectbox("Embedder Model", ["BAAI/bge-m3"])
                submit_button = st.form_submit_button(label='Generate Index')
                
                if submit_button:
                    with st.spinner('Generating index...'):
                        faiss_index_path, faiss_index_obj = create_index(input_docs_dir, output_index_dir, embedder_model)
                        if faiss_index_path.exists():
                            st.session_state.faiss_index = faiss_index_obj
                            st.success(f"FAISS index created successfully in {faiss_index_path}.")
                            st.session_state.show_index_form = False  # Hide the form after successful creation
                        else:
                            st.error("Failed to create FAISS index.")
                

        # ChatBot loading form
        if 'show_load_chatbot_form' in st.session_state and st.session_state.show_load_chatbot_form:
            
            with st.form(key='chatbot_load_form'):
                st.write("Load a FAISS index from a pickle file.")
                index_file = st.file_uploader("Upload Index Pickle File", type=["pkl"], accept_multiple_files=False)
                k = st.number_input("Top-k RAG docs number", min_value=1, value=5)
                submit_button = st.form_submit_button(label='Load ChatBot')

                if submit_button:
                    if index_file is None:
                        st.error("Please upload a pickle file.")
                    else:
                        with st.spinner('Loading ChatBot...'):
                            #st.write(f"Index file: {type(index_file.getvalue())}")
                            model_name = Path(index_file.name).stem.split("_faiss_index")[0].replace('__', '/')
                            st.session_state.qa_agent = create_qa_agent(model_name, index_file.getvalue(), k)
                            if st.session_state.qa_agent:
                                st.session_state.show_load_chatbot_form = False  # Hide the form after successful creation
                                st.success("ChatBot loaded successfully.")
                            else:
                                st.error("Failed to load ChatBot.")

    if page == "Chatbot":
        if "qa_agent" not in st.session_state:
            st.warning("Please load a FAISS index first from the Home page.")
            st.stop()
        
        with st.form(key='chatbot_qa'):

            query = st.text_input("Fai una domanda!", key='query')
            print(query)
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                with st.spinner('Generating answer...'):
                    s = time.time()
                    docs = st.session_state.qa_agent.ask(query)
                    e = time.time()
                    st.write(f"Answer: {docs}")
                    st.write(f"Time taken: {e-s:.2f} seconds")






    elif page == "GPU Usage":
        st.header("GPU Usage Monitoring")

        collect_gpu_data()

        gpu_usage = st.empty()
        plotly_chart = st.empty()
    
        gpu_usage.metric("Current GPU Usage", f"{st.session_state.gpu_usage[-1]:.2f}%")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=st.session_state.time_labels,
                y=st.session_state.gpu_usage,
                mode='lines',
                name='GPU Usage'
            )
        )
        fig.update_layout(
            title='GPU Usage Over Time',
            xaxis_title='Time',
            yaxis_title='GPU Usage (%)',
            xaxis=dict(
                tickmode='array',
                tickvals=st.session_state.time_labels,
                ticktext=st.session_state.time_labels
            )
        )
        plotly_chart.plotly_chart(fig)


if __name__ == "__main__":
    main()
