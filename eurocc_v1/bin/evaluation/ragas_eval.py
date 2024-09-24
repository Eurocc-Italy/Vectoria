from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision, 
)

def load_model_open_ai_wrapped():
    # this works
    import os
    from langchain_openai import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings

    os.environ["OPENAI_API_KEY"] = "sk-proj-GWmRyOB4lCfpnotiF5wsT3BlbkFJf9FvlkHdBA4CU6qv2KSC"

    embeddings = OpenAIEmbeddings(model="gpt-4o-mini")

    llm = ChatOpenAI(
        model="gpt-4o-mini"
    )
    return llm, embeddings

def load_model(model_name, quantize=False):
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)


    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        #return_full_text=True,
        #max_new_tokens=100,
        batch_size=1
        #temperature=0.1, 
        #repetition_penalty=1.1
    )



    # pipe = pipeline(
    #     "text-generation", 
    #     model=model_name,
    #     trust_remote_code=True,
    #     device="cuda"
    # )

    return pipe

def wrap_to_ragas(pipe, embedder_model_name):
    # wrap the LLM and embeddings with the Langchain wrappers
    llm = HuggingFacePipeline(pipeline=pipe)
    embedder = HuggingFaceEmbeddings(model_name=embedder_model_name)

    # Wrap the LLM and embeddings with the RAGAS wrappers
    #llm = LangchainLLMWrapper(llm)
    #embedder = LangchainEmbeddingsWrapper(embedding_model)

    return llm, embedder

def load_dataset():
    # Example data samples
    data_samples = {
        'question': [
            'When was the first super bowl?', 
            'Who won the most super bowls?'
        ],
        'answer': [
            'The first superbowl was held on Jan 15, 1967', 
            'The most super bowls have been won by The New England Patriots'],
        'contexts': [
            [
                'The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'
            ],
            [
                'The Green Bay Packers...Green Bay, Wisconsin.', 
                'The Packers compete...Football Conference'
            ]
        ],
        'ground_truth': [
            'The first superbowl was held on January 15, 1967', 
            'The New England Patriots have won the Super Bowl a record six times'
        ]
    }

    dataset = Dataset.from_dict(data_samples)

    return dataset

def run_evaluation(dataset, langchain_llm, langchain_embeddings):

    result = evaluate(
        dataset=dataset,
        llm=langchain_llm,
        #embeddings=langchain_embeddings,
        metrics=[
            #context_precision,
            faithfulness,
            #answer_relevancy,
            #context_recall,
        ],
        raise_exceptions=True,
        is_async=True
    )

    evaluation_df = result.to_pandas()
    print("Evaluation Results:\n", evaluation_df)



def main():
    #model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    #model_name = "sapienzanlp/Minerva-3B-base-v1.0" # needs access
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    #model_name = "meta-llama/Llama-2-7b-hf"
    embedder_model_name = "BAAI/bge-m3"
    #embedder_model_name = "bert-base-uncased"

    # this works
    #langchain_llm, langchain_embeddings = load_model_open_ai_wrapped()

    pipe = load_model(model_name, quantize=True)
    langchain_llm, langchain_embeddings = wrap_to_ragas(pipe, embedder_model_name)
    dataset = load_dataset()

    """
    Utilizzando i modelli locali (Phi-3, llama-2 etc..) attraverso il wrapper di LangChain (che saranno poi wrappati da Ragas internamente)
    si ottiene l'errore:
    ../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
    Che descrive un suggerisce un problema nel dimensionamento degli embeddings durante l'allocazione 
    di questi nella GPU. 

    L'esperimento descritto sopra è stato eseguito quantizzando il modello (Leo's GPU size = 16Gb).
    * Provare ad eseguire l'esperimento senza quantizzazione sul DaVinci o su Leonardo (mi aspetto che l'errore persista)

    In ogni caso, togliendo la quantizzazione e lanciando il modello su CPU, il problema persiste (l'errore però lanciato
    dal runner thread è in questo caso sconosciuto).

    Utilizzando un wrapper langchain su OpenAI (chat-gpt-4o-mini) funziona correttamente.
    """
    
    run_evaluation(dataset, langchain_llm, langchain_embeddings)

if __name__ == "__main__":
    main()