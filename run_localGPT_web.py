from langchain.chains import RetrievalQA
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline, AutoTokenizer
import torch
import click
import gradio as gr

from constants import CHROMA_SETTINGS


def load_model():
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    model_id = "TheBloke/vicuna-7B-1.1-HF"
    model_id = "THUDM/chatglm-6b-int4"
    tokenizer = LlamaTokenizer.from_pretrained(model_id,trust_remote_code=True)

    model = LlamaForCausalLM.from_pretrained(model_id,
                                                load_in_8bit=True, # set these options if your GPU supports them!
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                          trust_remote_code=True
                                             )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


# @click.command()
# @click.option('--device_type', default='gpu', help='device to run on, select gpu or cpu')
# def main(device_type, ):
#     # load the instructorEmbeddings
#     if device_type in ['cpu', 'CPU']:
#         device='cpu'
#     else:
#         device='cuda'


 ## for M1/M2 users:

@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
def main(device_type, ):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    elif device_type in ['mps', 'MPS']:
        device='mps'
    else:
        device='cuda'

    print(f"Running on: {device}")

    # hkunlp/instructor-xl
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": device})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses. 
    llm = load_model()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">localGPT</h1>""")
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def respond(message, chat_history):
            # Get the answer from the chain
            res = qa(message)
            answer, docs = res['result'], res['source_documents']

            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

            chat_history.append((message, answer))
            return "", chat_history


        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(share=True, inbrowser=True)


if __name__ == "__main__":
    main()
