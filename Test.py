import langchain
from langchain_huggingface import ChatHuggingFace
#from langchain_huggingface import Hu
#from langchain.llms import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained LLaMA model and tokenizer
model_name = "decapoda-research/llama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a LangChain LLM wrapper around the LLaMA model
llm = ChatHuggingFace(model=model, tokenizer=tokenizer)

# Define a LangChain prompt template
template = langchain.prompts.PromptTemplate(
    input_variables=["input"],
    template="You are a helpful assistant. {input}",
)

# Create a LangChain chain with the LLaMA model and prompt template
chain = langchain.chains.LLMChain(llm=llm, prompt=template)

# Run the chain with an example input
input_text = "Tell me a joke."
output = chain.run(input=input_text)

print(output)