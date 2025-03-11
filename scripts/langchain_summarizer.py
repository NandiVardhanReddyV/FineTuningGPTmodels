from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

# Load your fine-tuned model from the checkpoint directory
model_path = "models/results/checkpoint-89"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a Hugging Face pipeline for text generation
text_generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512, temperature=0.5)

# Use the pipeline with LangChain's HuggingFacePipeline wrapper
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define a prompt template for summarizing documents
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following document:\n\n{text}"
)

# Create the LangChain chain
chain = LLMChain(llm=llm, prompt=prompt)

# Test the chain with some document text
document = """Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. 
These processes include learning, reasoning, and self-correction."""
summary = chain.run(document)

print("Summary:", summary)
