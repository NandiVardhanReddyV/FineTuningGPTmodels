from transformers import pipeline

# Initialize the Hugging Face pipeline with your fine-tuned model
generator = pipeline("text-generation", model="models/results/checkpoint-89")

# Use the model to generate text
generated_text = generator("Write a poem about AI.", max_length=100)
print("Generated Text:", generated_text)
