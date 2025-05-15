import os
from langchain_cohere.chat_models import ChatCohere
from langchain.prompts import PromptTemplate
from google.colab import drive
# Set Cohere API Key
os.environ["COHERE_API_KEY"] = "YrqBdTypjvdKMc7bBljLwihs5TS54JCN8qjrLVQ5"  # Replace with your actual key

# Mount Google Drive & Load Text
# drive.mount('/content/drive')
with open("/content/drive/MyDrive/Colab Notebooks/document.txt", "r", encoding="utf-8") as f:
    text = f.read()   

# Define Prompt Template
template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following document:\n\n{text}\n\nSummary:"
)

# Load Cohere LLM and Generate Output
llm = ChatCohere(model_name="command-xlarge-nightly")
response = llm.invoke(template.format(text=text))

# Output
print("Summary:\n", response.content)
