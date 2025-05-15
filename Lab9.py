!pip install wikipedia_api langchain-cohere

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_cohere.chat_models import ChatCohere
from pydantic import BaseModel
import wikipediaapi
import os

# Define the schema
class InstitutionInfo(BaseModel):
    founder: str
    founded_year: str
    branches: str
    employees: str
    summary: str

# Custom output parser
def parse_response(text: str) -> InstitutionInfo:
    data = {k.lower(): "N/A" for k in ["founder", "founded", "branches", "employees", "summary"]}
    for line in text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if key in data:
                data[key] = value.strip()
    return InstitutionInfo(
        founder=data["founder"],
        founded_year=data["founded"],
        branches=data["branches"],
        employees=data["employees"],
        summary=data["summary"]
    )

# Setup
os.environ["COHERE_API_KEY"] = "YrqBdTypjvdKMc7bBljLwihs5TS54JCN8qjrLVQ5"
llm = ChatCohere(model_name="command-xlarge-nightly")
wiki = wikipediaapi.Wikipedia(user_agent='genai-lab', language='en')

# Input
institution = input("Enter Institution Name: ")
page = wiki.page(institution)

if not page.exists():
    print(f"No Wikipedia page found for '{institution}'.")
else:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Extract these details from the text:
        - Founder
        - Year Founded
        - Number of Branches
        - Number of Employees
        - A brief 4-line summary of the institution (e.g., "Founded in 1999, XYZ Corporation is a leading provider of...")

        Text: {text}

        Format:
        Founder: <...>
        Founded: <...>
        Branches: <...>
        Employees: <...>
        Summary: <...>
        """
    )
    response = llm.invoke(prompt.format(text=page.text))
    info = parse_response(response.content)
    print("\nInstitution Details:")
    print(f"Founder       : {info.founder}")
    print(f"Founded Year  : {info.founded_year}")
    print(f"Branches      : {info.branches}")
    print(f"Employees     : {info.employees}")
    print(f"Summary       : {info.summary}")
