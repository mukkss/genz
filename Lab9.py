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
    founder = "N/A"
    founded_year = "N/A"
    branches = "N/A"
    employees = "N/A"
    summary = "N/A"
    lines = text.strip().split("\n")

    for line in lines:
        if line.lower().startswith("founder:"):
            founder = line.split(":", 1)[1].strip()
        elif line.lower().startswith("founded:"):
            founded_year = line.split(":", 1)[1].strip()
        elif line.lower().startswith("branches:"):
            branches = line.split(":", 1)[1].strip()
        elif line.lower().startswith("employees:"):
            employees = line.split(":", 1)[1].strip()
        elif line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()

    return InstitutionInfo(
        founder=founder,
        founded_year=founded_year,
        branches=branches,
        employees=employees,
        summary=summary
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
