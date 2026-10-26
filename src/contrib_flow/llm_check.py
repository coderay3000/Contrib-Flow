import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model=os.environ["GROQ_MODEL"], temperature=0)
print(llm.invoke("Reply with the single word: ok").content)
