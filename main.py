from EssayWriter import EssayWriter
from EssayWriterGUI import WriterGUI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
_ = load_dotenv(dotenv_path='../.env/.env')
tavily_api_key = str(os.getenv('TAVILY_API_KEY'))
openai_api_key = str(os.getenv('OPENAI_API_KEY'))


model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=openai_api_key)
multi_agent = EssayWriter(model=model, tavily_api_key=tavily_api_key)
app = WriterGUI(multi_agent.graph)
app.launch()
