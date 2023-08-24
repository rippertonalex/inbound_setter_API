import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import openai





serper_api_key= os.environ['SERP_API_KEY']
brwoserless_api_key  = os.environ['BROWSERLESS_API_KEY']

print(brwoserless_api_key)

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

#search("what is a produc")


#tool for scraping 

def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text   #this text is going to be too large token limit issues 
    else:
        print(f"HTTP request failed with status code {response.status_code}")


#vector search is one way to get around token limit issues where the llm will search for relevant content
#you could also use map reduce which essentially summarizes the information to meet token requirements 
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


#gives a description of the inputs for the scrape website function so that the LLM agent knows what to pass when it calls it itself
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

#this class describes what the scrapewebsite tool is for the LLM and says when it should be run
class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

#create langchain agent with the tools above

#defining the agents two tools Search and scrape website 
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]
#consider adusting the system message for specifc usecases 
system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)
#agent is being fed chat history up to 1000 tokens then it just looks at summary information. as well as the system message 
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )



#unleash_agent_on_sheets(r'/Users/strippperton/Downloads/research_bot_input - Sheet1.csv', r'/Users/strippperton/Downloads/fifty_research_agent_output.csv')

def set_api_key(api_key):
  #OpenAI.api_key = api_key
  os.environ["OPENAI_API_KEY"] = api_key
  openai.api_key = api_key
  print(str(api_key))
  if is_api_key_valid():
      return 'API Key has been set'
  else:
    return 'Not a Valid Key'

def is_api_key_valid():
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
    except Exception as e :
        print(e)
        return False
    else:
        return True

app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content

#LETS  BUILD A GRADIO FRONTEND
'''
try:
    with gr.Blocks() as demo:
        gr.Markdown("This is a Lead Enhancement Application. You can upload thousands of leads at once and get insights into each through our CSV file option. You can test the application and get individual lead enhancements through our Lead Enhancement via text option.")
        with gr.Tab("Lead Enhancement via CSV file"):
            openai_api_key = gr.Textbox(label= 'OpenAIAPIKey: ', type= 'password' )
            api_out = gr.Textbox(label='API key stauts ')
            api_button = gr.Button('set api key')
            api_button.click(set_api_key, openai_api_key, api_out)
            #gr.Examples(examples=[os.path.join(os.path.dirname(__file__), "test.csv")], inputs= gr.File())
            file_input = gr.File(file_types = ['text'], label = 'Lead CSV with the following columns: full_name, title, location, company, company_industry, company_description (please match column names exactly. Values can be empty.)')
            gr.Examples(examples=[os.path.join(os.path.dirname(__file__), "test.csv")], inputs= file_input)
            #examples = gr.Examples()
            #df_output = gr.outputs.Dataframe(type = 'pandas')
            file_output = gr.outputs.File()
            text_button = gr.Button("Enhance my Leads")
            text_button.click(unleash_agent_on_sheets, file_input, file_output)
            #gr.Label("INPUT FILE EXAMPLE")
            #gr.Examples(examples=[os.path.join(os.path.dirname(__file__), "test.csv")], inputs= file_input)
        with gr.Tab("Lead Enhancement via text"):
            openai_api_key = gr.Textbox(label= 'OpenAIAPIKey: ', type= 'password' )
            api_out = gr.Textbox(label='API key stauts ')
            api_button = gr.Button('set api key')
            api_button.click(set_api_key, openai_api_key, api_out)
            text_input = gr.Textbox(label= 'Example usage: Give me information for a sales call with Rod Noles an employee at Noles-Frye Realty. A Real Estate company that specializes in home sale across Louisiana & Mississippi based out of Alexandria, Louisiana, United States. ')
            text_output = gr.outputs.Textbox(label='Enhanced Leads')
            text_button = gr.Button("Enhance my Lead")
            text_button.click(unleash_agent_on_text, text_input, text_output)
        with gr.Tab("Lead Generation (Beta)"):
            openai_api_key = gr.Textbox(label= 'OpenAIAPIKey: ', type= 'password' )
            api_out = gr.Textbox(label='API key stauts ')
            api_button = gr.Button('set api key')
            api_button.click(set_api_key, openai_api_key, api_out)
            text_input = gr.Textbox(label= 'Describe the kind of leads you are looking for. Example: I want leads for small to medium sized ciminal defense law firms that are looking to buy AI automation software in southern california.')
            text_output = gr.outputs.Textbox(label='Generated Leads')
            text_button = gr.Button("Generate Leads")
            text_button.click(generate_leads_from_text, text_input, text_output)
    demo.launch() #share= True
except:
    gr.Error(message= 'Make sure your API Key is correct and try again')'''

#Give me information for a sales call with Rod Noles an employee at Noles-Frye Realty. A Real Estate company that specializes in home sale across Louisiana & Mississippi based out of Alexandria, Louisiana, United States.
# 1.Click on the example file 
#         2. Click the Blue Download button
#         3. Put the newly downloaded test.csv in the above drop file location
#         4. Click on the enhance my leads button
#         5. Download the output file once the program is done running
