# crew.py

import os
import openai
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, ScrapeElementFromWebsiteTool
from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

openai.api_key = api_key
# Initialize the LLM
llm                   = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
gpt_mini              = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
gpt4o_mini_2024_07_18 = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", openai_api_key=api_key)
gpt4o                 = ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key)
gpt_o1                = ChatOpenAI(model_name="o1-preview", openai_api_key=api_key)
gpt_o1_mini           = ChatOpenAI(model_name="o1-mini", openai_api_key=api_key)


# Initialize tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
scrape_element_tool = ScrapeElementFromWebsiteTool()

@CrewBase
class SwotCrew():
	"""Swot crew"""

	@agent
	def agente_extracao(self) -> Agent:
		return Agent(
			config=self.agents_config['agente_extracao'],
			tools=[search_tool], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			allow_delegation=True,
            allow_interruption=True,
            allow_fallback=True,
            memory=True,
            llm=llm
		)

	@agent
	def agente_solucoes_ia(self) -> Agent:
		return Agent(
			config=self.agents_config['agente_solucoes_ia'],
			tools=[scrape_tool],
			verbose=True,
			allow_delegation=True,
            allow_interruption=True,
            allow_fallback=True,
            memory=True,
            llm=llm
		)

	@agent
	def analista_swot(self) -> Agent:
		return Agent(
			config=self.agents_config['analista_swot'],
			tools=[scrape_tool],
			verbose=True,
			allow_delegation=True,
            allow_interruption=True,
            allow_fallback=True,
            memory=True,
            llm=llm
		)

	@agent
	def analista_financiamento(self) -> Agent:
		return Agent(
			config=self.agents_config['analista_financiamento'],
			tools=[scrape_tool],
			verbose=True,
			allow_delegation=True,
            allow_interruption=True,
            allow_fallback=True,
            memory=True,
            llm=llm
		)

	@agent
	def agente_analise_recomendacao(self) -> Agent:
		return Agent(
			config=self.agents_config['agente_analise_recomendacao'],
			verbose=True,
			allow_delegation=True,
            allow_interruption=True,
            allow_fallback=True,
            memory=True,
            llm=llm
		)


	@task
	def extrair_informacoes_site(self) -> Task:
		return Task(
			config=self.tasks_config['extrair_informacoes_site'],
			guardrails=[{"output_format": "markdown"}, {"max_length": 8000}],
		)

	@task
	def pesquisar_solucoes_ia(self) -> Task:
		return Task(
			config=self.tasks_config['pesquisar_solucoes_ia'],
   			output_file='pesquisar_solucoes_ia.md',
            guardrails=[{"output_format": "markdown"}, {"max_length": 8000}],
		)
  
	@task
	def analise_swot(self) -> Task:
		return Task(
			config=self.tasks_config['analise_swot'],
   			output_file='analise_swot.md',
            guardrails=[{"output_format": "markdown"}, {"max_length": 8000}],
		)
  
	@task
	def financiamento_estrategico(self) -> Task:
		return Task(
			config=self.tasks_config['financiamento_estrategico'],
   			output_file='financiamento_estrategico.md',
            guardrails=[{"output_format": "markdown"}, {"max_length": 8000}],
		)
  
	@task
	def analisar_recomendar(self) -> Task:
		return Task(
			config=self.tasks_config['analisar_recomendar'],
			output_file='analisar_recomendar_swot.md',
            guardrails=[{"output_format": "markdown"}, {"max_length": 8000}]
		)


	@crew
	def crew(self) -> Crew:
		"""Creates the Swot crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			memory=True  # Activate memory
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)