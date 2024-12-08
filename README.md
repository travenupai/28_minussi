# SWOT

Welcome to the Swot Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

# Novo projeto com Streamlit do ZERO, nada de reaproveitar:
1. Crie um diretório na pasta projetos python e de um número sequencial.


crewai create crew swot  #swot ou o nome que quiser

cd swot

2. Crie um ambiente virtual seguindo os passos abaixo:

se for no python 3.11 para o streamlit:


pip install poetry
py -3.11 -m venv .venv
.venv\Scripts\activate
uv init
uv add crewai
.venv\Scripts\activate
==0.61.0
crewai create crew swot  #swot ou o nome que quiser
cd swot

uv add crewai-tools
uv add streamlit
uv add python-docx
uv add pydantic
uv add langchain-community


ATENCAO: SÓ PRECISA DAR CREWAI INSTAL SE NAO DER PIP INSTALL CREWAI CREWAI-TOOLS



SE VOCÊ CRIA UM NOVO PROJETO OU FLOW OU PIPELINE, DE ESSES COMANDOS:
crewai create crew nome_do_projeto
ou
crewai create flow nome_do_flow
ou
crewai create pipeline nome_do_pipeline


PARA O CASO DE NOVO .VENV :

pip install uv
crewai install
uv lock     # um chamado se abre no antivirus para liberar forever
uv sync     # começa a roda muitas linhas de código pela primeira vez 


Ensure you have Python >=3.10 <=3.13 installed on your system. 
This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.


PARA QUEM JÁ CRIOU O AMBIENTE VIRTUAL
vá para o diretório do projeto antes de src

C:\projetos-python\15_swot\swot
.venv\Scripts\activate    

Não há necessidade de repetir crewai install a menos que você tenha mudado o ambiente (criado um novo ambiente virtual) ou atualizado o pyproject.toml.
uv lock e uv sync: Esses comandos só são necessários para atualizar dependências ou se alguma mudança significativa foi feita no pyproject.toml.

Para rodar o programa:

crewai run      # Para executar o seu projeto principal

# criar um NOVO projeto em python 3.11 com Streamlit :

1. Crie um diretório na pasta projetos python e de um número sequencial.

C:\Users\Ashram\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt
uv lock
uv sync

crewai create crew swot

cd swot

2. Crie um ambiente virtual seguindo os passos abaixo:

C:\Users\Ashram\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt
uv lock
uv sync

streamlit run app.py

## 3. Subir um repositório do VSCode para o GitHub

1. No VSCode, abra o terminal integrado (`Ctrl + '`).
2. Navegue até o diretório do seu projeto, caso ainda não esteja nele:
   ```bash
   cd caminho/para/seu/projeto
   ```
3. Inicialize o repositório local do Git, se ainda não foi feito:
   ```bash
   git init
   ```
4. Adicione os arquivos ao repositório:
   ```bash
   git add .
   ```
5. Faça o primeiro commit:
   ```bash
   git commit -m "Primeiro commit"
   ```
6. Conecte seu repositório local ao GitHub. Use a URL do repositório que você criou no GitHub:
   ```bash
   git remote add origin https://github.com/seu-usuario/nome-repositorio.git
   ```
7. Faça o envio (push) do repositório para o GitHub:
   ```bash
   git push -u origin main
   ```

## 4. Atualizar um repositório do VSCode no GitHub

1. Adicione os novos arquivos ou mudanças ao repositório:
   ```bash
   git add .
   ```
2. Faça o commit com uma mensagem que descreva as alterações:
   ```bash
   git commit -m "Descrição das alterações"
   ```
3. Envie as alterações para o GitHub:
   ```bash
   git push origin main
   ```

## 5. Publicar um projeto do GitHub no Streamlit

1. Crie um arquivo **`requirements.txt`** na raiz do projeto, listando todas as bibliotecas usadas no projeto, por exemplo:
   ```plaintext
   streamlit
   pandas
   numpy
   ```
2. No site do [Streamlit](https://share.streamlit.io/), faça login.
3. Clique em **New App** para iniciar o processo de deploy.
4. No campo **Repository**, insira a URL do seu repositório GitHub, por exemplo:
   ```plaintext
   https://github.com/seu-usuario/nome-repositorio
   ```
5. No campo **Branch**, selecione **main**.
6. No campo **Main file path**, insira o caminho do arquivo Python principal que roda o Streamlit, por exemplo:
   ```plaintext
   src/app.py
   ```
7. Clique em **Deploy** para publicar seu app.

## 6. Voltar para uma versão anterior no GitHub usando o VSCode

1. Verifique o histórico de commits com:
   ```bash
   git log
   ```
   Isso mostrará a lista de commits, incluindo os códigos de hash dos commits (identificadores únicos).
   
2. Para voltar para um commit anterior temporariamente, use o comando:
   ```bash
   git checkout <hash-do-commit>
   ```
   Isso permite que você explore o código como ele estava naquele commit.

3. Para voltar permanentemente para um commit anterior, faça um reset:
   ```bash
   git reset --hard <hash-do-commit>
   ```
   **Atenção:** Isso sobrescreverá as mudanças feitas após esse commit. Use com cuidado.

4. Se você já tiver feito um `reset` e precisar atualizar o repositório remoto, faça o push forçado:
   ```bash
   git push --force
   ```

---

## Como apagar o venv:
No terminal:

deactivate

Remove-Item -Recurse -Force .venv

python -m venv .venv

.venv\Scripts\activate

pip install -U pip
pip install uv
pip install crewai
pip install crewai-tools

uv lock
uv sync



### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/sales_offer/config/agents.yaml` to define your agents
- Modify `src/sales_offer/config/tasks.yaml` to define your tasks
- Modify `src/sales_offer/crew.py` to add your own logic, tools and specific args
- Modify `src/sales_offer/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```
or
```bash
uv run sales_offer
```

This command initializes the sales-offer Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The sales-offer Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the SalesOffer Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
