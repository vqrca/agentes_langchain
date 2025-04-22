import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from ferramentas import criar_ferramentas
import os

# Inicia o app
st.set_page_config(page_title="Assistente de Dados com IA", layout="centered")
st.title("🦜 Assistente de análise de dados com IA")
# Descrição da ferramenta
st.info("""
Este assistente utiliza um agente, criado com Langchain, para te ajudar a explorar, analisar e visualizar dados de forma interativa.
Basta fazer o upload de um arquivo CSV e você poderá:

- 📄 **Gerar relatórios automáticos**:
    - **Relatório de informações gerais**: apresenta a dimensão do DataFrame, nomes e tipos das colunas, contagem de dados nulos e duplicados, além de sugestões de tratamentos e análises adicionais.
    - **Relatório de estatísticas descritivas**: exibe valores como média, mediana, desvio padrão, mínimo e máximo; identifica possíveis outliers e sugere próximos passos com base nos padrões detectados.

- 🔎 **Fazer perguntas simples sobre os dados**: como "Qual é a média do tempo de entrega?", "Quantos registros existem para cada categoria?", ou "Qual é o valor máximo da coluna de vendas?".
                
- 📊 **Criar gráficos automaticamente** com base em perguntas em linguagem natural.


        
Ideal para analistas, cientistas de dados e equipes que buscam agilidade e insights rápidos com apoio de IA.
""")

# Upload do CSV
# Upload do CSV
st.markdown("### 📁 Faça upload do seu arquivo CSV")
arquivo_carregado = st.file_uploader("Selecione um arquivo CSV", type="csv", label_visibility="collapsed")

if arquivo_carregado:
    df = pd.read_csv(arquivo_carregado)
    st.success("Arquivo carregado com sucesso!")
    st.markdown("### 🔍 Primeiras linhas do DataFrame")
    st.dataframe(df.head())

    # LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0
    )

    # Ferramentas
    tools = criar_ferramentas(df)

    # Prompt react
    df_head = df.head().to_markdown()

    prompt_react_pt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"df_head": df_head},
        template="""
        Você é um assistente que sempre responde em português.

        Você tem acesso a um dataframe pandas chamado `df`.
        Aqui estão as primeiras linhas do DataFrame, obtidas com `df.head().to_markdown()`:

        {df_head}

        Responda às seguintes perguntas da melhor forma possível.

        Para isso, você tem acesso às seguintes ferramentas:

        {tools}

        Use o seguinte formato:

        Question: a pergunta de entrada que você deve responder  
        Thought: você deve sempre pensar no que fazer  
        Action: a ação a ser tomada, deve ser uma das [{tool_names}]  
        Action Input: a entrada para a ação  
        Observation: o resultado da ação  
        ... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
        Thought: Agora eu sei a resposta final  
        Final Answer: a resposta final para a pergunta de entrada original.
        Quando usar a ferramenta_python: formate sua resposta final de forma clara, em lista, com valores separados por vírgulas e duas casas decimais sempre que apresentar números.

        Comece!

        Question: {input}  
        Thought: {agent_scratchpad}"""
    )

    
    # Agente
    agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt)
    orquestrador = AgentExecutor(agent=agente,
                                tools=tools,
                                verbose=True,
                                handle_parsing_errors=True)

    # AÇÕES RÁPIDAS
    st.markdown("---")
    st.markdown("## ⚡ Ações rápidas")

    # Relatório de informações gerais
    if st.button("📄 Relatório de informações gerais", key="botao_relatorio_geral"):
        with st.spinner("Gerando relatório 🦜"):
            resposta = orquestrador.invoke({"input": "Quero um relatório com informações sobre os dados"})
            st.session_state['relatorio_geral'] = resposta["output"]

    # Exibe o relatório com botão de download
    if 'relatorio_geral' in st.session_state:
        with st.expander("Resultado: Relatório de informações gerais"):
            st.markdown(st.session_state['relatorio_geral'])

            st.download_button(
                label="📥 Baixar relatório",
                data=st.session_state['relatorio_geral'],
                file_name="relatorio_informacoes_gerais.md",
                mime="text/markdown"
            )

    # Relatório de estatísticas descritivas
    if st.button("📄 Relatório de estatísticas descritivas", key="botao_relatorio_estatisticas"):
        with st.spinner("Gerando relatório 🦜"):
            resposta = orquestrador.invoke({"input": "Quero um relatório de estatísticas descritivas"})
            st.session_state['relatorio_estatisticas'] = resposta["output"]

    # Exibe o relatório salvo com opção de download
    if 'relatorio_estatisticas' in st.session_state:
        with st.expander("Resultado: Relatório de estatísticas descritivas"):
            st.markdown(st.session_state['relatorio_estatisticas'])

            st.download_button(
                label="📥 Baixar relatório",
                data=st.session_state['relatorio_estatisticas'],
                file_name="relatorio_estatisticas_descritivas.md",
                mime="text/markdown"  # <- use mime correto
            )

    # PERGUNTA SOBRE OS DADOS
    st.markdown("---")
    st.markdown("## 🔎 Perguntas sobre os dados")
    pergunta_sobre_dados = st.text_input("Faça uma pergunta sobre os dados (ex: 'Qual é a média do tempo de entrega?')")
    if st.button("Responder pergunta", key="responder_pergunta_dados"):
        with st.spinner("Analisando os dados 🦜"):
            resposta = orquestrador.invoke({"input": pergunta_sobre_dados})
            st.markdown((resposta["output"]))

       
    # GERAÇÃO DE GRÁFICOS
    st.markdown("---")
    st.markdown("## 📊 Criar gráfico com base em uma pergunta")

    pergunta_grafico = st.text_input("Digite o que deseja visualizar (ex: 'Crie um gráfico da média de tempo de entrega por clima.')")
    if st.button("Gerar gráfico", key="gerar_grafico"):
        with st.spinner("Gerando o gráfico 🦜"):
            orquestrador.invoke({"input": pergunta_grafico})




