# 📊 PH Score — Scouting & Estatísticas dos Jogos (Big 5 + UEFA)

Painel web interativo e pipeline automatizado em **Python + GitHub Actions** para acompanhamento de confrontos, estatísticas dos últimos 5 jogos (em todas as competições), matriz individual de jogadores e **Top 5 Insights** das partidas.

---

## 🏗️ Arquitetura e Stack Tecnológica

* **Linguagem & Runtime:** Python 3.11
* **Coleta de Confrontos & Histórico:** `football-data.org` API v4 e bases CSV históricas (`pandas`) para os **últimos 5 jogos gerais** de cada equipe (independente de competição ou mando de campo)
* **Enriquecimento com IA:** Google Gemini 2.5 Flash (`google-genai` com Structured Outputs via `pydantic`) para dados de arbitragem, estádio, cores dos clubes e elencos/métricas de jogadores
* **Automação & Publicação:** GitHub Actions (`cron` diário às **07h30 BRT** / `10:30 UTC` + disparo manual via `workflow_dispatch`) publicando o `index.html` no **GitHub Pages**
* **Frontend:** HTML, CSS e JavaScript puro com números tabulares, filtros por competição, ordenação de tabelas e resumo copiável

---

## ⚙️ Funcionalidades do Painel

* **Janela de Jogos Otimizada:** Coleta os confrontos em uma janela deslizante de **3 dias** (dia atual + 2 dias seguintes), alinhada à atualização diária da manhã para manter boa performance e estabilidade nas chamadas de API.
* **8 Competições Cobertas:**
  * Premier League
  * La Liga
  * Serie A
  * Bundesliga
  * Ligue 1
  * UEFA Champions League
  * UEFA Europa League
  * UEFA Conference League
* **Comparativo das 18 Métricas:** Exibe a sequência dos últimos 5 jogos e a média por partida para cada equipe, com opção **⚡ Só Vantagens** para destacar indicadores favoráveis.
* **Matriz Individual de Scouting:** Tabelas comparativas por elenco com busca por nome/posição/status e ordenação por `FIN`, `SOT`, `DES`, `FC` e `FS`.
* **Top 5 Insights ao Final da Página:** Bloco final que destaca automaticamente os 5 principais pontos de atenção estatística (projeção de gols, volume ofensivo e destaques individuais) com base nos jogos carregados.
* **Interface Focada em Dados:** Sem botão de adição manual de jogos e com paleta escura de alto contraste sob a marca **PH Score**.

---

## 🔐 Secrets Necessários (GitHub Actions)

Configure em **Settings → Secrets and variables → Actions** no repositório:

* `FOOTBALL_DATA_KEY`: Chave de API do `football-data.org`.
* `GEMINI_API_KEY`: Chave da API do Google Gemini.

> **Permissões do Workflow:** Em **Settings → Actions → General**, garanta que **Workflow permissions** esteja habilitado com **Read and write permissions** para que o workflow possa fazer commit e push do `index.html` atualizado.

---

## 📁 Estrutura do Repositório

```text
Estatisticas-dos-Jogos-do-Dia/
├── .github/
│   └── workflows/
│       └── atualizar_painel.yml   # Execução diária às 07h30 BRT (10:30 UTC) + manual
├── gerar_painel.py                # Coleta dados, cruza últimos 5 jogos, consulta Gemini e gera index.html
├── template.html                  # Template visual do PH Score com Top 5 Insights
├── requirements.txt               # Dependências Python
└── index.html                     # Painel publicado no GitHub Pages (gerado automaticamente)
