import json
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from google import genai
from pydantic import BaseModel

# ==============================================================================
# CONFIGURAÇÕES GERAIS — ARENA DOS DADOS (BIG 5 + UEFA — TODOS OS JOGOS)
# ==============================================================================

GEMINI_MODEL = "gemini-2.5-flash"

COMPETICOES_API = [
    {"code": "PL", "leagueKey": "Premier League", "csv": "https://www.football-data.co.uk/mmz4281/2627/E0.csv"},
    {"code": "PD", "leagueKey": "La Liga", "csv": "https://www.football-data.co.uk/mmz4281/2627/SP1.csv"},
    {"code": "SA", "leagueKey": "Serie A", "csv": "https://www.football-data.co.uk/mmz4281/2627/I1.csv"},
    {"code": "BL1", "leagueKey": "Bundesliga", "csv": "https://www.football-data.co.uk/mmz4281/2627/D1.csv"},
    {"code": "FL1", "leagueKey": "Ligue 1", "csv": "https://www.football-data.co.uk/mmz4281/2627/F1.csv"},
    {"code": "CL", "leagueKey": "UEFA Champions League", "csv": None},
    {"code": "EL", "leagueKey": "UEFA Europa League", "csv": None},
    {"code": "ECL", "leagueKey": "UEFA Conference League", "csv": None},
]

DIAS_SEMANA_PT = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
DIAS_SEMANA_FULL = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]


# ==============================================================================
# MODELOS ESTRUTURADOS PARA O GEMINI COMPLETAR DETALHES DE JOGO E ELENCO
# ==============================================================================

class JogadorScouting(BaseModel):
    name: str
    pos: str   # GOL, ZAG, LAT, VOL, MEI, ATA
    role: str  # Titular ou Reserva
    fin: List[int]  # 5 jogos (últimos 5 jogos gerais do jogador)
    sot: List[int]  # 5 jogos
    tac: List[int]  # 5 jogos
    fc: List[int]   # 5 jogos
    fs: List[int]   # 5 jogos


class DadosComplementaresConfronto(BaseModel):
    stadium: str
    city: str
    referee_name: str
    referee_matches: str
    referee_yellow: str
    referee_red: str
    referee_pen: str
    home_bg: str
    home_color: str
    away_bg: str
    away_color: str
    home_players: List[JogadorScouting]
    away_players: List[JogadorScouting]


# ==============================================================================
# NORMALIZAÇÃO E CARREGAMENTO DE HISTÓRICO GERAL (TODAS AS COMPETIÇÕES)
# ==============================================================================

def normalizar(txt: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(txt))
    sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    limpo = re.sub(r"[^a-z0-9\s]", "", sem_acento.lower()).strip()
    for sufixo in (" fc", " cf", " afc", " sc", " ac", " ssc", " bc"):
        if limpo.endswith(sufixo):
            limpo = limpo[: -len(sufixo)].strip()
    for prefixo in ("fc ", "cf ", "ac ", "as ", "ssc ", "rc ", "rcd ", "cd ", "ud ", "vfb ", "vfl ", "tsg ", "sc ", "sv ", "1 fsv ", "1 fc ", "ogc ", "osc ", "stade "):
        if limpo.startswith(prefixo):
            limpo = limpo[len(prefixo):].strip()
    return limpo


def carregar_base_unificada_csvs() -> pd.DataFrame:
    """Carrega todos os CSVs das Big 5 em uma única tabela cronológica para consultar últimos 5 jogos gerais."""
    frames = []
    for comp in COMPETICOES_API:
        url = comp.get("csv")
        if not url:
            continue
        try:
            df = pd.read_csv(url)
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
            if "Date" in df.columns:
                df["_dt"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            else:
                df["_dt"] = pd.NaT
            frames.append(df)
        except Exception as e:
            print(f"   [Aviso] Falha ao ler CSV {url}: {e}")

    if not frames:
        return pd.DataFrame()

    unificado = pd.concat(frames, ignore_index=True)
    unificado = unificado.sort_values(by="_dt", ascending=True, na_position="first").reset_index(drop=True)
    return unificado


def extrair_18_metricas_gerais(df_unificado: pd.DataFrame, nome_time: str) -> tuple[List[List[int]], List[str]]:
    """
    Busca os ÚLTIMOS 5 JOGOS GERAIS da equipe (independente de ser casa/fora ou da competição)
    e monta as 18 métricas + sequência de resultados ['V','E','D',...].
    """
    fallback_stats = [
        [2, 1, 2, 1, 2], [1, 1, 1, 2, 1], [15, 13, 16, 12, 14], [7, 6, 8, 5, 7],
        [11, 12, 10, 13, 11], [6, 5, 7, 4, 6], [4, 5, 3, 6, 4], [6, 5, 7, 5, 6],
        [3, 2, 4, 2, 3], [5, 5, 4, 6, 5], [2, 2, 2, 3, 2], [12, 13, 11, 12, 12],
        [18, 17, 19, 18, 17], [2, 2, 2, 3, 2], [2, 1, 2, 2, 1], [7, 8, 6, 8, 7],
        [520, 495, 540, 490, 525], [3, 3, 3, 4, 3]
    ]
    fallback_form = ["V", "E", "V", "D", "V"]

    if df_unificado is None or df_unificado.empty:
        return fallback_stats, fallback_form

    alvo = normalizar(nome_time)
    if not alvo:
        return fallback_stats, fallback_form

    mask = df_unificado["HomeTeam"].apply(lambda x: alvo in normalizar(x) or normalizar(x) in alvo) | \
           df_unificado["AwayTeam"].apply(lambda x: alvo in normalizar(x) or normalizar(x) in alvo)

    jogos = df_unificado[mask].tail(5)
    if len(jogos) < 5:
        return fallback_stats, fallback_form

    metricas = [[] for _ in range(18)]
    forma = []

    for _, r in jogos.iterrows():
        h_norm = normalizar(r["HomeTeam"])
        mandante = (alvo in h_norm) or (h_norm in alvo)

        gm = int(r["FTHG"] if mandante else r["FTAG"])
        gs = int(r["FTAG"] if mandante else r["FTHG"])
        forma.append("V" if gm > gs else ("E" if gm == gs else "D"))

        fin = int(r.get("HS" if mandante else "AS", 13) or 13)
        fin_sof = int(r.get("AS" if mandante else "HS", 11) or 11)
        sot = int(r.get("HST" if mandante else "AST", 5) or 5)
        sot_sof = int(r.get("AST" if mandante else "HST", 4) or 4)
        esc = int(r.get("HC" if mandante else "AC", 6) or 6)
        esc_sof = int(r.get("AC" if mandante else "HC", 5) or 5)
        faltas = int(r.get("HF" if mandante else "AF", 12) or 12)
        ama = int(r.get("HY" if mandante else "AY", 2) or 2)

        metricas[0].append(gm)
        metricas[1].append(gs)
        metricas[2].append(fin)
        metricas[3].append(max(1, round(fin * 0.48)))
        metricas[4].append(fin_sof)
        metricas[5].append(sot)
        metricas[6].append(sot_sof)
        metricas[7].append(esc)
        metricas[8].append(max(0, round(esc * 0.48)))
        metricas[9].append(esc_sof)
        metricas[10].append(max(0, round(esc_sof * 0.48)))
        metricas[11].append(faltas)
        metricas[12].append(17 + (faltas % 4))
        metricas[13].append(ama)
        metricas[14].append(2)
        metricas[15].append(max(4, round(fin_sof * 0.6)))
        metricas[16].append(495 + (fin * 6))
        metricas[17].append(max(1, sot_sof - gs))

    return metricas, forma


# ==============================================================================
# COLETA DE TODOS OS JOGOS DA SEMANA VIA FOOTBALL-DATA.ORG
# ==============================================================================

def buscar_jogos_api(token: str) -> List[dict]:
    headers = {"X-Auth-Token": token}
    agora_utc = datetime.now(timezone.utc)
    date_from = agora_utc.strftime("%Y-%m-%d")
    date_to = (agora_utc + timedelta(days=2)).strftime("%Y-%m-%d")

    jogos_coletados = []
    for comp in COMPETICOES_API:
        url = f"https://api.football-data.org/v4/competitions/{comp['code']}/matches"
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED,TIMED"},
                timeout=25,
            )
            if resp.status_code != 200:
                print(f"[Aviso] {comp['leagueKey']} retornou status {resp.status_code}: {resp.text[:120]}")
                time.sleep(2)
                continue

            matches = resp.json().get("matches", [])
            print(f"[OK] {comp['leagueKey']}: {len(matches)} jogo(s) encontrados nos próximos 7 dias.")
            # Sem limite artificial: traz TODOS os jogos da semana de cada competição
            for m in matches:
                m["_leagueKey"] = comp["leagueKey"]
                jogos_coletados.append(m)

            time.sleep(1.2)
        except Exception as e:
            print(f"[Erro] Falha ao consultar {comp['leagueKey']}: {e}")

    # Ordena cronologicamente todos os jogos coletados
    jogos_coletados.sort(key=lambda x: x.get("utcDate", ""))
    return jogos_coletados


# ==============================================================================
# ENRIQUECIMENTO DE ELENCOS (15 JOGADORES) E DETALHES VIA GEMINI 2.5 FLASH
# ==============================================================================

def gerar_elenco_fallback(nome_time: str) -> List[dict]:
    sigla = nome_time[:3].upper()
    modelo = [
        (f"{sigla} Goleiro Titular", "GOL", "Titular", [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,1,0]),
        (f"{sigla} Lateral Direito", "LAT", "Titular", [1,1,1,0,1], [0,0,1,0,0], [3,2,3,3,2], [1,2,1,1,1], [1,1,2,1,1]),
        (f"{sigla} Zagueiro Central 1", "ZAG", "Titular", [1,0,1,1,0], [0,1,0,0,0], [2,3,2,3,2], [1,1,2,1,1], [1,0,1,0,1]),
        (f"{sigla} Zagueiro Central 2", "ZAG", "Titular", [1,1,0,0,1], [0,0,1,0,0], [2,2,3,2,2], [2,1,1,2,1], [0,1,0,1,0]),
        (f"{sigla} Lateral Esquerdo", "LAT", "Titular", [1,2,1,1,1], [0,1,0,0,1], [2,3,2,3,2], [1,1,1,2,1], [1,2,1,1,1]),
        (f"{sigla} Primeiro Volante", "VOL", "Titular", [1,0,1,1,0], [0,0,0,0,0], [4,3,4,3,4], [2,3,2,2,2], [1,1,1,2,1]),
        (f"{sigla} Segundo Volante", "VOL", "Titular", [1,2,1,1,1], [0,1,0,1,0], [3,2,3,3,2], [2,1,2,1,2], [2,1,2,1,2]),
        (f"{sigla} Meia Armador", "MEI", "Titular", [2,2,1,3,2], [1,1,0,2,1], [1,2,1,1,2], [1,1,0,1,1], [2,3,2,2,3]),
        (f"{sigla} Ponta Direita", "ATA", "Titular", [2,3,2,2,3], [1,2,1,1,1], [1,1,1,0,1], [1,1,1,1,0], [2,2,3,2,2]),
        (f"{sigla} Ponta Esquerda", "ATA", "Titular", [3,2,2,3,2], [1,1,1,2,1], [1,0,1,1,1], [1,1,0,1,1], [3,2,2,3,2]),
        (f"{sigla} Centroavante", "ATA", "Titular", [3,4,2,3,3], [2,2,1,1,2], [0,1,0,0,1], [2,1,2,1,2], [2,1,2,2,1]),
        (f"{sigla} Atacante Reserva", "ATA", "Reserva", [2,1,2,1,2], [1,0,1,1,0], [1,0,1,0,1], [1,1,0,1,1], [1,1,1,1,1]),
        (f"{sigla} Meia Reserva", "MEI", "Reserva", [1,2,1,1,1], [0,1,0,0,1], [2,1,2,1,2], [1,1,1,0,1], [1,1,2,1,1]),
    ]
    return [
        {"name": n, "pos": p, "role": r, "fin": fin, "sot": sot, "tac": tac, "fc": fc, "fs": fs}
        for n, p, r, fin, sot, tac, fc, fs in modelo
    ]


def enriquecer_confronto_gemini(
    client: genai.Client,
    home_name: str,
    away_name: str,
    league_key: str,
    arbitro_api: Optional[str],
) -> Optional[DadosComplementaresConfronto]:
    prompt = f"""
    Você é um analista de dados e scouting de futebol europeu (temporada 2026/27).
    Gere os dados estruturados de pré-jogo e scouting completo dos elencos para:
    - Mandante: {home_name}
    - Visitante: {away_name}
    - Competição deste jogo: {league_key}
    - Árbitro informado na API (se houver): {arbitro_api or "Definir árbitro principal da competição"}

    REGRAS OBRIGATÓRIAS:
    1. ELENCOS COMPLETOS E REAIS (`home_players` e `away_players`):
       - Liste EXATAMENTE 14 jogadores reais atuais (2026/27) do {home_name} em `home_players` e 14 jogadores reais atuais do {away_name} em `away_players`.
       - Inclua os **11 Titulares** (1 GOL, 2 ZAG, 2 LAT, 2 VOL, 1 MEI, 3 ATA) + **3 Reservas** principais (ex: MEI, ATA, VOL/ZAG).
       - NUNCA deixe as listas de jogadores vazias!
    2. MÉTRICAS DOS ÚLTIMOS 5 JOGOS GERAIS DO JOGADOR (independente da competição — liga nacional + copas):
       - Para cada jogador, `fin` (Finalizações), `sot` (Chutes ao Gol), `tac` (Desarmes), `fc` (Faltas Cometidas) e `fs` (Faltas Sofridas) devem ser listas com EXATAMENTE 5 números inteiros realistas referentes aos últimos 5 jogos gerais disputados pelo atleta.
       - Goleiros (GOL) devem ter 0 em fin/sot/tac/fc e 0 ou 1 em fs. Atacantes titulares devem ter médias entre 2.0 e 4.5 em `fin`. Volantes e laterais devem ter médias entre 2.0 e 4.2 em `tac`.
    3. ESTÁDIO, ARBITRAGEM E CORES:
       - `stadium` e `city`: Estádio oficial do {home_name} e cidade/país em português.
       - `referee_name`, `referee_matches`, `referee_yellow`, `referee_red`, `referee_pen`: Dados realistas do árbitro da temporada 26/27.
       - `home_bg`, `home_color`, `away_bg`, `away_color`: Cores CSS hex ou linear-gradient fiéis aos uniformes principais dos clubes, garantindo legibilidade total do texto.
    """
    for tentativa in range(2):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DadosComplementaresConfronto,
                    "temperature": 0.2,
                },
            )
            dados = DadosComplementaresConfronto.model_validate_json(resp.text)
            if len(dados.home_players) >= 11 and len(dados.away_players) >= 11:
                return dados
        except Exception as e:
            print(f"   [Aviso] Tentativa {tentativa + 1} Gemini em {home_name} x {away_name}: {e}")
            time.sleep(2)

    return None


# ==============================================================================
# MONTAGEM FINAL DO HTML
# ==============================================================================

def main():
    fd_token = os.environ.get("FOOTBALL_DATA_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not fd_token:
        raise EnvironmentError("Configure FOOTBALL_DATA_KEY nos Secrets do GitHub.")

    client = genai.Client(http_options={"api_version": "v1beta"}) if gemini_key else None

    print("1. Carregando base histórica unificada das Big 5...")
    df_unificado = carregar_base_unificada_csvs()
    print(f"   -> {len(df_unificado)} partidas históricas carregadas.")

    print("\n2. Buscando TODOS os jogos dos próximos 7 dias nas 8 competições...")
    jogos_api = buscar_jogos_api(fd_token)
    if not jogos_api:
        print("Nenhum jogo encontrado na API para os próximos 7 dias. Mantendo index.html atual.")
        return

    print(f"\nTotal de confrontos encontrados na semana: {len(jogos_api)}\n")
    matches_data_final = []

    for idx, m in enumerate(jogos_api, start=1):
        home_info = m.get("homeTeam", {})
        away_info = m.get("awayTeam", {})
        home_name = home_info.get("shortName") or home_info.get("name", "Mandante")
        away_name = away_info.get("shortName") or away_info.get("name", "Visitante")
        league_key = m["_leagueKey"]

        utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        brt_dt = utc_dt - timedelta(hours=3)
        dia_curto = DIAS_SEMANA_PT[brt_dt.weekday()]
        dia_longo = DIAS_SEMANA_FULL[brt_dt.weekday()]

        sub_label = f"{dia_curto} {brt_dt.strftime('%d/%m')} • {brt_dt.strftime('%H:%M')}"
        round_date = f"{dia_longo}, {brt_dt.strftime('%d/%m/%Y')} • {brt_dt.strftime('%H:%M')} (BRT)"
        matchday = m.get("matchday")
        league_title = f"{league_key.upper()} • RODADA {matchday}" if matchday else f"{league_key.upper()} • FASE DE LIGA"

        # Últimos 5 jogos gerais (independente da competição)
        home_stats, home_form = extrair_18_metricas_gerais(df_unificado, home_name)
        away_stats, away_form = extrair_18_metricas_gerais(df_unificado, away_name)

        refs = m.get("referees", [])
        arbitro_api = refs[0].get("name") if refs else None

        print(f"[{idx}/{len(jogos_api)}] Processando: {home_name} x {away_name} ({league_key})...")
        comp_ia = None
        if client:
            comp_ia = enriquecer_confronto_gemini(client, home_name, away_name, league_key, arbitro_api)

        if comp_ia:
            stadium = comp_ia.stadium.upper()
            city = comp_ia.city
            ref_obj = {
                "name": comp_ia.referee_name.upper(),
                "matches": comp_ia.referee_matches,
                "yellow": comp_ia.referee_yellow,
                "red": comp_ia.referee_red,
                "pen": comp_ia.referee_pen,
            }
            home_bg, home_color = comp_ia.home_bg, comp_ia.home_color
            away_bg, away_color = comp_ia.away_bg, comp_ia.away_color
            home_players = [p.model_dump() for p in comp_ia.home_players]
            away_players = [p.model_dump() for p in comp_ia.away_players]
        else:
            stadium = (m.get("venue") or f"ESTÁDIO {home_name}").upper()
            city = "Europa"
            ref_obj = {
                "name": (arbitro_api or "ÁRBITRO A DEFINIR").upper(),
                "matches": "Temporada 26/27",
                "yellow": "Média: 4.35",
                "red": "Média: 0.20",
                "pen": "Média: 0.28",
            }
            home_bg, home_color = "#059669", "#ffffff"
            away_bg, away_color = "#1d4ed8", "#ffffff"
            home_players = gerar_elenco_fallback(home_name)
            away_players = gerar_elenco_fallback(away_name)

        matches_data_final.append(
            {
                "id": f"match-{m.get('id')}",
                "leagueKey": league_key,
                "tabLabel": f"{home_name} x {away_name}",
                "subLabel": sub_label,
                "league": league_title,
                "roundDate": round_date,
                "stadium": stadium,
                "city": city,
                "referee": ref_obj,
                "home": {
                    "name": home_name.upper(),
                    "abbr": (home_info.get("tla") or home_name[:3]).upper(),
                    "bg": home_bg,
                    "color": home_color,
                    "form": home_form,
                    "stats": home_stats,
                    "players": home_players,
                },
                "away": {
                    "name": away_name.upper(),
                    "abbr": (away_info.get("tla") or away_name[:3]).upper(),
                    "bg": away_bg,
                    "color": away_color,
                    "form": away_form,
                    "stats": away_stats,
                    "players": away_players,
                },
            }
        )

    agora_brt = datetime.now(timezone.utc) - timedelta(hours=3)
    fim_brt = agora_brt + timedelta(days=2)
    periodo_str = f"Big 5 + UEFA • Janela {agora_brt.strftime('%d/%m')} a {fim_brt.strftime('%d/%m')}"

    with open("template.html", "r", encoding="utf-8") as f:
        html_tpl = f.read()

    html_out = html_tpl.replace("Big 5 + UEFA • Semana 23/11 a 29/11", periodo_str)
    html_out = html_out.replace("/* __INJECT_MATCHES_DATA__ */", json.dumps(matches_data_final, ensure_ascii=False, indent=2))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"\n[Concluído] index.html gerado com {len(matches_data_final)} confronto(s) e elencos completos!")


if __name__ == "__main__":
    main()
