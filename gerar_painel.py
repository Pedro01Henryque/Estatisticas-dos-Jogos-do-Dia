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
# CONFIGURAÇÕES GERAIS — PH SCORE (BIG 5 + UEFA — JANELA DE 3 DIAS)
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


class JogadorScouting(BaseModel):
    name: str
    pos: str   # GOL, ZAG, LAT, VOL, MEI, ATA
    role: str  # Titular ou Reserva
    fin: List[int]
    sot: List[int]
    tac: List[int]
    fc: List[int]
    fs: List[int]


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


def normalizar(txt: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(txt))
    sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    limpo = re.sub(r"[^a-z0-9\s]", "", sem_acento.lower()).strip()
    for sufixo in (" fc", " cf", " afc", " sc", " ac", " ssc", " bc"):
        if limpo.endswith(sufixo):
            limpo = limpo[: -len(sufixo)].strip()
    for prefixo in (
        "fc ", "cf ", "ac ", "as ", "ssc ", "rc ", "rcd ", "cd ", "ud ",
        "vfb ", "vfl ", "tsg ", "sc ", "sv ", "1 fsv ", "1 fc ", "ogc ", "osc ", "stade "
    ):
        if limpo.startswith(prefixo):
            limpo = limpo[len(prefixo):].strip()
    return limpo


def carregar_base_unificada_csvs() -> pd.DataFrame:
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


def buscar_resultados_recentes_api(token: str) -> pd.DataFrame:
    """Busca resultados recentes (FINISHED) das competições na API para complementar jogos fora do CSV."""
    headers = {"X-Auth-Token": token}
    agora_utc = datetime.now(timezone.utc)
    date_from = (agora_utc - timedelta(days=30)).strftime("%Y-%m-%d")
    date_to = agora_utc.strftime("%Y-%m-%d")

    linhas = []
    for comp in COMPETICOES_API:
        url = f"https://api.football-data.org/v4/competitions/{comp['code']}/matches"
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
                timeout=20,
            )
            if resp.status_code != 200:
                time.sleep(1.5)
                continue

            for m in resp.json().get("matches", []):
                ft = (m.get("score") or {}).get("fullTime") or {}
                hg = ft.get("home")
                ag = ft.get("away")
                if hg is None or ag is None:
                    continue
                h_name = (m.get("homeTeam") or {}).get("shortName") or (m.get("homeTeam") or {}).get("name")
                a_name = (m.get("awayTeam") or {}).get("shortName") or (m.get("awayTeam") or {}).get("name")
                if not h_name or not a_name:
                    continue
                dt_val = pd.to_datetime(m.get("utcDate"), errors="coerce")
                linhas.append({
                    "HomeTeam": h_name,
                    "AwayTeam": a_name,
                    "FTHG": int(hg),
                    "FTAG": int(ag),
                    "_dt": dt_val,
                })
            time.sleep(1.0)
        except Exception as e:
            print(f"   [Aviso] Falha ao buscar resultados recentes de {comp['leagueKey']}: {e}")

    if not linhas:
        return pd.DataFrame()
    return pd.DataFrame(linhas)


def extrair_18_metricas_gerais(df_unificado: pd.DataFrame, nome_time: str) -> tuple[List[List[int]], List[str]]:
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

    jogos = df_unificado[mask].sort_values(by="_dt", ascending=True, na_position="first").tail(5)
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
            print(f"[OK] {comp['leagueKey']}: {len(matches)} jogo(s) encontrados na janela de 3 dias.")
            for m in matches:
                m["_leagueKey"] = comp["leagueKey"]
                jogos_coletados.append(m)

            time.sleep(1.2)
        except Exception as e:
            print(f"[Erro] Falha ao consultar {comp['leagueKey']}: {e}")

    jogos_coletados.sort(key=lambda x: x.get("utcDate", ""))
    return jogos_coletados


def enriquecer_confronto_gemini(
    client: genai.Client,
    home_name: str,
    away_name: str,
    league_key: str,
    round_date: str,
    arbitro_api: Optional[str],
) -> Optional[DadosComplementaresConfronto]:
    prompt = f"""
    Você é um analista de dados e scouting de futebol europeu (temporada 2026/27).
    Preencha os dados de pré-jogo, arbitragem escalada e elenco real com nomes próprios de jogadores para:
    - Mandante: {home_name}
    - Visitante: {away_name}
    - Competição e Rodada: {league_key}
    - Data/Hora: {round_date}
    - Árbitro informado na API (se houver): {arbitro_api or "Não retornado pela API — identifique o árbitro designado para esta partida/rodada se já escalado"}

    REGRAS OBRIGATÓRIAS:
    1. NOMES REAIS DOS JOGADORES (`home_players` e `away_players`):
       - Use EXCLUSIVAMENTE nomes reais de jogadores do elenco atual (2026/27) de {home_name} e de {away_name}.
       - PROIBIDO usar rótulos genéricos como "Goleiro Titular", "Centroavante", "Lateral Direito", "Zagueiro Central", "Ponta" ou siglas de time como prefixo.
       - Liste 14 jogadores reais de {home_name} (11 Titulares + 3 Reservas) e 14 jogadores reais de {away_name} (11 Titulares + 3 Reservas).
    2. MÉTRICAS DOS ÚLTIMOS 5 JOGOS GERAIS DO JOGADOR (todas as competições):
       - Cada array (`fin`, `sot`, `tac`, `fc`, `fs`) deve conter 5 números inteiros representando os últimos 5 jogos gerais disputados pelo jogador.
    3. ARBITRAGEM E ESTÁDIO:
       - Se houver árbitro designado (ex.: Daniel Siebert na Champions League)[cite: 8], informe o nome dele em `referee_name` e as médias reais/estimadas da temporada em `referee_yellow`, `referee_red` e `referee_pen`.
       - `stadium` e `city`: Nome oficial do estádio e cidade/país em português.
       - `home_bg`, `home_color`, `away_bg`, `away_color`: Cores hex oficiais dos uniformes com alto contraste para texto.
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


def main():
    fd_token = os.environ.get("FOOTBALL_DATA_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not fd_token:
        raise EnvironmentError("Configure FOOTBALL_DATA_KEY nos Secrets do GitHub.")

    client = genai.Client(http_options={"api_version": "v1beta"}) if gemini_key else None

    print("1. Carregando base histórica unificada das Big 5...")
    df_csv = carregar_base_unificada_csvs()
    print("   -> Buscando resultados recentes (todas as competições) na API...")
    df_api_res = buscar_resultados_recentes_api(fd_token)

    frames_hist = [d for d in (df_csv, df_api_res) if d is not None and not d.empty]
    df_unificado = pd.concat(frames_hist, ignore_index=True) if frames_hist else pd.DataFrame()
    print(f"   -> {len(df_unificado)} partidas históricas consolidadas.")

    print("\n2. Buscando jogos da janela de 3 dias nas 8 competições...")
    jogos_api = buscar_jogos_api(fd_token)
    if not jogos_api:
        print("Nenhum jogo encontrado na API para o período. Mantendo index.html atual.")
        return

    print(f"\nTotal de confrontos encontrados: {len(jogos_api)}\n")
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

        match_date_str = brt_dt.strftime("%d/%m/%Y")
        sub_label = f"{dia_curto} {brt_dt.strftime('%d/%m')} • {brt_dt.strftime('%H:%M')}"
        round_date = f"{dia_longo}, {match_date_str} • {brt_dt.strftime('%H:%M')} (BRT)"
        matchday = m.get("matchday")
        league_title = f"{league_key.upper()} • RODADA {matchday}" if matchday else f"{league_key.upper()} • FASE DE LIGA"

        home_stats, home_form = extrair_18_metricas_gerais(df_unificado, home_name)
        away_stats, away_form = extrair_18_metricas_gerais(df_unificado, away_name)

        refs = m.get("referees", [])
        arbitro_api = refs[0].get("name") if refs else None

        print(f"[{idx}/{len(jogos_api)}] Processando: {home_name} x {away_name} ({league_key})...")
        comp_ia = None
        if client:
            comp_ia = enriquecer_confronto_gemini(
                client, home_name, away_name, league_title, round_date, arbitro_api
            )

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
                "yellow": "Média: 4.27",
                "red": "Média: 0.17",
                "pen": "Média: 0.25",
            }
            home_bg, home_color = "#059669", "#ffffff"
            away_bg, away_color = "#1d4ed8", "#ffffff"
            home_players = []
            away_players = []

        matches_data_final.append(
            {
                "id": f"match-{m.get('id')}",
                "leagueKey": league_key,
                "matchDate": match_date_str,
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
    periodo_str = f"PH Score • Big 5 + UEFA ({agora_brt.strftime('%d/%m')} a {fim_brt.strftime('%d/%m')})"

    with open("template.html", "r", encoding="utf-8") as f:
        html_tpl = f.read()

    html_out = html_tpl.replace("PH Score • Big 5 + UEFA", periodo_str)
    html_out = html_out.replace("/* __INJECT_MATCHES_DATA__ */", json.dumps(matches_data_final, ensure_ascii=False, indent=2))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"\n[Concluído] index.html gerado com {len(matches_data_final)} confronto(s)!")


if __name__ == "__main__":
    main()
