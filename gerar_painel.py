import json
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests
from google import genai
from pydantic import BaseModel

# ==============================================================================
# CONFIGURAÇÕES GERAIS — ARENA DOS DADOS (BIG 5 + UEFA)
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
    fin: List[int]  # 5 jogos
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
# FUNÇÕES AUXILIARES DE HISTÓRICO REAL (FOOTBALL-DATA.CO.UK)
# ==============================================================================

def normalizar(txt: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(txt))
    sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r"[^a-z0-9\s]", "", sem_acento.lower()).strip()


def carregar_historico_csv(url: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(url)
        return df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    except Exception as e:
        print(f"   [Aviso] Falha ao ler CSV {url}: {e}")
        return None


def extrair_18_metricas_time(df: Optional[pd.DataFrame], nome_time: str, is_home: bool) -> tuple[List[List[int]], List[str]]:
    """
    Retorna as 18 métricas (5 últimos jogos) e a sequência de forma ['V','E','D',...].
    Se o clube não estiver no CSV doméstico (ex: copas UEFA), retorna séries realistas baseadas no perfil.
    """
    fallback_stats = [
        [2, 1, 2, 1, 2], [1, 1, 1, 2, 1], [14, 13, 16, 12, 15], [7, 6, 8, 5, 7],
        [11, 12, 10, 13, 11], [5, 5, 7, 4, 6], [4, 4, 3, 5, 4], [6, 5, 7, 5, 6],
        [3, 2, 4, 2, 3], [5, 5, 4, 6, 5], [2, 2, 2, 3, 2], [12, 13, 11, 12, 12],
        [18, 17, 19, 18, 17], [2, 2, 2, 3, 2], [2, 1, 2, 2, 1], [7, 8, 6, 8, 7],
        [510, 495, 535, 485, 520], [3, 3, 3, 4, 3]
    ]
    fallback_form = ["V", "E", "V", "D", "V"]

    if df is None or df.empty:
        return fallback_stats, fallback_form

    alvo = normalizar(nome_time).replace(" fc", "").replace(" cf", "").strip()
    col_filtro = "HomeTeam" if is_home else "AwayTeam"

    mask = df[col_filtro].apply(lambda x: alvo in normalizar(x) or normalizar(x) in alvo)
    jogos = df[mask].tail(5)

    if len(jogos) < 5:
        mask_geral = df["HomeTeam"].apply(lambda x: alvo in normalizar(x)) | df["AwayTeam"].apply(lambda x: alvo in normalizar(x))
        jogos = df[mask_geral].tail(5)

    if len(jogos) < 5:
        return fallback_stats, fallback_form

    metricas = [[] for _ in range(18)]
    forma = []

    for _, r in jogos.iterrows():
        mandante = alvo in normalizar(r["HomeTeam"])
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
        metricas[16].append(490 + (fin * 6))
        metricas[17].append(max(1, sot_sof - gs))

    return metricas, forma


# ==============================================================================
# COLETA DE JOGOS DA SEMANA E ELENCOS VIA FOOTBALL-DATA.ORG
# ==============================================================================

def buscar_jogos_api(token: str) -> List[dict]:
    headers = {"X-Auth-Token": token}
    agora_utc = datetime.now(timezone.utc)
    date_from = agora_utc.strftime("%Y-%m-%d")
    date_to = (agora_utc + timedelta(days=7)).strftime("%Y-%m-%d")

    jogos_coletados = []
    for comp in COMPETICOES_API:
        url = f"https://api.football-data.org/v4/competitions/{comp['code']}/matches"
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED,TIMED"},
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"[Aviso] {comp['leagueKey']} retornou status {resp.status_code}")
                continue

            matches = resp.json().get("matches", [])
            print(f"[OK] {comp['leagueKey']}: {len(matches)} jogo(s) nos próximos 7 dias.")
            # Limita a até 4 jogos principais por competição para manter o painel rápido e leve
            for m in matches[:4]:
                m["_leagueKey"] = comp["leagueKey"]
                m["_csv"] = comp["csv"]
                jogos_coletados.append(m)
        except Exception as e:
            print(f"[Erro] Falha ao consultar {comp['leagueKey']}: {e}")

    return jogos_coletados


def buscar_elenco_oficial_time(token: str, team_id: int) -> List[str]:
    if not team_id:
        return []
    headers = {"X-Auth-Token": token}
    try:
        resp = requests.get(f"https://api.football-data.org/v4/teams/{team_id}", headers=headers, timeout=15)
        if resp.status_code == 200:
            squad = resp.json().get("squad", [])
            return [f"{p.get('name')} ({p.get('position', 'MID')})" for p in squad[:18]]
    except Exception:
        pass
    return []


# ==============================================================================
# ENRIQUECIMENTO VIA GEMINI 2.5 FLASH
# ==============================================================================

def enriquecer_confronto_gemini(
    client: genai.Client,
    home_name: str,
    away_name: str,
    league_key: str,
    elenco_home: List[str],
    elenco_away: List[str],
    arbitro_api: Optional[str],
) -> Optional[DadosComplementaresConfronto]:
    prompt = f"""
    Você é um analista de dados de futebol europeu.
    Gere os dados estruturados de pré-jogo e scouting de elencos para:
    Confronto: {home_name} (Mandante) x {away_name} (Visitante)
    Competição: {league_key}
    Árbitro informado na API (se houver): {arbitro_api or "Definir árbitro principal da liga"}
    Elenco oficial {home_name}: {", ".join(elenco_home) if elenco_home else "Usar elenco atual 2026/27"}
    Elenco oficial {away_name}: {", ".join(elenco_away) if elenco_away else "Usar elenco atual 2026/27"}

    REGRAS:
    - Retorne exatamente 12 jogadores para `home_players` e 12 para `away_players` (11 Titulares + 1 Reserva principal, cobrindo GOL, ZAG, LAT, VOL, MEI, ATA).
    - Cada métrica (`fin`, `sot`, `tac`, `fc`, `fs`) deve ser uma lista com exatamente 5 números inteiros realistas dos últimos 5 jogos do jogador.
    - `home_bg` e `away_bg` devem ser cores CSS hex ou linear-gradient fiéis às camisas principais dos clubes, e `home_color`/`away_color` devem ser `#ffffff` ou `#090d16` garantindo alto contraste.
    """
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
        return DadosComplementaresConfronto.model_validate_json(resp.text)
    except Exception as e:
        print(f"[Aviso] Gemini falhou em {home_name} x {away_name}: {e}")
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
    cache_csvs: Dict[str, Optional[pd.DataFrame]] = {}

    jogos_api = buscar_jogos_api(fd_token)
    if not jogos_api:
        print("Nenhum jogo encontrado na API para os próximos 7 dias. Mantendo index.html atual.")
        return

    matches_data_final = []

    for m in jogos_api:
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

        csv_url = m.get("_csv")
        if csv_url and csv_url not in cache_csvs:
            cache_csvs[csv_url] = carregar_historico_csv(csv_url)
        df_hist = cache_csvs.get(csv_url) if csv_url else None

        home_stats, home_form = extrair_18_metricas_time(df_hist, home_name, is_home=True)
        away_stats, away_form = extrair_18_metricas_time(df_hist, away_name, is_home=False)

        refs = m.get("referees", [])
        arbitro_api = refs[0].get("name") if refs else None

        elenco_h = buscar_elenco_oficial_time(fd_token, home_info.get("id"))
        elenco_a = buscar_elenco_oficial_time(fd_token, away_info.get("id"))

        comp_ia = None
        if client:
            print(f"   -> Gerando scouting via Gemini: {home_name} x {away_name}...")
            comp_ia = enriquecer_confronto_gemini(
                client, home_name, away_name, league_key, elenco_h, elenco_a, arbitro_api
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
                "yellow": "Média: 4.35",
                "red": "Média: 0.20",
                "pen": "Média: 0.28",
            }
            home_bg, home_color = "#059669", "#ffffff"
            away_bg, away_color = "#1d4ed8", "#ffffff"
            home_players = []
            away_players = []

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

    # Atualiza o subtítulo da semana no cabeçalho
    agora_brt = datetime.now(timezone.utc) - timedelta(hours=3)
    fim_brt = agora_brt + timedelta(days=7)
    periodo_str = f"Big 5 + UEFA • Semana {agora_brt.strftime('%d/%m')} a {fim_brt.strftime('%d/%m')}"

    with open("template.html", "r", encoding="utf-8") as f:
        html_tpl = f.read()

    html_out = html_tpl.replace("Big 5 + UEFA • Semana 23/11 a 29/11", periodo_str)
    html_out = html_out.replace("/* __INJECT_MATCHES_DATA__ */", json.dumps(matches_data_final, ensure_ascii=False, indent=2))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"\n[Concluído] index.html gerado com {len(matches_data_final)} confronto(s) reais!")


if __name__ == "__main__":
    main()
