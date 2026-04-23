#!/usr/bin/env python3
"""
pncp_radar.py — Gera mapa interativo de licitações prováveis

Lê pncp_similaridades.json e gera radar_licitacoes.html:
- Mapa do Brasil com bolinhas por município
- Cor = grau de aderência (azul escuro = maior score)
- Tamanho = valor estimado da licitação
- Clique = painel lateral com detalhes e link para o portal

Requer apenas um navegador para visualizar (sem instalação).

Uso:
  python pncp_radar.py                          # gera radar_licitacoes.html
  python pncp_radar.py --input outro.json       # fonte alternativa
  python pncp_radar.py --out mapa.html          # nome alternativo
  python pncp_radar.py --threshold 0.20         # exibe só acima do threshold
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuração ──────────────────────────────────────────────────────────────

INPUT_FILE   = Path("pncp_similaridades.json")
OUTPUT_FILE  = Path("radar_licitacoes.html")
COORDS_CACHE = Path("municipios_coords.json")

THRESHOLD_EXIBIR = 0.10   # abaixo disso, bolinha não aparece
PAUSA_GEOCODE    = 1.2    # segundos entre chamadas Nominatim (respeitar ToS)

# Centroides dos estados — fallback quando município não é geocodificado
_CENTROIDES_UF: dict[str, tuple[float, float]] = {
    "AC": (-9.02, -70.81), "AL": (-9.57, -36.78), "AM": (-3.47, -65.10),
    "AP": (1.41,  -51.77), "BA": (-12.96, -41.71), "CE": (-5.20, -39.53),
    "DF": (-15.78, -47.93), "ES": (-19.19, -40.34), "GO": (-15.98, -49.86),
    "MA": (-5.42, -45.44), "MG": (-18.51, -44.55), "MS": (-20.77, -54.79),
    "MT": (-12.64, -55.42), "PA": (-3.79, -52.48), "PB": (-7.28, -36.72),
    "PE": (-8.38, -37.86), "PI": (-7.72, -42.73), "PR": (-24.89, -51.55),
    "RJ": (-22.25, -42.66), "RN": (-5.81, -36.59), "RO": (-10.83, -63.34),
    "RR": (2.05,  -61.40), "RS": (-30.17, -53.50), "SC": (-27.45, -50.95),
    "SE": (-10.57, -37.38), "SP": (-22.25, -48.60), "TO": (-10.18, -48.33),
}


# ── Geocodificação ────────────────────────────────────────────────────────────

def _carregar_cache_coords() -> dict[str, dict]:
    if COORDS_CACHE.exists():
        try:
            with open(COORDS_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _salvar_cache_coords(cache: dict) -> None:
    with open(COORDS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _geocode_nominatim(municipio: str, uf: str) -> tuple[float, float] | None:
    """Consulta Nominatim (OpenStreetMap) para obter lat/lon do município."""
    query = quote(f"{municipio}, {uf}, Brasil")
    url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={query}&format=json&limit=1&countrycodes=br"
    )
    try:
        req = Request(url, headers={"User-Agent": "ResgatecnicaRadar/1.0 (licitacoes)"})
        with urlopen(req, timeout=10) as resp:
            results = json.loads(resp.read().decode("utf-8"))
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except (URLError, Exception):
        pass
    return None


def geocodificar_municipios(
    licitacoes: list[dict],
    cache: dict,
) -> dict[str, dict]:
    """
    Para cada município único nas licitações, obtém lat/lon.
    Usa cache primeiro, Nominatim como fallback, centroide do estado como último recurso.
    """
    municipios_unicos: set[tuple[str, str]] = set()
    for lic in licitacoes:
        mun = (lic.get("municipio", ""), lic.get("uf", ""))
        if mun[0]:
            municipios_unicos.add(mun)

    novos = [(m, u) for m, u in municipios_unicos if f"{m}_{u}" not in cache]
    if novos:
        print(f"\nGeocodificando {len(novos)} municípios novos via Nominatim...")
        print("  (isso é feito uma única vez — resultado salvo em cache)\n")

    for municipio, uf in novos:
        chave = f"{municipio}_{uf}"
        coords = _geocode_nominatim(municipio, uf)
        if coords:
            lat, lon = coords
            cache[chave] = {"lat": lat, "lon": lon, "fonte": "nominatim"}
            print(f"  OK  {municipio}/{uf}: {lat:.4f}, {lon:.4f}")
        else:
            lat, lon = _CENTROIDES_UF.get(uf, (-15.8, -47.9))
            cache[chave] = {"lat": lat, "lon": lon, "fonte": "centroide_uf"}
            print(f"  [~] {municipio}/{uf}: usando centroide do estado")

        _salvar_cache_coords(cache)
        if novos.index((municipio, uf)) < len(novos) - 1:
            time.sleep(PAUSA_GEOCODE)

    return cache


# ── Escala de cores ───────────────────────────────────────────────────────────

def _cor_score(score: float) -> str:
    """Score [0,1] → cor na escala cinza→âmbar→laranja→vermelho (frio a quente)."""
    if score >= 0.50:
        return "#B71C1C"   # vermelho escuro  — muito provável
    if score >= 0.40:
        return "#E53935"   # vermelho         — provável
    if score >= 0.30:
        return "#F57C00"   # laranja escuro   — moderado
    if score >= 0.20:
        return "#FFB300"   # âmbar            — baixo
    if score >= 0.15:
        return "#FFE082"   # âmbar claro      — muito baixo
    return "#BDBDBD"       # cinza            — limiar


def _raio_valor(valor: float) -> int:
    """Valor estimado → raio da bolinha em pixels."""
    if valor <= 0:
        return 8
    import math
    # log scale: R$10k → 6px, R$100k → 9px, R$1M → 13px, R$10M → 17px, R$100M → 21px
    raio = int(4 + 3 * math.log10(max(valor / 10_000, 1)))
    return max(6, min(raio, 28))


def _formatar_brl(valor: float) -> str:
    if valor <= 0:
        return "Não informado"
    return f"R$ {valor:_.2f}".replace("_", ".")


def _formatar_data(s: str) -> str:
    if not s or len(s) < 10:
        return "—"
    d, m, a = s[8:10], s[5:7], s[:4]
    return f"{d}/{m}/{a}"


# ── Geração do HTML ───────────────────────────────────────────────────────────

_LEGENDA_HTML = """
<div id="legenda">
  <b>Score de aderência</b><br>
  <span class="dot" style="background:#B71C1C"></span> ≥ 0.50 — muito provável<br>
  <span class="dot" style="background:#E53935"></span> ≥ 0.40 — provável<br>
  <span class="dot" style="background:#F57C00"></span> ≥ 0.30 — moderado<br>
  <span class="dot" style="background:#FFB300"></span> ≥ 0.20 — baixo<br>
  <span class="dot" style="background:#FFE082"></span> ≥ 0.15 — muito baixo<br>
  <span class="dot" style="background:#BDBDBD"></span> &lt; 0.15 — limiar<br>
  <hr style="margin:6px 0">
  <b>Tamanho</b> = valor estimado
</div>
"""

_CSS = """
  body { margin: 0; font-family: 'Segoe UI', Arial, sans-serif; }
  #map { position: absolute; top: 0; bottom: 0; width: 100%; }

  #painel {
    position: absolute; top: 0; left: 0; bottom: 0; width: 320px;
    background: white; z-index: 1000; overflow-y: auto;
    box-shadow: 2px 0 12px rgba(0,0,0,0.18);
    display: none; padding: 0;
  }
  #painel-conteudo { padding: 16px; }
  #painel-fechar {
    position: sticky; top: 0; background: #1565C0; color: white;
    padding: 10px 14px; font-size: 13px; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
  }
  #painel-fechar:hover { background: #0D47A1; }
  .painel-titulo { font-size: 15px; font-weight: 700; color: #1565C0; margin-bottom: 4px; }
  .painel-orgao  { font-size: 12px; color: #555; margin-bottom: 12px; }
  .painel-campo  { display: flex; gap: 8px; align-items: flex-start;
                   margin-bottom: 8px; font-size: 13px; }
  .painel-label  { color: #888; min-width: 90px; }
  .painel-valor  { color: #222; font-weight: 500; flex: 1; }
  .score-badge   {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    color: white; font-weight: 700; font-size: 13px; margin-bottom: 12px;
  }
  .itens-titulo  { font-size: 12px; font-weight: 700; color: #444;
                   margin: 14px 0 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  .item-row      { background: #F5F9FF; border-left: 3px solid #1976D2;
                   padding: 6px 10px; margin-bottom: 5px; border-radius: 3px;
                   font-size: 12px; }
  .item-score    { font-weight: 700; color: #1565C0; }
  .item-desc     { color: #333; margin-top: 2px; }
  .item-ref      { color: #888; font-style: italic; margin-top: 1px; font-size: 11px; }
  .btn-link      {
    display: block; margin-top: 14px; text-align: center;
    background: #1565C0; color: white; padding: 9px 14px;
    border-radius: 6px; text-decoration: none; font-size: 13px;
    font-weight: 600; transition: background 0.2s;
  }
  .btn-link:hover { background: #0D47A1; }

  #legenda {
    position: absolute; bottom: 30px; right: 10px; z-index: 1000;
    background: white; padding: 10px 14px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 12px; line-height: 1.7;
  }
  .dot {
    display: inline-block; width: 12px; height: 12px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
  }

  #contador {
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: white; padding: 8px 14px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15); font-size: 13px; font-weight: 600;
    color: #1565C0;
  }

  #filtro-bar {
    position: absolute; top: 10px; left: 10px; z-index: 1000;
    background: white; padding: 8px 14px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15); font-size: 12px;
    display: flex; align-items: center; gap: 10px;
  }
  #filtro-bar label { color: #555; }
  #filtro-threshold { width: 60px; }
"""

def _gerar_js(dados_json: str) -> str:
    return (
        "var dados = " + dados_json + ";\n\n"
        "var map = L.map('map').setView([-18.5, -44.5], 7);\n\n"
        "L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {\n"
        "  attribution: '\u00a9 <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> \u00a9 <a href=\"https://carto.com/\">CARTO</a>',\n"
        "  subdomains: 'abcd',\n"
        "  maxZoom: 19\n"
        "}).addTo(map);\n\n"
        "var markers = [];\n"
        "var camadaMarkers = L.layerGroup().addTo(map);\n\n"
        "function corScore(s) {\n"
        "  if (s >= 0.50) return '#B71C1C';\n"
        "  if (s >= 0.40) return '#E53935';\n"
        "  if (s >= 0.30) return '#F57C00';\n"
        "  if (s >= 0.20) return '#FFB300';\n"
        "  if (s >= 0.15) return '#FFE082';\n"
        "  return '#BDBDBD';\n"
        "}\n\n"
        "function raioValor(v) {\n"
        "  if (!v || v <= 0) return 8;\n"
        "  return Math.max(6, Math.min(4 + Math.round(3 * Math.log10(Math.max(v / 10000, 1))), 28));\n"
        "}\n\n"
        "function fmtBRL(v) {\n"
        "  if (!v || v <= 0) return 'N\u00e3o informado';\n"
        "  return 'R$ ' + v.toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, '.');\n"
        "}\n\n"
        "function fmtData(s) {\n"
        "  if (!s || s.length < 10) return '\u2014';\n"
        "  return s.substring(8,10) + '/' + s.substring(5,7) + '/' + s.substring(0,4);\n"
        "}\n\n"
        "function abrirPainel(lic) {\n"
        "  var cor = corScore(lic.score_max);\n"
        "  var html = '<div id=\"painel-fechar\" onclick=\"fecharPainel()\"><span>Detalhes da licita\u00e7\u00e3o</span><span>\u2715</span></div>';\n"
        "  html += '<div id=\"painel-conteudo\">';\n"
        "  html += '<p class=\"painel-titulo\">' + (lic.objeto || '\u2014') + '</p>';\n"
        "  html += '<p class=\"painel-orgao\">' + (lic.orgao || '') + '</p>';\n"
        "  html += '<span class=\"score-badge\" style=\"background:' + cor + '\">Score ' + lic.score_max.toFixed(3) + '</span>';\n"
        "  html += '<div class=\"painel-campo\"><span class=\"painel-label\">Munic\u00edpio</span><span class=\"painel-valor\">' + (lic.municipio || '\u2014') + ' / ' + (lic.uf || '') + '</span></div>';\n"
        "  html += '<div class=\"painel-campo\"><span class=\"painel-label\">Valor est.</span><span class=\"painel-valor\">' + fmtBRL(lic.valor_estimado) + '</span></div>';\n"
        "  html += '<div class=\"painel-campo\"><span class=\"painel-label\">Encerramento</span><span class=\"painel-valor\">' + fmtData(lic.data_encerramento) + '</span></div>';\n"
        "  html += '<div class=\"painel-campo\"><span class=\"painel-label\">Processo</span><span class=\"painel-valor\" style=\"font-size:11px\">' + (lic.processo || '\u2014') + '</span></div>';\n"
        "  if (lic.top_itens && lic.top_itens.length > 0) {\n"
        "    html += '<div class=\"itens-titulo\">Itens mais similares ao portf\u00f3lio</div>';\n"
        "    lic.top_itens.forEach(function(item) {\n"
        "      html += '<div class=\"item-row\">';\n"
        "      html += '<div class=\"item-score\">Score: ' + item.score.toFixed(3) + '</div>';\n"
        "      html += '<div class=\"item-desc\">' + (item.descricao || '') + '</div>';\n"
        "      if (item.produto_ref) html += '<div class=\"item-ref\">\u223c ' + item.produto_ref + '</div>';\n"
        "      html += '</div>';\n"
        "    });\n"
        "  }\n"
        "  if (lic.link) {\n"
        "    html += '<a class=\"btn-link\" href=\"' + lic.link + '\" target=\"_blank\">Ver no portal \u2197</a>';\n"
        "  }\n"
        "  html += '</div>';\n"
        "  document.getElementById('painel').innerHTML = html;\n"
        "  document.getElementById('painel').style.display = 'block';\n"
        "}\n\n"
        "function fecharPainel() {\n"
        "  document.getElementById('painel').style.display = 'none';\n"
        "}\n\n"
        "function renderizar(threshold) {\n"
        "  camadaMarkers.clearLayers();\n"
        "  markers = [];\n"
        "  var visiveis = 0;\n"
        "  dados.forEach(function(lic) {\n"
        "    if (!lic.lat || !lic.lon) return;\n"
        "    if (lic.score_max < threshold) return;\n"
        "    visiveis++;\n"
        "    var circulo = L.circleMarker([lic.lat, lic.lon], {\n"
        "      radius: raioValor(lic.valor_estimado),\n"
        "      fillColor: corScore(lic.score_max),\n"
        "      color: '#fff',\n"
        "      weight: 1.5,\n"
        "      opacity: 0.9,\n"
        "      fillOpacity: 0.85\n"
        "    });\n"
        "    circulo.on('click', function() { abrirPainel(lic); });\n"
        "    circulo.bindTooltip(\n"
        "      '<b>' + (lic.municipio || '') + '</b><br>' +\n"
        "      'Score: ' + lic.score_max.toFixed(3) + '<br>' +\n"
        "      fmtBRL(lic.valor_estimado),\n"
        "      {sticky: true}\n"
        "    );\n"
        "    camadaMarkers.addLayer(circulo);\n"
        "    markers.push(circulo);\n"
        "  });\n"
        "  document.getElementById('contador').textContent = visiveis + ' licita\u00e7\u00f5es';\n"
        "}\n\n"
        "var sliderEl = document.getElementById('filtro-threshold');\n"
        "var labelEl  = document.getElementById('threshold-label');\n"
        "sliderEl.addEventListener('input', function() {\n"
        "  labelEl.textContent = parseFloat(this.value).toFixed(2);\n"
        "  renderizar(parseFloat(this.value));\n"
        "});\n\n"
        "renderizar(parseFloat(sliderEl.value));\n"
    )


def gerar_html(
    licitacoes: list[dict],
    coords: dict,
    threshold: float,
    meta: dict,
) -> str:
    # Adicionar lat/lon aos dados
    dados_mapa = []
    for lic in licitacoes:
        if lic["score_max"] < threshold:
            continue
        chave = f"{lic['municipio']}_{lic['uf']}"
        coord = coords.get(chave)
        if coord:
            lic = dict(lic)
            lic["lat"] = coord["lat"]
            lic["lon"] = coord["lon"]
        else:
            lat, lon = _CENTROIDES_UF.get(lic.get("uf", ""), (-15.8, -47.9))
            lic = dict(lic)
            lic["lat"] = lat
            lic["lon"] = lon
        dados_mapa.append(lic)

    dados_json = json.dumps(dados_mapa, ensure_ascii=False)
    js = _gerar_js(dados_json)

    gerado_em = meta.get("gerado_em", "")[:16].replace("T", " ")
    backend   = meta.get("backend", "?")
    modelo    = meta.get("modelo", "?")
    threshold_str = f"{threshold:.2f}"

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Radar de Licitações — Resgatécnica</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>{_CSS}</style>
</head>
<body>
  <div id="map"></div>
  <div id="painel"></div>

  <div id="filtro-bar">
    <label>Score mínimo:</label>
    <input type="range" id="filtro-threshold"
           min="0.05" max="0.60" step="0.05" value="{threshold_str}">
    <span id="threshold-label">{threshold_str}</span>
    <span style="color:#aaa;font-size:11px">| {gerado_em} | {backend}</span>
  </div>

  <div id="contador">— licitações</div>

  {_LEGENDA_HTML}

  <script>
{js}
  </script>
</body>
</html>"""


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Gera mapa interativo de licitações")
    p.add_argument("--input",     default=str(INPUT_FILE))
    p.add_argument("--out",       default=str(OUTPUT_FILE))
    p.add_argument("--threshold", type=float, default=THRESHOLD_EXIBIR,
                   help="Score mínimo para exibir no mapa (default: 0.10)")
    p.add_argument("--sem-geocode", action="store_true",
                   help="Não buscar coordenadas novas (usa só cache)")
    args = p.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.out)

    if not input_path.exists():
        print(f"[ERRO] {input_path} não encontrado.")
        print("  Execute pncp_similaridade.py primeiro.")
        sys.exit(1)

    print(f"Carregando {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        dados = json.load(f)

    meta = dados.get("_meta", {})
    licitacoes = dados.get("licitacoes", dados) if isinstance(dados, dict) else dados

    print(f"  {len(licitacoes):,} licitações com score >= {meta.get('threshold', '?')}")
    acima = [l for l in licitacoes if l["score_max"] >= args.threshold]
    print(f"  {len(acima):,} com score >= {args.threshold:.2f} (filtro do mapa)")

    # Geocodificar municípios
    cache = _carregar_cache_coords()
    if not args.sem_geocode:
        cache = geocodificar_municipios(licitacoes, cache)

    # Gerar HTML
    print(f"\nGerando {output_path}...")
    html = gerar_html(licitacoes, cache, args.threshold, meta)
    output_path.write_text(html, encoding="utf-8")

    print(f"""
Mapa gerado com sucesso!
  Arquivo : {output_path}
  Licitações no mapa : {len(acima):,}

Abra o arquivo no navegador para visualizar.
""")


if __name__ == "__main__":
    main()
