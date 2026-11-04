"""ui.py - Omnix(TM) Integrix interface theme: Omnix teal brand, light (default) and dark modes.

The CSS is built from CSS variables so every custom block follows the active mode. The mode follows the
Streamlit theme the viewer picks (menu > Settings > Light / Dark), read with st.context.theme.type.
Nothing here computes results. Publication downloads always keep their white, journal-ready style.
"""
from __future__ import annotations

import base64
import functools
import html
import json
import os

# ------------------------------------------------------------------------------------------------ palettes
LIGHT = dict(
    bg="#F4F7F6", panel="#FFFFFF", card="#FFFFFF", card2="#F2F6F5", border="#DCE5E3", ink="#1A2B29",
    muted="#5B6E6A", faint="#8A9B98", brand="#00695C", brand2="#0F8F80", brand_t="#00574D",
    brand_soft="rgba(0,105,92,.07)", brand_line="rgba(0,105,92,.28)", green="#059669", amber="#C26A04",
    purple="#7C3AED", sky="#0284C7", rose="#E11D48", orange="#EA580C", grid="#E8EEED", axis="#C9D3D1",
    shadow="0 1px 2px rgba(16,40,36,.06), 0 4px 14px rgba(16,40,36,.05)",
)
DARK = dict(
    bg="#0A1312", panel="#0E1A19", card="#112120", card2="#162A28", border="#1F3634", ink="#E3EEEC",
    muted="#93AAA6", faint="#647C78", brand="#14B8A6", brand2="#2DD4BF", brand_t="#5EEAD4",
    brand_soft="rgba(20,184,166,.10)", brand_line="rgba(45,212,191,.35)", green="#34D399", amber="#F59E0B",
    purple="#A78BFA", sky="#38BDF8", rose="#FB7185", orange="#FB923C", grid="#1C302E", axis="#2C4542",
    shadow="none",
)
LAYER_COLORS = {"mRNA": "#6366F1", "protein": "#10B981", "metabolite": "#F59E0B"}


def palette(mode: str) -> dict:
    return DARK if mode == "dark" else LIGHT


def css(mode: str = "light") -> str:
    p = palette(mode)
    root = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in p.items())
    # the web font is imported in its own <style> block: if fonts.googleapis.com is blocked or slow (offline or
    # firewalled networks) only that block waits; every Integrix style still applies at once with system fonts
    return """""" + f"""
<style>
:root {{ {root}; }}
html, body, [class*="css"], .stMarkdown, button, input, select, textarea {{
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }}
.stApp {{ background: var(--bg); }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1480px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4, h5 {{ color: var(--ink) !important; font-weight: 650; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.12rem !important; }} h4 {{ font-size: 1.0rem !important; }}
[data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: var(--panel); border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 8px; }}
section[data-testid="stSidebar"] [role="radiogroup"] > label {{
  background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin: 0;
  width: 100%; transition: border-color .15s, background .15s; }}
section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{ border-color: var(--brand-line); }}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
  border-color: var(--brand); background: var(--brand-soft); box-shadow: 0 0 0 1px var(--brand-line) inset; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] details {{
  background: var(--card); border: 1px solid var(--border); border-radius: 10px; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{ font-weight: 550; }}
section[data-testid="stSidebar"] hr {{ border-color: var(--border); margin: 1.1rem 0 .6rem 0; }}

/* buttons */
.stButton > button, .stDownloadButton > button {{ border-radius: 9px; font-weight: 550; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"],
.stDownloadButton > button[kind="primary"] {{
  background: linear-gradient(90deg, #00574D 0%, #00695C 45%, #0F9D8C 100%); border: 0; color: #fff;
  font-weight: 650; box-shadow: 0 6px 18px rgba(0,105,92,.28); }}
.stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); color: #fff; }}

/* navigation: section switcher + tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ height: 42px; padding: 0 12px; color: var(--muted); background: transparent;
  border-radius: 8px 8px 0 0; }}
.stTabs [data-baseweb="tab"] p {{ font-size: .93rem; font-weight: 550; }}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--ink); background: var(--brand-soft); }}
.stTabs [aria-selected="true"] {{ color: var(--brand-t) !important; background: var(--brand-soft) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--brand); height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}
div.tr-nav + div [data-testid="stButtonGroup"] button, .st-key-nav_section button {{ font-weight: 600; }}

/* cards: bordered containers */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"] {{
  background: var(--card); border-color: var(--border) !important; border-radius: 14px; box-shadow: var(--shadow); }}
[data-testid="stExpander"] details {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; }}
/* side-by-side cards wrap onto separate rows on narrow (half-screen) windows */
div[data-testid="stHorizontalBlock"]:not(.st-key-seccards *):not(.st-key-pagebar *):has(> div[data-testid="stColumn"] > div > div[data-testid="stLayoutWrapper"]) {{
  flex-wrap: wrap; row-gap: 1rem; }}
div[data-testid="stHorizontalBlock"]:not(.st-key-seccards *):not(.st-key-pagebar *):has(> div[data-testid="stColumn"] > div > div[data-testid="stLayoutWrapper"]) > div[data-testid="stColumn"] {{
  flex: 1 1 400px !important; min-width: min(400px, 100%); }}

/* header */
.tr-head {{ display:flex; align-items:center; gap:16px; flex-wrap:wrap; padding: 2px 0 14px 0;
  border-bottom: 1px solid var(--border); margin-bottom: 14px; }}
.tr-logo {{ width:50px; height:50px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background: var(--card); border: 1px solid var(--border); box-shadow: var(--shadow); overflow:hidden; }}
.tr-logo img {{ width: 40px; height: 40px; object-fit: contain; }}
.tr-title {{ display:flex; flex-direction:column; line-height:1.2; }}
.tr-title b {{ font-size: 1.45rem; color: var(--ink); font-weight:700; letter-spacing:-.01em; }}
.tr-title b sup {{ font-size: .55em; color: var(--muted); font-weight: 600; }}
.tr-title b .mod {{ color: var(--brand-t); }}
.tr-title > span {{ color: var(--muted); font-size: .92rem; }}
.tr-title b .mod {{ font-size: inherit; margin-left: 2px; }}
.tr-ver {{ font-family:'JetBrains Mono',monospace; font-size:.76rem; color: var(--brand-t); border:1px solid var(--brand-line);
  background: var(--brand-soft); padding: 2px 9px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }}
.tr-pill {{ display:inline-flex; align-items:center; gap:8px; padding: 6px 14px; border-radius:999px;
  font-size:.85rem; font-weight:550; border:1px solid; white-space:nowrap; }}
.tr-pill .dot {{ width:9px; height:9px; border-radius:50%; }}
.tr-pill.ok {{ color: var(--green); border-color: color-mix(in srgb, var(--green) 45%, transparent);
  background: color-mix(in srgb, var(--green) 8%, transparent); }}
.tr-pill.ok .dot {{ background: var(--green); box-shadow: 0 0 8px var(--green); }}
.tr-pill.idle {{ color: var(--muted); border-color: var(--border); background: var(--card); }}
.tr-pill.idle .dot {{ background: var(--faint); }}
.tr-pill.info {{ color: var(--brand-t); border-color: var(--brand-line); background: var(--brand-soft); }}
.tr-pill .mono {{ font-family:'JetBrains Mono',monospace; }}
.tr-spacer {{ flex:1; }}

/* study banner */
.tr-banner {{ display:flex; gap:16px; align-items:flex-start; padding:15px 18px; border-radius:14px;
  border:1px solid var(--brand-line); background: linear-gradient(90deg, var(--brand-soft), transparent);
  margin: 0 0 12px 0; flex-wrap: wrap; }}
.tr-banner .ic {{ width:42px; height:42px; border-radius:10px; background: var(--brand-soft); border:1px solid var(--brand-line);
  display:flex; align-items:center; justify-content:center; color: var(--brand-t); font-size:1.3rem; flex:none; }}
.tr-banner .body {{ flex: 1 1 360px; min-width: 260px; }}
.tr-tag {{ display:inline-block; padding:3px 10px; border-radius:7px; background: var(--brand-soft);
  color: var(--brand-t); font-weight:600; font-size:.88rem; border:1px solid var(--brand-line); margin-right:10px; }}
.tr-contrast {{ font-family:'JetBrains Mono',monospace; font-size:.86rem; color: var(--muted); }}
.tr-contrast .case {{ color: var(--rose); }} .tr-contrast .ref {{ color: var(--green); }}
.tr-banner p {{ margin: 8px 0 0 0; color: var(--ink); opacity:.88; font-size:.92rem; }}

/* section + card titles */
.tr-sec {{ display:flex; align-items:center; gap:9px; color: var(--ink); font-weight:650; letter-spacing:.06em;
  text-transform:uppercase; font-size:.82rem; margin: 12px 0 10px 0; }}
.tr-sec .ic {{ color: var(--brand); font-size: 1rem; }}
.tr-sec .badge {{ margin-left:auto; text-transform:none; letter-spacing:0; font-family:'JetBrains Mono',monospace;
  font-size:.72rem; color: var(--brand-t); border:1px solid var(--brand-line); background: var(--brand-soft);
  padding:1px 8px; border-radius:6px; font-weight:500; }}
.tr-cardhead {{ margin: 2px 0 6px 0; }}
.tr-cardhead b {{ color: var(--ink); font-size:1.05rem; font-weight:650; display:flex; align-items:center; gap:9px; }}
.tr-cardhead b .ic {{ color: var(--brand); }}
.tr-cardhead span {{ color: var(--muted); font-size:.87rem; }}

/* metric cards */
.tr-metrics {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 14px; margin-bottom: 10px; }}
.tr-metric {{ background: var(--card); border:1px solid var(--border); border-radius:14px; padding: 15px 17px 13px 17px;
  border-bottom: 3px solid var(--accent); min-height: 128px; box-shadow: var(--shadow); }}
.tr-metric.hl {{ background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 8%, var(--card)), var(--card)); }}
.tr-metric .lab {{ color: var(--labc, var(--muted)); font-size:.88rem; font-weight:560; display:flex;
  justify-content:space-between; gap:8px; }}
.tr-metric .lab .ic {{ color: var(--accent); font-size:1.05rem; }}
.tr-metric .val {{ font-family:'JetBrains Mono',monospace; font-size: 2.1rem; font-weight:700; color: var(--valc, var(--ink));
  margin: 8px 0 6px 0; line-height:1.1; }}
.tr-metric .sub {{ color: var(--muted); font-size:.8rem; }}
.tr-metric .sub b {{ color: var(--accent); font-weight:600; }}

/* key findings */
.tr-find {{ display:flex; gap:12px; align-items:flex-start; padding: 10px 4px; border-bottom:1px solid var(--border); }}
.tr-find:last-child {{ border-bottom: 0; }}
.tr-find .kind {{ flex:none; font-size:.7rem; font-weight:650; letter-spacing:.04em; text-transform:uppercase;
  padding: 3px 8px; border-radius:6px; border:1px solid; min-width: 96px; text-align:center; margin-top: 2px; }}
.tr-find .txt {{ color: var(--ink); font-size:.91rem; line-height:1.45; }}
.tr-find .txt .stat {{ font-family:'JetBrains Mono',monospace; font-size:.78rem; color: var(--muted); }}
.tr-find .txt .where {{ color: var(--brand-t); font-size:.78rem; white-space: nowrap; }}

.trace-note {{ background: var(--brand-soft); border:1px solid var(--brand-line); border-left: 4px solid var(--brand);
  padding: 10px 14px; border-radius: 10px; margin: 6px 0 12px 0; color: var(--ink); }}
.trace-warn {{ background: color-mix(in srgb, var(--amber) 9%, transparent); border:1px solid color-mix(in srgb, var(--amber) 40%, transparent);
  border-left: 4px solid var(--amber); padding: 10px 14px; border-radius: 10px; margin: 6px 0 12px 0; color: var(--ink); }}
.tr-foot {{ margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--faint);
  font-size: .8rem; display:flex; gap: 16px; flex-wrap: wrap; }}
/* ---- top bar (v1.4.1): brand row + section row */
.block-container {{ padding-top: 2.6rem !important; }}
.st-key-brandbar {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 0 !important;
  box-shadow: var(--shadow); margin-bottom: 6px; overflow: hidden; gap: 0 !important; }}
.st-key-brandbar [data-testid="stElementContainer"]:has(.tr-topbar) {{ margin: 0; }}
.tr-topbar {{ display:flex; align-items:center; gap:18px; flex-wrap:wrap; padding: 18px 22px; }}
.tr-topbar.nav {{ border-bottom: 1px solid var(--border); }}
.tr-topbar .logo {{ width:56px; height:56px; flex:none; border-radius:14px; background: var(--bg);
  border:1px solid var(--border); display:flex; align-items:center; justify-content:center; }}
.tr-topbar .logo img {{ width:44px; height:44px; object-fit:contain; display:block; }}
.tr-topbar .name {{ display:flex; flex-direction:column; gap:4px; min-width: 0; flex: 1 1 260px; }}
.tr-topbar .name b {{ font-size: 1.5rem; line-height: 1.15; font-weight: 700; color: var(--ink); letter-spacing: -.01em;
  white-space: nowrap; }}
.tr-topbar .name b sup {{ font-size: .5em; color: var(--muted); font-weight: 600; margin-left: 1px; }}
.tr-topbar .name b .mod {{ color: var(--brand-t); }}
.st-key-seccards, .st-key-pagebar, div[class*="st-key-seccard_"] {{ background: transparent !important;
  border: 0 !important; box-shadow: none !important; padding: 0 !important; }}
.stTabs [role="tablist"] {{ background: var(--grid); border-radius: 11px; padding: 3px; gap: 2px; border: 0 !important;
  width: fit-content; max-width: 100%; flex-wrap: wrap; }}
.stTabs [role="tablist"] > :not([role="tab"]) {{ display: none !important; }}
.stTabs [role="tab"] {{ height: 38px !important; border-radius: 8px !important; padding: 0 16px !important;
  border: 0 !important; background: transparent; color: var(--muted); margin: 0 !important; }}
.stTabs [role="tab"]:hover {{ color: var(--ink); background: color-mix(in srgb, var(--card) 60%, transparent); }}
.stTabs [role="tab"][aria-selected="true"] {{ background: var(--card) !important; color: var(--brand-t) !important;
  box-shadow: 0 1px 2px rgba(16,40,36,.16); }}
.stTabs [role="tab"] p {{ font-weight: 560; }}
.stTabs [role="tab"][aria-selected="true"] p {{ font-weight: 650; }}
.tr-topbar .name > span {{ font-size: .88rem; color: var(--muted); line-height: 1.3; }}
.tr-topbar .name span .ver {{ font-family:'JetBrains Mono',monospace; font-size:.78rem; color: var(--brand-t);
  border:1px solid var(--brand-line); background: var(--brand-soft); padding: 1px 8px; border-radius: 999px; margin-left: 6px; }}
.tr-topbar .right {{ margin-left:auto; display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
/* ---- Option 1: section cards + pill page tabs */
.st-key-brandbar {{ margin-bottom: 10px; }}
.st-key-seccards {{ margin: 0 0 6px 0; }}
.st-key-seccards [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; row-gap: 12px !important; gap: 12px !important; }}
.st-key-seccards [data-testid="stColumn"] {{ flex: 1 1 220px !important; min-width: min(220px, 100%); }}
div[class*="st-key-seccard_"] {{ position: relative; gap: 0 !important; }}
div[class*="st-key-seccard_"] .tr-sc {{ margin-bottom: 1rem; }}
.st-key-pagebar .tr-study {{ margin-bottom: 1rem !important; }}
.st-key-brandbar .tr-topbar {{ margin-bottom: 1rem; }}
div[class*="st-key-seccard_"] .stButton, div[class*="st-key-seccard_"] [data-testid="stElementContainer"]:has(.stButton) {{
  position: absolute; inset: 0; margin: 0; z-index: 2; }}
div[class*="st-key-seccard_"] .stButton button {{ width: 100%; height: 100%; opacity: 0; cursor: pointer; }}
div[class*="st-key-seccard_"] .stButton button:focus-visible {{ opacity: 1; background: transparent;
  outline: 3px solid var(--brand); outline-offset: 2px; color: transparent; }}
.tr-sc {{ display:flex; gap:12px; align-items:flex-start; padding: 14px 16px; border-radius: 14px; min-height: 76px;
  border: 1px solid var(--border); background: var(--card); box-shadow: var(--shadow); box-sizing: border-box;
  transition: border-color .15s, background .15s; }}
div[class*="st-key-seccard_"]:hover .tr-sc {{ border-color: var(--brand-line); }}
.tr-sc.on {{ border: 2px solid var(--brand); background: color-mix(in srgb, var(--brand) 8%, var(--card)); }}
.tr-sc .ic {{ width: 38px; height: 38px; flex: none; border-radius: 10px; display:flex; align-items:center;
  justify-content:center; background: var(--brand-soft); color: var(--brand); }}
.tr-sc.on .ic {{ background: var(--brand); color: #fff; }}
.tr-sc .tx {{ display:flex; flex-direction:column; gap:3px; min-width:0; }}
.tr-sc .tx b {{ font-size: .98rem; color: var(--ink); }}
.tr-sc.on .tx b {{ color: var(--brand-t); }}
.tr-sc .tx span {{ font-size: .78rem; color: var(--muted); line-height: 1.35; }}
.st-key-pagebar {{ margin: 0; }}
.st-key-pagebar [data-testid="stElementContainer"] {{ min-height: 0; }}
.st-key-pagebar [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; row-gap: 6px; }}
.st-key-pagebar [data-testid="stColumn"]:nth-child(1) {{ flex: 1 1 360px !important; min-width: 0; }}
.st-key-pagebar [data-testid="stColumn"]:nth-child(2) {{ flex: 0 0 130px !important; min-width: 130px; }}
.st-key-pagebar .tr-study {{ margin-top: 0; }}
.stTabs [data-baseweb="tab-list"] {{ background: var(--grid); border-radius: 11px; padding: 3px; gap: 2px;
  border-bottom: 0 !important; width: fit-content; max-width: 100%; }}
.stTabs [data-baseweb="tab"] {{ height: 38px !important; border-radius: 8px !important; padding: 0 16px !important; }}
.stTabs [aria-selected="true"] {{ background: var(--card) !important; box-shadow: 0 1px 2px rgba(16,40,36,.14); }}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.tr-brand {{ display:flex; align-items:center; gap:11px; }}
.tr-brand img {{ width: 38px; height: 38px; object-fit: contain; }}
.tr-brand .t {{ display:flex; flex-direction:column; line-height:1.15; }}
.tr-brand b {{ font-size: 1.12rem; font-weight: 700; color: var(--ink); white-space: nowrap; }}
.tr-brand b sup {{ font-size: .55em; color: var(--muted); }}
.tr-brand b .mod {{ color: var(--brand-t); }}
.tr-brand span.s {{ font-size: .78rem; color: var(--muted); white-space: nowrap; }}
.tr-setup {{ display:flex; flex-direction:column; gap:8px; margin: 2px 0 10px 0; }}
.tr-setup .row {{ display:flex; justify-content:space-between; align-items:baseline; }}
.tr-setup b {{ font-size: 1rem; color: var(--ink); }}
.tr-setup span {{ font-size: .8rem; color: var(--muted); }}
.tr-setup .bar {{ height: 6px; border-radius: 999px; background: var(--grid); overflow: hidden; }}
.tr-setup .bar div {{ height: 6px; background: var(--brand); border-radius: 999px; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] details {{ border-radius: 12px; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] {{ border: 2px solid var(--brand);
  background: color-mix(in srgb, var(--brand) 3%, var(--card)); }}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{ color: var(--brand); }}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{ font-size: .9rem; }}
.tr-study {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin: 6px 0 2px 0; font-size:.82rem; color: var(--muted); }}
.tr-study .chip {{ padding: 3px 10px; border-radius: 999px; border: 1px solid var(--border); background: var(--card); color: var(--ink); }}
.tr-study .chip.ok {{ color: var(--green); border-color: color-mix(in srgb, var(--green) 40%, transparent);
  background: color-mix(in srgb, var(--green) 7%, transparent); }}
.tr-study .case {{ color: var(--rose); font-weight: 600; }} .tr-study .ref {{ color: var(--green); font-weight: 600; }}
.tr-ptitle {{ margin: 4px 0 6px 0; }}
.tr-ptitle h1 {{ margin: 0; font-size: 1.55rem !important; font-weight: 700; }}
.tr-ptitle p {{ margin: 4px 0 0 0; color: var(--muted); font-size: .92rem; }}
/* ---- Prioritised targets: ranked list (fills the card, reflows by card width) ---- */
.tr-tg {{ container-type: inline-size; }}
.tr-tg .hd, .tr-tg .row {{ display:grid; grid-template-columns: 30px 86px minmax(130px,1fr) minmax(240px,2.4fr) 200px;
  gap: 14px; align-items: center; padding: 10px 4px; }}
.tr-tg .hd {{ padding: 0 4px 8px; font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  color: var(--brand-t); border-bottom: 1px solid var(--border); }}
.tr-tg .row {{ border-top: 1px solid var(--grid); }}
.tr-tg .row:nth-child(2) {{ border-top: 0; }}
.tr-tg .rk {{ color: var(--faint); font: 600 .8rem 'JetBrains Mono', monospace; text-align: right; }}
.tr-tg .g {{ font-weight: 700; font-size: .95rem; color: var(--ink); }}
.tr-tg .g small {{ display: block; font-weight: 500; font-size: .74rem; }}
.tr-tg .up {{ color: #C62828; }} .tr-tg .dn {{ color: #1565C0; }}
.tr-tg .bar {{ height: 10px; border-radius: 5px; background: var(--card2); position: relative; }}
.tr-tg .bar i {{ position: absolute; inset: 0 auto 0 0; border-radius: 5px; background: var(--brand); }}
.tr-tg .bar.ch i {{ background: #E65100; }}
.tr-tg .sc {{ font: 500 .72rem 'JetBrains Mono', monospace; color: var(--muted); margin-top: 4px; }}
.tr-tg .rx {{ font-weight: 600; font-size: .9rem; line-height: 1.3; color: var(--ink); overflow-wrap: anywhere; }}
.tr-tg .ev {{ color: var(--muted); font-size: .8rem; margin-top: 2px; }}
.tr-tg .pill {{ display: inline-block; font-size: .76rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
  line-height: 1.25; background: var(--brand-soft); color: var(--brand-t); border: 1px solid var(--brand-line); }}
.tr-tg .pill.bn {{ background: rgba(230,81,0,.10); color: #C2410C; border-color: rgba(230,81,0,.35); }}
.tr-tg .pill.na {{ background: var(--card2); color: var(--muted); border-color: var(--border); }}
.tr-tg .lg {{ margin-top: 10px; font-size: .78rem; color: var(--muted); }}
@container (max-width: 760px) {{
  .tr-tg .hd {{ display: none; }}
  .tr-tg .row {{ grid-template-columns: 26px 80px 1fr; row-gap: 6px; }}
  .tr-tg .row > .rxc {{ grid-column: 2 / -1; }}
  .tr-tg .row > .pc {{ grid-column: 2 / -1; }}
}}
</style>
"""


ICONS = {"layers": "&#9776;", "chart": "&#9638;", "check": "&#10003;", "flask": "&#9879;", "branch": "&#10023;",
         "stack": "&#9781;", "trend": "&#8599;", "bolt": "&#9889;", "db": "&#9707;", "spark": "&#10022;",
         "search": "&#9906;", "grid": "&#9638;", "target": "&#9678;", "flow": "&#8644;"}


def _esc(x) -> str:
    return html.escape(str(x))


@functools.lru_cache(maxsize=4)
def logo_data_uri(path: str) -> str | None:
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    return None


def header(version: str, logo_path: str | None, status: str, kb_text: str) -> str:
    uri = logo_data_uri(logo_path)
    logo = f'<img src="{uri}" alt="Omnix"/>' if uri else '<span style="font-size:1.4rem">&#10022;</span>'
    return (f'<div class="tr-head"><div class="tr-logo">{logo}</div>'
            f'<div class="tr-title"><b>Omnix<sup>&trade;</sup> <span class="mod">Integrix</span>'
            f'<span class="tr-ver">v{_esc(version)}</span></b>'
            f'<span>Transcriptomics: Proteomics: Metabolomics: Integration</span></div>{status}'
            f'<div class="tr-spacer"></div><span class="tr-pill info">{_esc(kb_text)}</span></div>')


def topbar(version: str, logo_path: str | None, status: str, kb_text: str, with_nav: bool = True) -> str:
    uri = logo_data_uri(logo_path)
    logo = f'<div class="logo"><img src="{uri}" alt="Omnix"/></div>' if uri else ""
    return (f'<div class="tr-topbar{" nav" if with_nav else ""}">{logo}<div class="name">'
            f'<b>Omnix<sup>&trade;</sup> <span class="mod">Integrix</span></b>'
            f'<span>Transcriptomics: Proteomics: Metabolomics: Integration</span></div>'
            f'<div class="right"></div></div>')


_SEC_SVG = [
    '<path d="M4 19V5M4 19h16M8 15l4-5 3 3 5-6"/>',
    '<circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/>'
    '<path d="M8.5 12h3l4-5M11.5 12l4 5"/>',
    '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/>',
    '<path d="M4 7h6l2 2h8v10H4z"/><path d="M12 12v5M9.5 14.5 12 17l2.5-2.5"/>',
]


def section_card(name: str, pages: str, idx: int, active: bool) -> str:
    svg = (f'<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
           f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_SEC_SVG[idx % 4]}</svg>')
    return (f'<div class="tr-sc{" on" if active else ""}"><span class="ic">{svg}</span>'
            f'<span class="tx"><b>{_esc(name)}</b><span>{_esc(pages)}</span></span></div>')


def brand(version: str, logo_path: str | None) -> str:
    uri = logo_data_uri(logo_path)
    logo = f'<img src="{uri}" alt="Omnix"/>' if uri else ""
    return (f'<div class="tr-brand">{logo}<div class="t"><b>Omnix<sup>&trade;</sup> <span class="mod">Integrix</span></b>'
            f'<span class="s">Transcriptomics: Proteomics: Metabolomics: Integration</span></div></div>')


def setup_head(n_ready: int, total: int) -> str:
    pct = int(100 * n_ready / max(total, 1))
    return (f'<div class="tr-setup"><div class="row"><b>Analysis setup</b><span>{n_ready} of {total} ready</span></div>'
            f'<div class="bar"><div style="width:{pct}%"></div></div></div>')


def study_line(tier_text: str, case: str, ref: str, alpha: float) -> str:
    tier = tier_text.split(":")[0].strip()
    return (f'<div class="tr-study"><span class="chip">{_esc(tier)}</span><span class="chip"><span class="case">'
            f'{_esc(case)}</span> vs <span class="ref">{_esc(ref)}</span></span>'
            f'<span class="chip ok">&#10003; FDR calibrated &middot; q &lt; {alpha:g}</span></div>')


def page_title(title: str, subtitle: str = "") -> str:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return f'<div class="tr-ptitle"><h1>{_esc(title)}</h1>{sub}</div>'


def status_pill(done: bool, seconds: float | None = None) -> str:
    if done:
        t = f' <span class="mono">&middot; {seconds:.1f}s</span>' if seconds is not None else ""
        return f'<span class="tr-pill ok"><span class="dot"></span>Complete{t}</span>'
    return '<span class="tr-pill idle"><span class="dot"></span>Ready to run</span>'


def banner(tier_text: str, case: str, ref: str, alpha: float) -> str:
    tier, _, desc = tier_text.partition(":")
    return (f'<div class="tr-banner"><div class="ic">{ICONS["stack"]}</div><div class="body">'
            f'<span class="tr-tag">{_esc(tier.strip())}</span>'
            f'<span class="tr-contrast">Contrast: <span class="case">{_esc(case)}</span> vs '
            f'<span class="ref">{_esc(ref)}</span></span>'
            f'<p>Directional integration of <b>transcriptomics &rarr; proteomics &rarr; enzyme function &rarr; '
            f'metabolomics</b>. {_esc(desc.strip()).capitalize()}. Every reaction is scored for directional '
            f'concordance, calibrated by label and network-topology permutations.</p></div>'
            f'<span class="tr-pill ok" style="align-self:center">&#10003; FDR calibrated (q &lt; {alpha:g})</span></div>')


def section(title: str, icon: str = "chart", badge: str | None = None) -> str:
    b = f'<span class="badge">{_esc(badge)}</span>' if badge else ""
    return f'<div class="tr-sec"><span class="ic">{ICONS.get(icon, icon)}</span>{_esc(title)}{b}</div>'


def targets_list(tg, top: int = 20) -> str:
    """Prioritised targets as a ranked list: rank, gene and direction, score bar (-log10 RRA; orange = choke
    point) with q-value, full best reaction with its evidence, and the reaction pattern as a tag."""
    import numpy as np
    d = tg.head(top).reset_index(drop=True)
    v = np.nan_to_num(-np.log10(d["rra_score"].astype(float).clip(1e-30, 1)))
    vmax = float(v.max()) if len(v) and v.max() > 0 else 1.0
    out = ['<div class="tr-tg"><div class="hd"><span>#</span><span>Gene</span><span>Score</span>'
           '<span>Best reaction &middot; evidence</span><span>Pattern</span></div>']
    for i, r in d.iterrows():
        z = r.get("enzyme_z")
        dr = ""
        if z is not None and str(z) != "nan":
            dr = '<small class="up">&#9650; up</small>' if float(z) > 0 else '<small class="dn">&#9660; down</small>'
        q = r.get("q_rra")
        qtx = f"q = {float(q):.1e}" if q is not None and str(q) != "nan" else ""
        pat = str(r.get("best_reaction_pattern", "") or "")
        pat = "" if pat == "nan" else pat
        pcls = "bn" if "bottleneck" in pat.lower() else ("na" if not pat else "")
        ev = str(r.get("evidence", "") or "").replace("; ", " \u00b7 ")
        ev = "" if ev == "nan" else ev
        rx = str(r.get("best_reaction", "") or "")
        rx = "" if rx == "nan" else rx
        out.append(
            f'<div class="row"><div class="rk">{i + 1}</div><div class="g">{_esc(r["gene"])}{dr}</div>'
            f'<div><div class="bar{" ch" if bool(r.get("choke_point")) else ""}"><i style="width:{100 * v[i] / vmax:.1f}%"></i></div>'
            f'<div class="sc">{qtx}</div></div>'
            f'<div class="rxc"><div class="rx">{_esc(rx) or "&mdash;"}</div><div class="ev">{_esc(ev)}</div></div>'
            f'<div class="pc"><span class="pill {pcls}">{_esc(pat or "no pattern")}</span></div></div>')
    out.append('<div class="lg">Bar length = &minus;log10 rank-aggregation score; orange bar = choke point; '
               '&#9650;/&#9660; = enzyme change in the case group.</div></div>')
    return "".join(out)


def card_head(title: str, subtitle: str = "", icon: str = "chart") -> str:
    sub = f"<span>{subtitle}</span>" if subtitle else ""
    return f'<div class="tr-cardhead"><b><span class="ic">{ICONS.get(icon, icon)}</span>{_esc(title)}</b>{sub}</div>'


def metric_cards(items: list[dict]) -> str:
    """items: dict(label, value, sub, accent (a CSS colour or var(--...)), icon, highlight=False)."""
    out = []
    for it in items:
        hl = it.get("highlight", False)
        style = f'--accent:{it["accent"]};' + (f'--labc:{it["accent"]};--valc:{it["accent"]};' if hl else "")
        out.append(f'<div class="tr-metric{" hl" if hl else ""}" style="{style}">'
                   f'<div class="lab"><span>{_esc(it["label"])}</span><span class="ic">{ICONS.get(it.get("icon", "chart"), "")}</span></div>'
                   f'<div class="val">{_esc(it["value"])}</div><div class="sub">{it.get("sub", "")}</div></div>')
    return '<div class="tr-metrics">' + "".join(out) + "</div>"


KIND_VAR = {"Bottleneck": "var(--orange)", "Throughput": "var(--green)", "Reaction": "var(--brand)",
            "Metabolite-driven": "var(--amber)", "Pathway": "var(--purple)", "Canonical": "var(--sky)",
            "Target": "var(--rose)", "Concordance": "var(--brand2)"}


def findings(rows: list[tuple[str, str]]) -> str:
    """rows: (kind, html text)."""
    if not rows:
        return '<div class="tr-find"><div class="txt" style="color:var(--muted)">No findings match this filter.</div></div>'
    out = []
    for kind, txt in rows:
        c = KIND_VAR.get(kind, "var(--muted)")
        out.append(f'<div class="tr-find"><span class="kind" style="color:{c};border-color:color-mix(in srgb, {c} 45%, transparent);'
                   f'background:color-mix(in srgb, {c} 7%, transparent)">{_esc(kind)}</span><div class="txt">{txt}</div></div>')
    return "".join(out)


def footer(version: str, kb: str) -> str:
    return (f'<div class="tr-foot"><span>Omnix&trade; Integrix v{_esc(version)}</span><span>Knowledge base {_esc(kb)}</span>'
            f'<span>Results describe evidence consistent with a mechanism, not proof of causation.</span></div>')


# --------------------------------------------------------------------------- dark previews of static figures
# Publication figures are drawn white; in dark mode their on-screen preview is recoloured to the dark palette.
def _svg_dark_map():
    p = DARK
    return {"#ffffff": p["card"], "#1f2a28": p["ink"], "#5d6d6a": p["muted"], "#4a5754": p["muted"],
            "#000000": "#C9D8D5", "#e3f2ef": "#123A35", "#00695c": "#14B8A6", "#00574d": "#5EEAD4",
            "#e4e9e8": p["grid"], "#eceeef": p["grid"], "#eceff0": p["grid"], "#eeeeee": p["grid"],
            "#f4f8fb": "#0D1A19", "#c9d3d1": p["axis"], "#333333": "#9FB3AF", "#444444": "#9FB3AF",
            "#555555": "#9FB3AF", "#8a9593": "#6E8581", "#b9c4c2": "#4A605C", "#9aa5a2": "#7F9692",
            "#262626": "#C9D8D5", "#f2f4f3": p["ink"]}


def darken_svg(svg: bytes) -> bytes:
    s = svg.decode("utf-8")
    for a, b in _svg_dark_map().items():
        s = s.replace(a, b.lower()).replace(a.upper(), b.lower())
    i = s.find(">", s.find("<svg")) + 1
    s = s[:i] + '<style>g[id^="text_"]{fill:#C9D8D5}</style>' + s[i:]
    return s.encode("utf-8")


# --------------------------------------------------------------------------- Plotly figures per mode
def _plotly_map(mode):
    if mode != "dark":
        return {'"#E65100"': '"#EA580C"'}
    p = DARK
    return {'"#1F2A28"': f'"{p["ink"]}"', '"rgba(255,255,255,0.8)"': '"rgba(17,33,32,0.88)"', '"white"': f'"{p["card"]}"',
            '"#555"': '"#9FB3AF"', '"#666"': '"#9FB3AF"', '"#888"': '"#9FB3AF"', '"#333"': '"#9FB3AF"',
            '"#999"': '"#5E7672"', '"#BBB"': '"#4A605C"', '"#BBBBBB"': '"#6E8581"', '"#CCC"': '"#4A605C"',
            '"#6B7775"': f'"{p["muted"]}"', '"#7F8C8D"': '"#6E8581"', '"#F0F2F2"': p["grid"].join('""'),
            '"#C9D3D1"': f'"{p["axis"]}"', '"#D5DBDB"': '"#2C4542"', '"#D0D5D4"': '"#3A524E"',
            '"#E3F2EF"': '"#123A35"', '"#00695C"': '"#14B8A6"', '"#F4F6F6"': f'"{p["grid"]}"', '"#E65100"': '"#FB923C"'}


def theme_figure(fig, mode: str = "light"):
    """Style a Integrix Plotly figure for the active mode (screen only; downloads keep the publication style)."""
    import plotly.graph_objects as go
    p = palette(mode)
    js = fig.to_json()
    for a, b in _plotly_map(mode).items():
        js = js.replace(a, b)
    out = go.Figure(json.loads(js))
    out.update_layout(template="plotly_dark" if mode == "dark" else "simple_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", color=p["ink"]),
                      hoverlabel=dict(bgcolor=p["card2"], bordercolor=p["border"], font=dict(color=p["ink"])),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    ax = dict(gridcolor=p["grid"], linecolor=p["axis"], zerolinecolor=p["axis"], tickcolor=p["axis"],
              tickfont=dict(color=p["muted"]), title_font=dict(color=p["muted"]))
    out.update_xaxes(**ax)
    out.update_yaxes(**ax)
    if out.layout.title and out.layout.title.text:
        out.layout.title.font.color = p["ink"]
    return out


# backward-compatible name
def dark_figure(fig):
    return theme_figure(fig, "dark")


def fluid_svg(svg: bytes) -> bytes:
    """Drop the fixed pt width/height of the root <svg> so the browser sizes it from its viewBox only (prevents
    letter-boxing when the image is scaled to the column width)."""
    import re
    t = svg.decode("utf-8")
    m = re.search(r"<svg\b[^>]*>", t)
    if m:
        tag = re.sub(r'\s(width|height)="[^"]*"', "", m.group(0))
        t = t[:m.start()] + tag + t[m.end():]
    return t.encode("utf-8")
