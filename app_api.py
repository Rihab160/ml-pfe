"""
═══════════════════════════════════════════════════════════════════════
Dashboard PFE - AWS CPU Monitor — VERSION API
───────────────────────────────────────────────────────────────────────
Frontend Streamlit qui consomme l'API Flask backend (api_server.py)
via des requêtes HTTP REST. Architecture découplée Backend/Frontend.

Prérequis :
  - api_server.py doit être lancé dans un terminal séparé sur le port 5000

Lancement :
  pip install streamlit pandas numpy requests
  streamlit run app_api.py
═══════════════════════════════════════════════════════════════════════
"""
import streamlit as st
import pandas as pd
import numpy as np
import json, smtplib, warnings, requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
API_URL = "http://127.0.0.1:5000"
SEUIL   = 0.60

st.set_page_config(page_title="PFE - AWS CPU Monitor (API)", page_icon="API",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, .stApp { background: var(--color-background-tertiary, #F1EFE8) !important; font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="block-container"] { padding: 0.3rem 1.5rem 2rem !important; max-width: 100% !important; }
[data-testid="stHeader"]  { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.api-pill-ok { font-size:10px; font-weight:500; background:#EAF3DE; color:#3B6D11; padding:3px 10px; border-radius:20px; border:0.5px solid #C0DD97; margin-left:6px; }
.api-pill-ko { font-size:10px; font-weight:500; background:#FCEBEB; color:#A32D2D; padding:3px 10px; border-radius:20px; border:0.5px solid #F5C6C6; margin-left:6px; }
.sec-label { font-size:10px; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:.10em; margin-bottom:12px; margin-top:16px; display:flex; align-items:center; gap:8px; }
.sec-label::before { content:""; display:inline-block; width:3px; height:12px; background:#7B3FE4; border-radius:2px; }
.mcard { background:var(--color-background-secondary); border-radius:8px; padding:12px 14px; }
.mlabel { font-size:11px; color:var(--color-text-secondary); margin-bottom:5px; text-transform:uppercase; letter-spacing:.04em; }
.mval { font-size:22px; font-weight:500; color:var(--color-text-primary); }
.msub { font-size:11px; margin-top:3px; }
.divider { height:0.5px; background:var(--color-border-tertiary); margin:18px 0; }
.badge { font-size:10px; padding:2px 8px; border-radius:6px; font-weight:500; }
.badge-crit { background:#FCEBEB; color:#A32D2D; }
.badge-warn { background:#FAEEDA; color:#854F0B; }
.badge-suspect { background:#F1F0EE; color:#6B6B6B; }
.arow { display:flex; align-items:center; gap:8px; padding:7px 10px; background:var(--color-background-secondary); border-radius:8px; margin-bottom:4px; }
.dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.stButton > button { background:#7B3FE4 !important; color:#fff !important; border:none !important; border-radius:8px !important; font-size:12px !important; font-weight:500 !important; padding:7px 18px !important; }
.welcome { display:flex; flex-direction:column; align-items:center; padding:60px 20px; text-align:center; }
.welcome h2 { font-size:1.4rem; font-weight:600; margin-bottom:10px; }
.welcome p { font-size:13px; color:var(--color-text-secondary); margin-bottom:24px; }
.api-status-banner { background:#FFF7ED; border:1px solid #FED7AA; border-radius:8px; padding:10px 14px; margin-bottom:12px; font-size:12px; color:#9A3412; }
.footer { display:flex; justify-content:space-between; padding:8px 0; margin-top:16px; border-top:0.5px solid var(--color-border-tertiary); font-size:10px; color:var(--color-text-secondary); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# CLIENTS API — Communication HTTP avec api_server.py
# ═══════════════════════════════════════════════════════════════════
def api_status():
    try:
        r = requests.get(f"{API_URL}/", timeout=3)
        return r.json() if r.ok else None
    except: return None

def api_get_servers():
    try:
        r = requests.get(f"{API_URL}/servers", timeout=5)
        return r.json() if r.ok else []
    except: return []

def api_get_metrics(fichier):
    try:
        r = requests.get(f"{API_URL}/cpu/metrics",
                         params={"file": fichier}, timeout=15)
        return r.json() if r.ok else None
    except Exception as e:
        st.error(f"Erreur API metrics : {e}")
        return None

def api_pipeline(data_records):
    """Appelle l'endpoint /pipeline qui fait Phase2+3+4 côté serveur."""
    try:
        r = requests.post(f"{API_URL}/pipeline",
                          json={"data": data_records}, timeout=60)
        try: return r.json()
        except: return {"error": f"HTTP {r.status_code} — {r.text[:300]}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Connexion refusée — vérifier que api_server.py est lancé sur {API_URL}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ═══════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════
def send_email(to, frm, pwd, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = frm; msg["To"] = to; msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls(); s.login(frm, pwd); s.sendmail(frm, to, msg.as_string())
        return True
    except: return False

def make_labels(timestamps):
    return [str(t)[11:16] for t in timestamps]


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
if "df_raw" not in st.session_state:
    st.session_state.df_raw      = None
    st.session_state.source_name = None
if "email_cfg" not in st.session_state:
    st.session_state.email_cfg = {"to":"", "from":"", "pwd":"", "seuil_w":75,
                                   "seuil_c":90, "notif_crit":True,
                                   "notif_warn":True, "saved":False}


# ═══════════════════════════════════════════════════════════════════
# TOPBAR avec indicateur API
# ═══════════════════════════════════════════════════════════════════
now_str   = datetime.now().strftime("%d %B %Y · %H:%M")
api_info  = api_status()
api_ok    = api_info is not None
api_pill  = (f'<span class="api-pill-ok">API connectée : {API_URL}</span>' if api_ok
             else f'<span class="api-pill-ko">API déconnectée</span>')

st.markdown(f"""
<div style="padding:4px 4px 8px 4px;margin-bottom:8px;">
  <div style="font-size:20px;font-weight:700;color:#1E293B;line-height:1.2">
    AWS CPU Monitor
    <span style="font-size:12px;font-weight:400;color:#7B3FE4;margin-left:10px">— Architecture API REST</span>
  </div>
  <div style="margin-top:6px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <span style="font-size:11px;color:#CBD5E1">{now_str}</span>
    {api_pill}
  </div>
</div>
""", unsafe_allow_html=True)

if not api_ok:
    st.markdown(f"""<div class="api-status-banner">
    <b>API Backend non accessible.</b> Lancez dans un terminal séparé :<br>
    <code>python api_server.py</code><br>
    L'API doit tourner sur <code>{API_URL}</code> avant d'utiliser ce dashboard.
    </div>""", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# SOURCE DE DONNÉES
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">Source de donnees (via API REST)</div>',
            unsafe_allow_html=True)

with st.expander("Charger des données via l'API", expanded=st.session_state.df_raw is None):
    # Option 1 — Sélection serveur depuis l'API
    st.markdown('<div style="font-size:11px;color:#374151;font-weight:600;margin-bottom:6px">'
                'Option 1 — Choisir un serveur depuis l\'API</div>', unsafe_allow_html=True)
    serveurs_api = api_get_servers()
    if serveurs_api:
        c1, c2 = st.columns([4, 1])
        with c1:
            srv_api = st.selectbox("", ["—"] + serveurs_api, key="srv_api",
                                    label_visibility="collapsed")
        with c2:
            api_btn = st.button("Charger via API", use_container_width=True)

        if api_btn and srv_api and srv_api != "—":
            with st.spinner(f"GET {API_URL}/cpu/metrics?file={srv_api}..."):
                data = api_get_metrics(srv_api)
                if data and not isinstance(data, dict) or (
                   isinstance(data, dict) and "error" not in data):
                    st.session_state.df_raw = pd.DataFrame(
                        data if isinstance(data, list) else [data])
                    st.session_state.source_name = f"API: {srv_api}"
                    st.success(f"{len(st.session_state.df_raw):,} points chargés via API")
                    st.rerun()
                else:
                    st.error(f"Erreur : {data.get('error', 'inconnue')}")
    else:
        st.warning(f"Aucun CSV trouvé dans le dossier de api_server.py")

    st.markdown('<div style="height:1px;background:#F1F5F9;margin:14px 0"></div>',
                unsafe_allow_html=True)

    # Option 2 — Upload CSV local
    st.markdown('<div style="font-size:11px;color:#374151;font-weight:600;margin-bottom:6px">'
                'Option 2 — Uploader un fichier CSV local</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([4, 1])
    with c3:
        uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
    with c4:
        imp_btn = st.button("Importer CSV", use_container_width=True)

    if uploaded and imp_btn:
        df_up = pd.read_csv(uploaded)
        st.session_state.df_raw = df_up
        st.session_state.source_name = uploaded.name
        st.success(f"{len(df_up):,} lignes importées")
        st.rerun()


if st.session_state.df_raw is None:
    st.markdown("""<div class="welcome">
    <h2>Architecture Backend/Frontend découplée</h2>
    <p>Le pipeline complet (Phase 2 + Phase 3 + Phase 4) s'exécute sur le serveur Flask.<br>
    Streamlit n'est ici qu'un client HTTP qui consomme l'API REST.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# APPEL PIPELINE API
# ═══════════════════════════════════════════════════════════════════
df_raw = st.session_state.df_raw.copy()
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
df_raw = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

if "serveur_id" not in df_raw.columns:
    df_raw["serveur_id"] = st.session_state.source_name or "serveur1"

serveurs = sorted(df_raw["serveur_id"].unique())
srv_sel  = st.selectbox("", serveurs, format_func=lambda x: f"EC2 — {x}",
                         label_visibility="collapsed")

df_srv_full = df_raw[df_raw["serveur_id"] == srv_sel].sort_values("timestamp").reset_index(drop=True)
ts_all = pd.to_datetime(df_srv_full["timestamp"])
ts_min = ts_all.min(); ts_max = ts_all.max()


# ═══════════════════════════════════════════════════════════════════
# FENÊTRE TEMPORELLE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">fenetre d\'affichage</div>', unsafe_allow_html=True)

cr1, _ = st.columns([1, 3])
with cr1:
    st.markdown('<div style="font-size:11px;font-weight:600;color:#374151;margin-bottom:8px;display:flex;align-items:center;gap:6px"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#378ADD"></span>Duree d\'affichage</div>',
                unsafe_allow_html=True)
    preset = st.selectbox("", ["6h","12h","24h","48h","Tout"],
                          index=1, key="pr", label_visibility="collapsed")

durees_map = {"6h":6,"12h":12,"24h":24,"48h":48}
if preset == "Tout":
    dt_debut, dt_fin = ts_min, ts_max
else:
    dt_debut = ts_min
    dt_fin   = ts_min + pd.Timedelta(hours=durees_map[preset])
dt_fin = min(dt_fin, ts_max)

mask    = (ts_all >= dt_debut) & (ts_all <= dt_fin)
df_srv  = df_srv_full[mask].reset_index(drop=True)

if len(df_srv) < 20:
    st.warning(f"Fenêtre trop courte ({len(df_srv)} obs)."); st.stop()


# ═══════════════════════════════════════════════════════════════════
# APPEL API /pipeline
# ═══════════════════════════════════════════════════════════════════
df_to_send = df_srv.copy()
df_to_send["timestamp"] = df_to_send["timestamp"].astype(str)
records = df_to_send[["timestamp", "value", "serveur_id"]].to_dict(orient="records")

with st.spinner(f"POST {API_URL}/pipeline — Phase 2 + Phase 3 + Phase 4..."):
    result = api_pipeline(records)

if result is None or "error" in (result or {}):
    err = result.get("error", "inconnue") if result else "pas de réponse"
    st.error(f"Erreur API pipeline : {err}")
    st.stop()

pred_vals = result["prediction"]["values"]
mae_srv   = result["prediction"]["mae"]
r2_srv    = result["prediction"]["r2"]
confirmed = np.array(result["detection"]["confirmed"])
nb_accord = np.array(result["detection"]["consensus"])
score     = np.array(result["detection"]["score"])


# ═══════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════
cpu       = np.clip(df_srv["value"].values * 100, 0, 100)
ts        = df_srv["timestamp"]
mu        = float(cpu.mean())
split_idx = int(len(cpu) * 0.7)
seuil_vis = float(np.percentile(cpu[:split_idx], 95)) if split_idx > 0 else float(cpu.max()*0.9)
n_anom    = int(confirmed.sum())
n_crit    = int((nb_accord == 3).sum())
cpu_now   = float(cpu[-1])
cpu_pred  = min(float(pred_vals[-1])*100, 100.0) if pred_vals else cpu_now
dispo     = round(100 - n_anom/len(df_srv)*100, 1)
cpu_trend = cpu_now - float(cpu[-6]) if len(cpu) > 5 else 0
pct_anom  = n_anom / len(df_srv) * 100

mae_display = f"{mae_srv:.5f}" if mae_srv > 0 else "N/A"
r2_display  = f"{r2_srv:.4f}"  if r2_srv  != 0 else "N/A"

col_cpu = "#A32D2D" if cpu_now>=90 else "#854F0B" if cpu_now>=75 else "#3B6D11"
col_anom = "#A32D2D" if n_anom>10 else "#854F0B" if n_anom>3 else "#3B6D11"
t_sym = "↑" if cpu_trend>0 else "↓"
t_col = "#A32D2D" if cpu_trend>0 else "#3B6D11"


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — KPI
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">section 1 — indicateurs cles (donnees via api)</div>',
            unsafe_allow_html=True)
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:4px">
  <div class="mcard"><div class="mlabel">CPU moyen</div>
    <div class="mval" style="color:{col_cpu}">{mu:.1f}%</div>
    <div class="msub" style="color:{t_col}">{t_sym} {abs(cpu_trend):.1f}% vs 30 min</div></div>
  <div class="mcard"><div class="mlabel">Disponibilite</div>
    <div class="mval" style="color:#3B6D11">{dispo}%</div>
    <div class="msub">{len(serveurs)} serveurs</div></div>
  <div class="mcard"><div class="mlabel">Anomalies (API)</div>
    <div class="mval" style="color:{col_anom}">{n_anom}</div>
    <div class="msub" style="color:#A32D2D">{n_crit} critiques · score≥0.6</div></div>
  <div class="mcard"><div class="mlabel">XGBoost (via API)</div>
    <div class="mval" style="color:#7B3FE4">{cpu_pred:.1f}%</div>
    <div class="msub">MAE {mae_display} · R² {r2_display}</div></div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTIONS 2-3 — Graphiques
# ═══════════════════════════════════════════════════════════════════
N         = len(df_srv)
lbl_list  = make_labels(ts.values)
real_vals = [round(float(c), 2) for c in cpu]
thr_arr   = [round(seuil_vis, 1)] * N

if pred_vals and len(pred_vals) == N:
    pred_arr = [round(float(p)*100, 2) for p in pred_vals]
else:
    pred_arr = [0] * N

anom_pts = [{"x": lbl_list[i], "y": max(round(float(cpu[i]), 2), 8.0)}
            for i in range(N) if confirmed[i] == 1]

# Anomalies récentes HTML
anom_rows_html = ""
for idx in np.where(confirmed == 1)[0][::-1][:25]:
    i = int(idx)
    nb    = int(nb_accord[i]); cpu_v = float(cpu[i]); sc_v = float(score[i])
    ts_v  = str(df_srv["timestamp"].iloc[i])[11:16]
    srv_n = str(df_srv["serveur_id"].iloc[i])
    if   nb == 3: dc, bc, bl = "E24B4A", "badge-crit",    "critique"
    elif nb == 2: dc, bc, bl = "EF9F27", "badge-warn",    "warning"
    else:         dc, bc, bl = "6B6B6B", "badge-suspect", "suspect"
    anom_rows_html += (
        f'<div class="arow"><div class="dot" style="background:#{dc}"></div>'
        f'<div style="flex:1"><div style="font-size:12px;font-weight:500">{srv_n}</div>'
        f'<div style="font-size:11px;color:#64748B">'
        f'{ts_v} · score {sc_v:.2f} · CPU {cpu_v:.1f}%</div></div>'
        f'<span class="badge {bc}">{bl}</span></div>'
    )
if not anom_rows_html:
    anom_rows_html = ('<div class="arow"><div class="dot" style="background:#639922"></div>'
                      '<div style="font-size:12px;color:#64748B">Aucune anomalie detectee</div></div>')


CHARTS_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif;}}
body{{background:transparent;}}
.sl{{font-size:10px;font-weight:500;color:#64748B;text-transform:uppercase;margin:16px 0 10px;}}
.card{{background:#fff;border:0.5px solid #E2E8F0;border-radius:12px;padding:14px 16px;margin-bottom:10px;}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
.ct{{font-size:12px;font-weight:500;color:#64748B;margin-bottom:10px;}}
.legend{{display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;}}
.li{{display:flex;align-items:center;gap:5px;font-size:11px;color:#64748B;}}
.ll{{width:14px;height:2px;}} .ld{{width:8px;height:8px;border-radius:50%;}}
.arow{{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#F8FAFC;border-radius:8px;margin-bottom:4px;}}
.dot{{width:8px;height:8px;border-radius:50%;}}
.badge{{font-size:10px;padding:2px 8px;border-radius:6px;font-weight:500;}}
.badge-crit{{background:#FCEBEB;color:#A32D2D;}}
.badge-warn{{background:#FAEEDA;color:#854F0B;}}
.badge-suspect{{background:#F1F0EE;color:#6B6B6B;}}
</style></head><body>

<div class="sl">section 2 — cpu reel · cpu predit xgboost (via API)</div>
<div class="card">
  <div class="ct">CPU reel — {srv_sel} · {dt_debut.strftime("%d/%m %H:%M")} → {dt_fin.strftime("%d/%m %H:%M")}</div>
  <div style="position:relative;width:100%;height:180px"><canvas id="c1"></canvas></div>
  <div class="legend">
    <div class="li"><div class="ll" style="background:#378ADD"></div>CPU reel</div>
    <div class="li"><div class="ll" style="background:#E24B4A"></div>seuil p95</div>
    <div class="li"><div class="ld" style="background:#E24B4A"></div>anomalie (via API)</div>
  </div>
</div>
<div class="card">
  <div class="ct">CPU predit XGBoost (via API) · MAE {mae_display} · R² {r2_display}</div>
  <div style="position:relative;width:100%;height:180px"><canvas id="c2"></canvas></div>
  <div class="legend">
    <div class="li"><div class="ll" style="background:#7B3FE4"></div>CPU predit (réponse API)</div>
    <div class="li"><div class="ll" style="background:#E24B4A"></div>seuil p95</div>
  </div>
</div>

<div class="sl">section 3 — detection anomalies (POST /pipeline · IF+LOF+SVM)</div>
<div class="g2">
  <div class="card">
    <div class="ct">{n_anom} anomalies confirmees / {N} observations</div>
    <div style="position:relative;width:100%;height:160px"><canvas id="c3"></canvas></div>
  </div>
  <div class="card">
    <div class="ct">Anomalies recentes — score fusion ≥ 0.6</div>
    <div style="display:flex;flex-direction:column;gap:4px;max-height:220px;overflow-y:auto;">
      {anom_rows_html}
    </div>
  </div>
</div>

<script>
const lbl = {json.dumps(lbl_list)};
const real = {json.dumps(real_vals)};
const pred = {json.dumps(pred_arr)};
const thr  = {json.dumps(thr_arr)};
const anom = {json.dumps(anom_pts)};
const gridC = "rgba(128,128,128,0.12)";
const tFont = {{size:10}};
const mkX = () => ({{grid:{{color:gridC}},ticks:{{font:tFont,maxRotation:0,autoSkip:true,maxTicksLimit:72,
  callback:function(val,idx){{const lv=this.getLabelForValue(val);return lv?lv:"";}}
}}}});

const yAxis = {{min:0,max:100,grid:{{color:gridC}},ticks:{{font:tFont}}}};
const mkBase = () => ({{responsive:true,maintainAspectRatio:false,animation:false,
  plugins:{{legend:{{display:false}}}},scales:{{x:mkX(),y:yAxis}}}});

new Chart(document.getElementById("c1"),{{type:"line",data:{{labels:lbl,datasets:[
  {{data:real,borderColor:"#378ADD",borderWidth:1.5,pointRadius:0,tension:.2,fill:"origin"}},
  {{data:thr,borderColor:"#E24B4A",borderWidth:1.2,borderDash:[5,3],pointRadius:0,fill:false}},
  {{type:"scatter",data:anom,backgroundColor:"#E24B4A",pointRadius:5}}
]}},options:mkBase()}});

new Chart(document.getElementById("c2"),{{type:"line",data:{{labels:lbl,datasets:[
  {{data:pred,borderColor:"#7B3FE4",borderWidth:1.5,pointRadius:0,tension:.2,fill:"origin"}},
  {{data:thr,borderColor:"#E24B4A",borderWidth:1.2,borderDash:[5,3],pointRadius:0,fill:false}}
]}},options:mkBase()}});

new Chart(document.getElementById("c3"),{{type:"line",data:{{labels:lbl,datasets:[
  {{data:real,borderColor:"#378ADD",borderWidth:1.2,pointRadius:0,tension:.2,fill:false}},
  {{data:thr,borderColor:"#E24B4A",borderWidth:1.2,borderDash:[5,3],pointRadius:0,fill:false}},
  {{type:"scatter",data:anom,backgroundColor:"#E24B4A",pointRadius:5}}
]}},options:mkBase()}});
</script></body></html>"""

components.html(CHARTS_HTML, height=900)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — Alertes email
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">section 4 — alertes email</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    cfg_to   = st.text_input("Destinataire", placeholder="admin@example.com",
                              value=st.session_state.email_cfg["to"])
    cfg_from = st.text_input("Expediteur Gmail", placeholder="monitor@gmail.com",
                              value=st.session_state.email_cfg["from"])
    cfg_pwd  = st.text_input("Mot de passe app Gmail", type="password",
                              placeholder="xxxx xxxx xxxx xxxx")
    cn1, cn2 = st.columns(2)
    with cn1: seuil_w = st.number_input("Seuil warning (%)",
                            value=st.session_state.email_cfg["seuil_w"], min_value=0, max_value=99)
    with cn2: seuil_c = st.number_input("Seuil critique (%)",
                            value=st.session_state.email_cfg["seuil_c"], min_value=0, max_value=100)
with c2:
    notif_crit = st.checkbox("Alerte CRITIQUE", value=st.session_state.email_cfg["notif_crit"])
    notif_warn = st.checkbox("Alerte WARNING",  value=st.session_state.email_cfg["notif_warn"])
    notif_anom = st.checkbox("Alerte ANOMALIE (score ≥ 0.6)", value=True)
    if st.button("Sauvegarder"):
        st.session_state.email_cfg.update({
            "to": cfg_to, "from": cfg_from, "pwd": cfg_pwd,
            "seuil_w": seuil_w, "seuil_c": seuil_c,
            "notif_crit": notif_crit, "notif_warn": notif_warn, "saved": True
        })
        st.success("Parametres sauvegardes")

cfg = st.session_state.email_cfg
if cfg["saved"] and cfg["to"] and cfg["from"] and cfg["pwd"]:
    if cfg["notif_crit"] and cpu_now >= cfg["seuil_c"]:
        body = f"<h2>CRITIQUE CPU {cpu_now:.1f}% — {srv_sel}</h2><p>Anomalies : {n_anom} ({n_crit} critiques)</p>"
        if send_email(cfg["to"], cfg["from"], cfg["pwd"],
                      f"CRITIQUE CPU {cpu_now:.1f}% — {srv_sel}", body):
            st.error(f"ALERTE CRITIQUE envoyee a {cfg['to']}")
    elif cfg["notif_warn"] and cpu_now >= cfg["seuil_w"]:
        body = f"<h2>WARNING CPU {cpu_now:.1f}% — {srv_sel}</h2>"
        if send_email(cfg["to"], cfg["from"], cfg["pwd"],
                      f"WARNING CPU {cpu_now:.1f}% — {srv_sel}", body):
            st.warning(f"WARNING envoye a {cfg['to']}")

st.markdown(f"""
<div class="footer">
  <span>PFE — Dashboard Streamlit (ARCHITECTURE API)</span>
  <span>Backend Flask {API_URL} · Phase 2 + 3 + 4 côté serveur</span>
  <span>{st.session_state.source_name or "—"} · {datetime.now().strftime("%Y-%m-%d")}</span>
</div>
""", unsafe_allow_html=True)