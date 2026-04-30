import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import nltk
import plotly.graph_objects as go
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

st.set_page_config(page_title="Moteur de Recommandation", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.55) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.3);
    }

    .hero {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 20px;
        padding: 45px 50px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 10px 0; letter-spacing: -1px; }
    .hero p  { font-size: 1.05rem; opacity: 0.75; margin: 0; line-height: 1.7; }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 16px;
    }

    .card {
        background: rgba(255,255,255,0.92);
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.07), inset 0 0 0 1px rgba(255,255,255,0.6);
        backdrop-filter: blur(5px);
        margin-bottom: 20px;
        border-top: 5px solid #3498db;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 16px 35px rgba(0,0,0,0.12); border-top: 5px solid #e74c3c; }
    .card h4 { color: #1a1a2e; font-weight: 800; margin: 0 0 12px 0; font-size: 1.05rem; }
    .card .meta { color: #666; font-size: 0.85rem; margin-bottom: 12px; }

    .tag {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white; padding: 4px 11px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; display: inline-block;
        margin-right: 4px; margin-bottom: 6px; transition: all 0.2s ease;
    }
    .tag:hover { background: linear-gradient(135deg, #2ecc71, #27ae60); transform: scale(1.05); }

    .explanation {
        background-color: rgba(52,152,219,0.08);
        border-left: 4px solid #3498db;
        padding: 9px 12px; border-radius: 6px;
        color: #2471a3; font-size: 0.82rem; font-weight: 600; margin-top: 12px;
    }

    .score-bar-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; font-weight: 600; }
    .score-bar-wrap { background: #eee; border-radius: 10px; height: 8px; margin-bottom: 8px; overflow: hidden; }
    .score-bar-fill { height: 8px; border-radius: 10px; transition: width 0.6s ease; }

    .stat-box {
        background: rgba(255,255,255,0.85); border-radius: 14px;
        padding: 18px 22px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    .stat-box .num { font-size: 2rem; font-weight: 800; color: #0f3460; }
    .stat-box .lbl { font-size: 0.8rem; color: #888; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Configuration base de données ────────────────────────────────────────────
import os
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://sr_j833_user:WXMnVS2PVorml3YjLDz9LhWRZ6VHgemr@dpg-d7pl468js32c73dva8k0-a.oregon-postgres.render.com:5432/sr_j833"
)

# ── Chargement données ───────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(db_url):
    try:
        conn = psycopg2.connect(db_url)
        df_pdt   = pd.read_sql("SELECT * FROM Produit", conn)
        df_users = pd.read_sql("SELECT * FROM Users",   conn)
        df_notes = pd.read_sql("SELECT * FROM Notes",   conn)
        conn.close()
        df_notes['timestamp'] = pd.to_datetime(df_notes['timestamp'])
        return df_pdt, df_users, df_notes, True
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False

df_pdt, df_users, df_notes, is_connected = load_data(DATABASE_URL)

if not is_connected or df_pdt.empty:
    st.error("Impossible de se connecter a la base PostgreSQL. Verifiez les parametres.")
    st.stop()

# ── Algorithmes ──────────────────────────────────────────────────────────────
def get_content_based(user_id):
    stemmer    = FrenchStemmer()
    stop_words = set(stopwords.words('french'))
    dictProduits, TotaliteMots = {}, set()

    for _, row in df_pdt.iterrows():
        mots = nltk.word_tokenize(str(row['description']).lower(), language='french')
        stems = [stemmer.stem(m) for m in mots if m.isalnum() and stemmer.stem(m) not in stop_words]
        dictProduits[row['id']] = stems
        TotaliteMots.update(stems)

    TotaliteMots = list(TotaliteMots)
    pdt_ids = list(dictProduits.keys())
    mat = np.zeros((len(pdt_ids), len(TotaliteMots)))
    for i, pid in enumerate(pdt_ids):
        for j, m in enumerate(TotaliteMots):
            if m in dictProduits[pid]: mat[i][j] = 1

    def cos(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return np.dot(a, b) / n if n else 0

    sim = np.zeros((len(pdt_ids), len(pdt_ids)))
    for i in range(len(pdt_ids)):
        for j in range(len(pdt_ids)):
            sim[i][j] = cos(mat[i], mat[j])

    user_r   = df_notes[df_notes['iduser'] == user_id]
    top_pdts = user_r[user_r['note'] >= 4]['idproduit'].tolist() or user_r['idproduit'].tolist()[:1]
    if not top_pdts: return pd.DataFrame()

    scores = np.zeros(len(pdt_ids))
    for p in top_pdts:
        if p in pdt_ids: scores += sim[pdt_ids.index(p)]

    res = pd.DataFrame({'id': pdt_ids, 'cb_score': scores})
    return res[~res['id'].isin(user_r['idproduit'].tolist())]

def get_collaborative(user_id):
    from sklearn.metrics.pairwise import cosine_similarity
    mat = df_notes.pivot(index='iduser', columns='idproduit', values='note').fillna(0)
    if user_id not in mat.index: return pd.DataFrame()
    sim_df = pd.DataFrame(cosine_similarity(mat), index=mat.index, columns=mat.index)
    if len(sim_df) < 2: return pd.DataFrame()
    voisins = sim_df[user_id].sort_values(ascending=False)[1:4]
    preds = {}
    for pdt in mat.columns:
        if mat.loc[user_id, pdt] == 0:
            num = sum(sim * mat.loc[v, pdt] for v, sim in voisins.items() if mat.loc[v, pdt] > 0)
            den = sum(sim for v, sim in voisins.items() if mat.loc[v, pdt] > 0)
            preds[pdt] = num / den if den else 0
    return pd.DataFrame({'id': list(preds.keys()), 'cf_score': list(preds.values())})

def get_time_aware():
    df_n = df_notes.copy()
    now  = df_n['timestamp'].max()
    df_n['w'] = np.exp(-(now - df_n['timestamp']).dt.days / 365.0) * df_n['note']
    res = df_n.groupby('idproduit')['w'].sum().reset_index()
    return res.rename(columns={'idproduit': 'id', 'w': 'time_score'})

def get_hybrid(user_id):
    cb = get_content_based(user_id)
    cf = get_collaborative(user_id)
    ta = get_time_aware()
    h  = df_pdt.copy()

    def norm_merge(h, df, col):
        if df is not None and not df.empty:
            df[col] = df[col] / (df[col].max() + 1e-9)
            return pd.merge(h, df, on='id', how='left')
        h[col] = 0; return h

    h = norm_merge(h, cb, 'cb_score')
    h = norm_merge(h, cf, 'cf_score')
    h = norm_merge(h, ta, 'time_score')
    h.fillna(0, inplace=True)

    h['time_score'] = np.where(
        h['saison'].str.contains("Hiver") | (h['saison'] == 'Toutes'),
        h['time_score'] * 1.5, h['time_score'])

    h['final_score'] = h['cb_score'] * 0.4 + h['cf_score'] * 0.4 + h['time_score'] * 0.2
    rated = df_notes[df_notes['iduser'] == user_id]['idproduit'].tolist()
    return h[~h['id'].isin(rated)].sort_values('final_score', ascending=False).head(6)

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <span class="badge">Content-Based</span>
        <span class="badge">Collaboratif</span>
        <span class="badge">Time-Aware</span>
        <span class="badge">Hybride</span>
    </div>
    <h1>Moteur de Recommandation Hybride</h1>
    <p>Systeme intelligent combinant 3 algorithmes de filtrage — base sur le contenu, collaboratif 
    et temporel — pour proposer des experiences personnalisees a chaque utilisateur.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats globales ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in [
    (c1, len(df_pdt),   "Experiences"),
    (c2, len(df_users), "Utilisateurs"),
    (c3, len(df_notes), "Interactions"),
    (c4, "3",           "Algorithmes")]:
    col.markdown(f'<div class="stat-box"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar : choix user + historique ─────────────────────────────────────────
st.sidebar.markdown("## Choisissez un utilisateur")
selected_user  = st.sidebar.selectbox("Utilisateur", df_users['iduser'].tolist(), label_visibility="collapsed")
user_history   = df_notes[df_notes['iduser'] == selected_user]

st.sidebar.markdown("---")
st.sidebar.markdown("### Historique")
st.sidebar.write(f"- **Experiences notees :** {len(user_history)}")
if len(user_history) > 0:
    st.sidebar.write(f"- **Note moyenne :** {user_history['note'].mean():.2f} / 5")

# ── Recommandations ───────────────────────────────────────────────────────────
st.markdown(f"### Recommandations pour l'Utilisateur {selected_user}")

recos = get_hybrid(selected_user)

if recos.empty:
    st.info("Pas assez de donnees pour faire une recommandation.")
else:
    cols = st.columns(3)
    for idx, row in enumerate(recos.itertuples()):
        cb  = round(float(row.cb_score)   * 100, 1)
        cf  = round(float(row.cf_score)   * 100, 1)
        ta  = round(float(row.time_score) * 100, 1)

        if row.cb_score >= row.cf_score and row.cb_score >= row.time_score:
            explication = "Recommande car vous avez aime des experiences similaires (Contenu)."
            bar_color   = "#3498db"
        elif row.cf_score >= row.cb_score and row.cf_score >= row.time_score:
            explication = "Recommande car des utilisateurs similaires ont aime cette activite (Collaboratif)."
            bar_color   = "#9b59b6"
        else:
            explication = "Recommande car c'est une activite tres populaire en ce moment (Temporel)."
            bar_color   = "#e67e22"

        tags_html = "".join([f'<span class="tag">#{t.strip()}</span>' for t in row.tags.split(',')])

        score_bars = f"""
        <div style="margin-top:12px;">
            <div class="score-bar-label">Contenu {cb}%</div>
            <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{cb}%;background:#3498db;"></div></div>
            <div class="score-bar-label">Collaboratif {cf}%</div>
            <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{cf}%;background:#9b59b6;"></div></div>
            <div class="score-bar-label">Temporel {ta}%</div>
            <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{ta}%;background:#e67e22;"></div></div>
        </div>"""

        with cols[idx % 3]:
            st.markdown(f"""
            <div class="card">
                <h4>{row.nompdt}</h4>
                <div class="meta"><b>Categorie :</b> {row.categorie} &nbsp;|&nbsp; <b>Effort :</b> {row.effort}</div>
                {tags_html}
                {score_bars}
                <div class="explanation">{explication}</div>
            </div>""", unsafe_allow_html=True)

# ── Graphique comparatif des scores ──────────────────────────────────────────
st.markdown("---")
st.markdown("### Analyse des scores par recommandation")

fig = go.Figure()
names = [r.nompdt[:22] + "…" if len(r.nompdt) > 22 else r.nompdt for r in recos.itertuples()]
fig.add_bar(name="Contenu",      x=names, y=recos['cb_score'].round(2),   marker_color="#3498db")
fig.add_bar(name="Collaboratif", x=names, y=recos['cf_score'].round(2),   marker_color="#9b59b6")
fig.add_bar(name="Temporel",          x=names, y=recos['time_score'].round(2), marker_color="#e67e22")
fig.update_layout(
    barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter", size=12), legend=dict(orientation="h", y=1.1),
    margin=dict(l=0, r=0, t=20, b=0), height=360,
    xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
    yaxis=dict(gridcolor='rgba(0,0,0,0.05)', range=[0, 1.1])
)
st.plotly_chart(fig, use_container_width=True)

# ── Historique utilisateur ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Detail de vos notes precedentes")
history_merged = pd.merge(user_history, df_pdt[['id','nompdt','categorie']], left_on='idproduit', right_on='id')
st.dataframe(
    history_merged[['nompdt','categorie','note','timestamp']].sort_values('timestamp', ascending=False),
    use_container_width=True
)
