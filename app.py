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

import base64

# Lire et encoder le logo en base64
with open("logo-nav.svg", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

st.set_page_config(
    page_title="Moteur de Recommandation",
    page_icon=f"data:image/svg+xml;base64,{logo_b64}",
    layout="wide"
)

# On injectera le CSS plus bas après avoir déterminé le thème


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
        return df_pdt, df_users, df_notes, True, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False, str(e)

df_pdt, df_users, df_notes, is_connected, error_msg = load_data(DATABASE_URL)

if not is_connected and error_msg:
    st.error(f"DB Error: {error_msg}")


if is_connected and not df_pdt.empty:
    # Calcul des notes moyennes et nombre d'avis
    stats = df_notes.groupby('idproduit')['note'].agg(['mean', 'count']).reset_index()
    stats.columns = ['id', 'avg_note', 'nb_avis']
    df_pdt = pd.merge(df_pdt, stats, on='id', how='left')
    df_pdt['avg_note'] = df_pdt['avg_note'].fillna(0)
    df_pdt['nb_avis'] = df_pdt['nb_avis'].fillna(0).astype(int)

def get_star_rating(rating):
    full_stars = int(rating)
    half_star = "½" if (rating - full_stars) >= 0.5 else ""
    return "★" * full_stars + half_star

# ── Mapping des images ──────────────────────────────────────────────────────
PRODUCT_IMAGES = {
    1: "https://images.unsplash.com/photo-1551632811-561732d1e306?auto=format&fit=crop&w=800&q=80",
    2: "https://images.unsplash.com/photo-1501555088652-021faa106b9b?auto=format&fit=crop&w=800&q=80",
    3: "https://resaprivee.com/img/cms/zaghouane/djbel.jpg",
    4: "https://idwey.tn/uploads/0000/624/2021/02/17/69466303-678030916033653-6953890590136205312-o.jpg",
    5: "https://images.unsplash.com/photo-1511497584788-876760111969",
    6: "https://www.kharjet.tn/wp-content/uploads/2019/05/Bike-Paddle-Kayak.jpg",
    7: "https://images.unsplash.com/photo-1544551763-46a013bb70d5",
    8: "https://static-cms.routard.com/web-routard/uploads/pt144568_1353070_9648cdfd7d.jpg",
    9: "https://www.djerbaguide.com/wp-content/uploads/2024/06/plongee-sous-marine-a-Djerba.jpg",
    10: "https://cdn.sanity.io/images/nxpteyfv/goguides/1fc744761a73d98f36ea587160e3cfe7f9517cea-1600x1066.jpg",
    11: "https://cdn.getyourguide.com/img/tour/5c711bef4637d6588e85573bb2b86bd4f432cd8780532cdc0cac163c2492d44b.jpg/68.jpg",
    12: "https://israatravel.com/public/images/image/6_0.57392000-1697618712.jpg",
    13: "https://tunisie.co/uploads/images/content/tozeur-une-tc-060225.jpg",
    14: "https://media-cdn.tripadvisor.com/media/attractions-splice-spp-720x480/0e/ba/45/30.jpg",
    15: "https://nabeul.gov.tn/fr/wp-content/uploads/2015/04/gargoulettes-potier.jpg",
    16: "https://images.unsplash.com/photo-1512453979798-5ea266f8880c",
    17: "https://guide-voyage-tunisie.com/wp-content/uploads/2022/11/carthage5-1024x768.webp",
    18: "https://media-cdn.tripadvisor.com/media/attractions-splice-spp-674x446/15/1e/91/6d.jpg",
    19: "https://www.guidesulysse.com/images/destinations/iStock-101567872.jpg",
    20: "https://alpinemag.fr/wp-content/uploads/2023/11/BE23_Highway-to-Harr-26.jpeg",
}

if not is_connected or df_pdt.empty:
    st.error("Impossible de se connecter a la base PostgreSQL. Verifiez les parametres.")
    st.stop()

# ── Algorithmes ──────────────────────────────────────────────────────────────
@st.cache_data
def compute_similarity_matrix(df_pdt):
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
            
    return sim, pdt_ids

def get_content_based(user_id):
    sim, pdt_ids = compute_similarity_matrix(df_pdt)
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

    h['cb_score']   = h['cb_score'].clip(upper=1.0)
    h['cf_score']   = h['cf_score'].clip(upper=1.0)
    h['time_score'] = h['time_score'].clip(upper=1.0)

    h['final_score'] = h['cb_score'] * 0.4 + h['cf_score'] * 0.4 + h['time_score'] * 0.2
    rated = df_notes[df_notes['iduser'] == user_id]['idproduit'].tolist()
    return h[~h['id'].isin(rated)].sort_values('final_score', ascending=False).head(6)

# ── Hero Header — CSS hero uniquement (ne touche pas au reste) ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

.hero-banner {
    position: relative;
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    border-radius: 20px;
    padding: 52px 56px;
    margin-bottom: 40px;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 85% 20%, rgba(52,152,219,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(15,52,96,0.20) 0%, transparent 60%);
    pointer-events: none;
}
.hero-banner::after {
    content: 'تونس';
    position: absolute;
    right: 48px;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'DM Serif Display', serif;
    font-size: 8rem;
    color: rgba(255,255,255,0.04);
    line-height: 1;
    pointer-events: none;
}
.hero-eyebrow {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    position: relative;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(255,255,255,0.70);
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #FEFCE8;
    line-height: 1.15;
    margin: 0 0 14px 0;
    position: relative;
}
.hero-title em {
    font-style: italic;
    color: #5dade2;
}
.hero-sub {
    color: rgba(255,255,255,0.52);
    font-size: 0.95rem;
    font-weight: 300;
    line-height: 1.75;
    max-width: 540px;
    margin: 0;
    position: relative;
    font-family: 'DM Sans', sans-serif;
}
</style>

<div class="hero-banner">
    <div class="hero-eyebrow">
        <span class="hero-pill">Content-Based</span>
        <span class="hero-pill">Collaboratif</span>
        <span class="hero-pill">Time-Aware</span>
        <span class="hero-pill">Hybride</span>
    </div>
    <h1 class="hero-title">Découvrez la Tunisie,<br><em>à votre façon.</em></h1>
    <p class="hero-sub">Système intelligent combinant 3 algorithmes de filtrage — basé sur le contenu, collaboratif et temporel — pour proposer des expériences personnalisées à chaque utilisateur.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats globales ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl in [
    (c1, len(df_pdt),   "Experiences"),
    (c2, len(df_users), "Utilisateurs"),
    (c3, len(df_notes), "Interactions"),
    (c4, "4",           "Méthodes")]:
    col.markdown(f'<div class="stat-box"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Filtre par Catégorie ──────────────────────────────────────────────────────
st.markdown("###  Filtrer par Catégorie")
all_categories = ["Toutes"] + sorted(df_pdt['categorie'].unique().tolist())
selected_category = st.pills("Catégorie", all_categories, selection_mode="single", default="Toutes", label_visibility="collapsed")



# ── Sidebar : choix user + historique ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    dark_mode = st.toggle("🌙 Mode Sombre Premium", value=False)
    
    st.markdown("---")
    st.markdown("## 👤 Utilisateur")
    selected_user  = st.selectbox("Utilisateur", df_users['iduser'].tolist(), label_visibility="collapsed")
    user_history   = df_notes[df_notes['iduser'] == selected_user]

    theme_color = "#3498db"
    bg_gradient = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"

    # Injection du CSS (inchangé — uniquement .hero retiré car remplacé ci-dessus)
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        /* Keyframes Animations */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-8px); }}
            100% {{ transform: translateY(0px); }}
        }}

        .stApp {{ 
            background: {bg_gradient if not dark_mode else "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"};
            transition: background 0.5s ease;
        }}

        [data-testid="stSidebar"] {{
            background: {"rgba(255,255,255,0.55)" if not dark_mode else "rgba(15,23,42,0.8)"} !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255,255,255,0.3);
        }}

        /* Application de float aux pilules du hero si présentes */
        .hero-pill {{ animation: float 3s ease-in-out infinite; }}
        .hero-banner {{ animation: fadeInUp 0.8s ease-out; }}

        .card {{
            background: {"rgba(255,255,255,0.92)" if not dark_mode else "rgba(30,41,59,0.7)"};
            padding: 22px;
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.07);
            backdrop-filter: blur(5px);
            margin-bottom: 20px;
            border-top: 5px solid {theme_color if not dark_mode else "#38bdf8"};
            color: {"#1e293b" if not dark_mode else "#f1f5f9"};
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInUp 0.6s ease-out backwards;
            border-left: 1px solid rgba(255,255,255,0.1);
            border-right: 1px solid rgba(255,255,255,0.1);
        }}
        .card:hover {{ 
            transform: scale(1.03) translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            border-top: 5px solid #e74c3c;
            background: {"rgba(255,255,255,1.0)" if not dark_mode else "rgba(30,41,59,0.9)"};
        }}

        .tag {{
            background: {"linear-gradient(135deg, #3498db, #2980b9)" if not dark_mode else "rgba(56,189,248,0.2)"};
            color: white !important;
            padding: 4px 11px;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 4px;
            margin-bottom: 6px;
            transition: all 0.3s ease;
        }}
        .tag:hover {{ transform: translateY(-2px); filter: brightness(1.1); }}

        .stat-box {{
            background: {"rgba(255,255,255,0.85)" if not dark_mode else "rgba(30,41,59,0.5)"};
            border-radius: 14px;
            padding: 18px 22px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.06);
            color: {"#0f3460" if not dark_mode else "#f8fafc"};
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .stat-box:hover {{ transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); }}
        .stat-box .num {{ color: {theme_color if not dark_mode else "#38bdf8"}; font-size: 2.2rem; font-weight: 800; }}

        .explanation {{
            background: {"rgba(52,152,219,0.08)" if not dark_mode else "rgba(255,255,255,0.05)"};
            border-left: 4px solid {theme_color if not dark_mode else "#38bdf8"};
            padding: 12px 15px;
            border-radius: 8px;
            color: {"#2471a3" if not dark_mode else "#38bdf8"};
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 12px;
            transition: background 0.3s ease;
        }}

        .score-bar-label {{ font-size: 0.75rem; color: #888; margin-bottom: 2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        .score-bar-wrap {{ background: rgba(0,0,0,0.05); border-radius: 10px; height: 8px; margin-bottom: 10px; overflow: hidden; }}
        .score-bar-fill {{ height: 8px; border-radius: 10px; transition: width 1s cubic-bezier(0.65, 0, 0.35, 1); }}

        .rating-stars {{ color: #f1c40f; font-size: 1.1rem; margin-right: 5px; }}
        .rating-value {{ font-weight: 700; color: {"#1a1a2e" if not dark_mode else "#f8fafc"}; font-size: 0.95rem; }}
        .rating-count {{ color: #888; font-size: 0.8rem; margin-left: 4px; }}
        .rating-container {{ margin-bottom: 12px; display: flex; align-items: center; }}

        .card-img {{
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 14px;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: transform 0.5s ease;
        }}
        .card:hover .card-img {{ transform: scale(1.05); }}
        
        h3, h4 {{ color: {"#1a1a2e" if not dark_mode else "#f8fafc"} !important; font-weight: 800; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Historique")
    st.write(f"- **Expériences notées :** {len(user_history)}")
    if len(user_history) > 0:
        st.write(f"- **Note moyenne :** {user_history['note'].mean():.2f} / 5")


# ── Recommandations ───────────────────────────────────────────────────────────
st.markdown(f"### Recommandations pour l'Utilisateur {selected_user}")

recos = get_hybrid(selected_user)

if selected_category != "Toutes":
    recos = recos[recos['categorie'] == selected_category]

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

        rating_stars = get_star_rating(row.avg_note)
        rating_html = f"""
        <div class="rating-container">
            <span class="rating-stars">{rating_stars}</span>
            <span class="rating-value">{row.avg_note:.1f}</span>
            <span class="rating-count">({row.nb_avis} avis)</span>
        </div>"""

        pid = row.id
        nom = row.nompdt
        
        img_url = PRODUCT_IMAGES.get(int(row.id), "https://images.unsplash.com/photo-1506744038136-46273834b3fb")
        img_url += "?auto=format&fit=crop&w=600&h=400&q=80"

        with cols[idx % 3]:
            st.markdown(f"""
            <div class="card">
                <img src="{img_url}" class="card-img">
                <h4>{row.nompdt}</h4>
                <div class="meta"><b>Categorie :</b> {row.categorie} &nbsp;|&nbsp; <b>Effort :</b> {row.effort}</div>
                {rating_html}
                {tags_html}
                {score_bars}
                <div class="explanation">{explication}</div>
            </div>""", unsafe_allow_html=True)

# ── Graphique comparatif des scores ──────────────────────────────────────────
st.markdown("---")
st.markdown("### Analyse des scores par recommandation")

fig = go.Figure()
names = [r.nompdt[:22] + "…" if len(r.nompdt) > 22 else r.nompdt for r in recos.itertuples()]
fig.add_bar(name="Contenu",         x=names, y=recos['cb_score'].round(2),    marker_color="#3498db")
fig.add_bar(name="Collaboratif",    x=names, y=recos['cf_score'].round(2),    marker_color="#9b59b6")
fig.add_bar(name="Temporel",        x=names, y=recos['time_score'].round(2),  marker_color="#e67e22")
fig.add_bar(name="Hybride (Final)", x=names, y=recos['final_score'].round(2), marker_color="#2ecc71")

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
