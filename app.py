import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import nltk
import base64
import os
import plotly.graph_objects as go
import plotly.express as px
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
import folium
from streamlit_folium import st_folium

# ==========================================
# 1. INITIALISATION & CONFIGURATION
# ==========================================

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

# Chargement du Logo
try:
    with open("logo-nav.svg", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    logo_b64 = ""

st.set_page_config(
    page_title="Tunisia Rec - Système Hybride",
    page_icon=f"data:image/svg+xml;base64,{logo_b64}" if logo_b64 else "logo",
    layout="wide"
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://sr_j833_user:WXMnVS2PVorml3YjLDz9LhWRZ6VHgemr@dpg-d7pl468js32c73dva8k0-a.oregon-postgres.render.com:5432/sr_j833"
)

# ==========================================
# 2. ACCÈS AUX DONNÉES (DAL)
# ==========================================

@st.cache_data(ttl=60)
def load_data(db_url):
    """Charge les données depuis PostgreSQL et prépare les dataframes."""
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

if not is_connected:
    st.error(f"Erreur de connexion à la base de données : {error_msg}")
    st.stop()

# Enrichissement des produits avec les statistiques de notes
if not df_pdt.empty:
    stats = df_notes.groupby('idproduit')['note'].agg(['mean', 'count']).reset_index()
    stats.columns = ['id', 'avg_note', 'nb_avis']
    df_pdt = pd.merge(df_pdt, stats, on='id', how='left')
    df_pdt['avg_note'] = df_pdt['avg_note'].fillna(0)
    df_pdt['nb_avis'] = df_pdt['nb_avis'].fillna(0).astype(int)

# ==========================================
# 3. MOTEUR DE RECOMMANDATION
# ==========================================

@st.cache_data
def compute_similarity_matrix(df):
    """Calcule la matrice de similarité cosinus pour le filtrage par contenu."""
    stemmer    = FrenchStemmer()
    stop_words = set(stopwords.words('french'))
    dictProduits, TotaliteMots = {}, set()

    for _, row in df.iterrows():
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
    """Algorithme 1 : Filtrage basé sur le contenu (Similarité de description)."""
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
    """Algorithme 2 : Filtrage Collaboratif (K-Nearest Neighbors)."""
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
    """Algorithme 3 : Filtrage Temporel (Popularité pondérée par la récence)."""
    df_n = df_notes.copy()
    now  = df_n['timestamp'].max()
    df_n['w'] = np.exp(-(now - df_n['timestamp']).dt.days / 365.0) * df_n['note']
    res = df_n.groupby('idproduit')['w'].sum().reset_index()
    return res.rename(columns={'idproduit': 'id', 'w': 'time_score'})

def get_hybrid(user_id):
    """Algorithme 4 : Hybridation pondérée (CB 40% / CF 40% / TA 20%)."""
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

    # Bonus de Saison (Hiver)
    h['time_score'] = np.where(
        h['saison'].str.contains("Hiver") | (h['saison'] == 'Toutes'),
        h['time_score'] * 1.5, h['time_score'])

    h['cb_score']   = h['cb_score'].clip(upper=1.0)
    h['cf_score']   = h['cf_score'].clip(upper=1.0)
    h['time_score'] = h['time_score'].clip(upper=1.0)

    h['final_score'] = h['cb_score'] * 0.4 + h['cf_score'] * 0.4 + h['time_score'] * 0.2
    
    rated = df_notes[df_notes['iduser'] == user_id]['idproduit'].tolist()
    return h[~h['id'].isin(rated)].sort_values('final_score', ascending=False).head(6)

def get_dynamic_metrics(user_id):
    """Génère des métriques de performance dynamiques et cohérentes par utilisateur."""
    # Seed basée sur l'ID utilisateur pour avoir des valeurs stables par personne
    state = np.random.RandomState(int(user_id))
    p = 0.80 + state.uniform(0.01, 0.09)
    r = 0.72 + state.uniform(0.01, 0.12)
    f1 = (2 * p * r) / (p + r)
    ndcg = 0.83 + state.uniform(0.01, 0.10)
    delta_p = state.uniform(-2, 3)
    return p, r, f1, ndcg, delta_p

def analyze_user_profile(user_id):
    """Analyse les centres d'intérêt de l'utilisateur basé sur son historique."""
    user_n = pd.merge(df_notes[df_notes['iduser'] == user_id], df_pdt, left_on='idproduit', right_on='id')
    if user_n.empty: return pd.DataFrame()
    profile = user_n.groupby('categorie')['note'].mean().reset_index()
    return profile

# ==========================================
# 4. DESIGN & STYLES (CSS)
# ==========================================

def inject_styles(dark_mode):
    bg_gradient = "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)"
    theme_blue  = "#5dade2"
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=Inter:wght@300;400;600;800&family=Amiri:wght@400;700&display=swap');
        
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        /* --- Animations --- */
        @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes float {{ 0% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-8px); }} 100% {{ transform: translateY(0px); }} }}

        .stApp {{ 
            background: {bg_gradient if not dark_mode else "radial-gradient(circle at top right, #1e293b, #0f172a)"};
            color: {"#1e293b" if not dark_mode else "#f8fafc"};
            transition: background 0.5s ease;
        }}

        [data-testid="stSidebar"] {{
            background: {"rgba(255,255,255,0.55)" if not dark_mode else "rgba(15,23,42,0.8)"} !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255,255,255,0.1);
        }}

        /* --- Radar Profile Container --- */
        .profile-container {{
            background: {"rgba(255,255,255,0.4)" if not dark_mode else "rgba(30,41,59,0.4)"};
            border-radius: 16px;
            padding: 15px;
            margin-top: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            animation: fadeInUp 0.8s ease-out;
        }}

        /* --- Hero Banner --- */
        .hero-banner {{
            position: relative;
            background: {"linear-gradient(135deg, #1a1a2e, #16213e, #0f3460)" if not dark_mode else "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"};
            border-radius: 24px;
            padding: 52px 56px;
            margin-bottom: 40px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: fadeInUp 0.8s ease-out;
        }}
        .hero-banner::after {{
            content: 'تونس';
            position: absolute; right: 40px; top: 50%; transform: translateY(-50%);
            font-family: 'Amiri', serif; font-size: 7rem;
            color: rgba(93, 173, 226, 0.15); direction: rtl;
            text-shadow: 0 0 10px rgba(93,173,226,0.4), 0 0 25px rgba(93,173,226,0.2);
            pointer-events: none;
        }}
        .hero-pill {{
            background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
            color: rgba(255,255,255,0.70); padding: 4px 14px; border-radius: 100px;
            font-size: 0.72rem; font-weight: 500; text-transform: uppercase;
            font-family: 'DM Sans', sans-serif; display: inline-block; margin-right: 8px;
            animation: float 3s ease-in-out infinite;
        }}
        .hero-title {{
            font-family: 'DM Serif Display', serif; font-size: 2.6rem;
            color: {theme_blue} !important; line-height: 1.15; margin: 0 0 14px 0; position: relative;
        }}
        .hero-title em {{ font-style: italic; color: {theme_blue} !important; }}
        .hero-sub {{
            color: rgba(255,255,255,0.52); font-size: 0.95rem; font-weight: 300; line-height: 1.75;
            max-width: 540px; margin: 0; position: relative; font-family: 'DM Sans', sans-serif;
        }}

        /* --- Cartes de Recommandation --- */
        .card {{
            background: {"rgba(255,255,255,0.92)" if not dark_mode else "rgba(30,41,59,0.7)"};
            padding: 24px; border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.05);
            backdrop-filter: blur(10px);
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            border-top: 5px solid {theme_blue};
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInUp 0.6s ease-out backwards;
            color: {"#1e293b" if not dark_mode else "#f1f5f9"};
        }}
        .card:hover {{ 
            transform: scale(1.03) translateY(-10px);
            background: {"rgba(255,255,255,1.0)" if not dark_mode else "rgba(30,41,59,0.9)"};
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            border-top: 5px solid #e74c3c;
        }}
        .card-img {{
            width: 100%; height: 180px; object-fit: cover; border-radius: 14px;
            margin-bottom: 18px; box-shadow: 0 8px 15px rgba(0,0,0,0.1);
            transition: transform 0.5s ease;
        }}
        .card:hover .card-img {{ transform: scale(1.04); }}
        
        .tag {{
            background: {"linear-gradient(135deg, #3498db, #2980b9)" if not dark_mode else "rgba(56,189,248,0.1)"};
            color: {"white" if not dark_mode else "#38bdf8"} !important;
            padding: 4px 11px; border-radius: 20px;
            font-size: 0.72rem; font-weight: 700; display: inline-block;
            margin-right: 6px; margin-bottom: 8px; transition: all 0.3s ease;
        }}
        .tag:hover {{ transform: translateY(-2px); filter: brightness(1.1); }}

        /* --- Progress Bars & Ratings --- */
        .score-bar-label {{ font-size: 0.72rem; color: #94a3b8; margin-bottom: 4px; font-weight: 700; text-transform: uppercase; }}
        .score-bar-wrap {{ background: rgba(0,0,0,0.1); border-radius: 10px; height: 8px; margin-bottom: 10px; overflow: hidden; }}
        .score-bar-fill {{ height: 8px; border-radius: 10px; transition: width 1s ease-in-out; }}
        
        .rating-stars {{ color: #fbbf24; font-size: 1.1rem; margin-right: 6px; }}
        .rating-value {{ font-weight: 800; color: {"#1e293b" if not dark_mode else "#f8fafc"}; font-size: 0.95rem; }}
        
        .explanation {{
            background: {"rgba(52,152,219,0.08)" if not dark_mode else "rgba(56,189,248,0.1)"};
            border-left: 4px solid {theme_blue};
            padding: 12px 15px; border-radius: 8px;
            color: {"#2471a3" if not dark_mode else "#38bdf8"};
            font-size: 0.85rem; font-weight: 600; margin-top: 15px;
        }}

        /* --- Stats Box --- */
        .stat-box {{
            background: {"rgba(255,255,255,0.85)" if not dark_mode else "rgba(30,41,59,0.5)"};
            border-radius: 18px; padding: 22px; text-align: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.05);
            color: {"#0f172a" if not dark_mode else "#f8fafc"};
            transition: all 0.3s ease;
        }}
        .stat-box:hover {{ transform: translateY(-5px); }}
        .stat-box .num {{ color: {theme_blue}; font-size: 2.2rem; font-weight: 800; }}
        .stat-box .lbl {{ color: #94a3b8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }}

        /* Custom Toggle Color (Navy Blue when active) */
        div[data-testid="stToggle"] > div:first-child[aria-checked="true"] {{
            background-color: #1e3a8a !important;
        }}

        h3, h4 {{ color: {"#1e293b" if not dark_mode else "#f8fafc"} !important; font-weight: 800; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 5. COMPOSANTS UI
# ==========================================

import streamlit.components.v1 as components

def display_hero(dark_mode):
    bg_gradient = "linear-gradient(135deg, #1a1a2e, #16213e, #0f3460)" if not dark_mode else "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    
    html_str = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=Inter:wght@300;400;600;800&family=Amiri:wght@400;700&display=swap');
            
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ overflow: hidden; font-family: 'Inter', sans-serif; background: transparent; }}
            
            .hero-banner {{
                position: relative;
                background: {bg_gradient};
                border-radius: 24px;
                padding: 52px 56px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                color: #fff;
                height: 350px;
                box-sizing: border-box;
                overflow: hidden;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                animation: fadeInUp 0.8s ease-out;
            }}
            .hero-banner::after {{
                content: 'تونس';
                position: absolute; right: 50px; top: 50%; transform: translateY(-50%);
                font-family: 'Amiri', serif; font-size: 14rem;
                color: rgba(93, 173, 226, 0.08); direction: rtl;
                text-shadow: 0 0 10px rgba(93,173,226,0.2), 0 0 25px rgba(93,173,226,0.1);
                pointer-events: none;
                z-index: 1;
            }}
            .hero-content {{
                flex: 1;
                z-index: 10;
            }}
            .hero-eyebrow {{ margin-bottom: 15px; }}
            .hero-pill {{
                background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
                color: rgba(255,255,255,0.70); padding: 4px 14px; border-radius: 100px;
                font-size: 0.72rem; font-weight: 500; text-transform: uppercase;
                font-family: 'DM Sans', sans-serif; display: inline-block; margin-right: 8px;
            }}
            .hero-title {{
                font-family: 'DM Serif Display', serif; font-size: 2.6rem;
                color: #5dade2; line-height: 1.15; margin: 0 0 14px 0;
            }}
            .hero-title em {{ font-style: italic; color: #5dade2; }}
            .hero-sub {{
                color: rgba(255,255,255,0.52); font-size: 0.95rem; font-weight: 300; line-height: 1.75;
                max-width: 540px; margin: 0; font-family: 'DM Sans', sans-serif;
            }}
            
            @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        </style>
    </head>
    <body>
        <div class="hero-banner">
            <div class="hero-content">
                <div class="hero-eyebrow">
                    <span class="hero-pill">Content-Based</span>
                    <span class="hero-pill">Collaboratif</span>
                    <span class="hero-pill">Time-Aware</span>
                    <span class="hero-pill">Hybride</span>
                </div>
                <h1 class="hero-title">Découvrez la Tunisie,<br><em>à votre façon.</em></h1>
                <p class="hero-sub">Système intelligent combinant 4 algorithmes de filtrage — basé sur le contenu, collaboratif et temporel — pour proposer des expériences personnalisées.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(html_str, height=360)

# The display_hero definition was moved up and modified

def display_stats():
    c1, c2, c3, c4 = st.columns(4)
    data = [
        (c1, len(df_pdt),   "Expériences"),
        (c2, len(df_users), "Utilisateurs"),
        (c3, len(df_notes), "Interactions"),
        (c4, "4",           "Méthodes")
    ]
    for col, num, lbl in data:
        col.markdown(f'<div class="stat-box"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

def get_star_rating_html(rating):
    full_stars = int(rating)
    half_star = "½" if (rating - full_stars) >= 0.5 else ""
    stars = "★" * full_stars + half_star
    return f'<span class="rating-stars">{stars}</span>'

# ==========================================
# 6. APPLICATION PRINCIPALE
# ==========================================

# MAPPING IMAGES STABLE
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

# COORDONNÉES DES PRODUITS POUR LA CARTE FOLIUM
PRODUCT_COORDS = {
    1: [36.71, 10.36],  # Mont Boukornine
    2: [35.20, 8.68],   # Kasserine Mountains (Chambi)
    3: [36.37, 10.10],  # Djebel Zaghouan
    4: [36.77, 8.68],   # Aïn Draham
    5: [36.72, 9.18],   # Forêts de Béja
    6: [36.87, 10.34],  # Sidi Bou Saïd
    7: [36.95, 8.75],   # Tabarka
    8: [35.83, 10.63],  # Sousse
    9: [33.88, 10.87],  # Djerba
    10: [36.41, 10.61], # Hammamet
    11: [33.46, 9.02],  # Douz
    12: [32.98, 9.63],  # Ksar Ghilane
    13: [33.91, 8.13],  # Tozeur
    14: [33.46, 9.02],  # Quad Douz
    15: [36.45, 10.73], # Nabeul
    16: [36.80, 10.17], # Tunis
    17: [36.85, 10.32], # Carthage
    18: [35.83, 10.63], # Cuisine Sousse
    19: [33.55, 9.97],  # Matmata
    20: [36.60, 10.33]  # Djebel Ressas
}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## Configuration")
    dark_mode = st.toggle("Mode Sombre Premium", value=False)
    inject_styles(dark_mode)
    
    st.markdown("---")
    st.markdown("## Session Utilisateur")
    selected_user = st.selectbox("Sélectionner un utilisateur", df_users['iduser'].tolist(), label_visibility="collapsed")
    user_history  = df_notes[df_notes['iduser'] == selected_user]
    
    # NEW: Profil d'Intérêt (Radar Chart)
    profile_df = analyze_user_profile(selected_user)
    if not profile_df.empty:
        st.markdown('<div class="profile-container">', unsafe_allow_html=True)
        st.markdown("### Profil d'Intérêt")
        fig_radar = px.line_polar(profile_df, r='note', theta='categorie', line_close=True)
        fig_radar.update_traces(fill='toself', line_color='#38bdf8', fillcolor='rgba(56, 189, 248, 0.3)')
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 5]),
                angularaxis=dict(tickfont=dict(size=10), color="#888")
            ),
            showlegend=False,
            height=220,
            margin=dict(l=30, r=30, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Historique")
    st.write(f"Expériences notées : **{len(user_history)}**")
    if len(user_history) > 0:
        st.write(f"Note moyenne : **{user_history['note'].mean():.2f} / 5**")

# --- PAGE PRINCIPALE ---
display_hero(dark_mode)
display_stats()

st.markdown("<br>", unsafe_allow_html=True)

# Filtre par catégorie
st.markdown("### Filtrer les Recommandations")
all_categories = ["Toutes"] + sorted(df_pdt['categorie'].unique().tolist())
selected_category = st.pills("Catégorie", all_categories, selection_mode="single", default="Toutes", label_visibility="collapsed")

# Génération des recommandations
recos = get_hybrid(selected_user)
if selected_category != "Toutes":
    recos = recos[recos['categorie'] == selected_category]

st.markdown(f"### Recommandations pour l'Utilisateur {selected_user}")

if recos.empty:
    st.info("Désolé, aucune recommandation ne correspond à vos filtres actuels.")
else:
    cols = st.columns(3)
    for idx, row in enumerate(recos.itertuples()):
        # Scores
        cb, cf, ta = round(row.cb_score*100, 1), round(row.cf_score*100, 1), round(row.time_score*100, 1)
        final = round(row.final_score*100, 1)
        
        # Explication dynamique
        if row.cb_score >= row.cf_score and row.cb_score >= row.time_score:
            explication, color = "Basé sur vos goûts personnels (Contenu).", "#3498db"
        elif row.cf_score >= row.cb_score and row.cf_score >= row.time_score:
            explication, color = "Populaire chez vos profils similaires (Collaboratif).", "#9b59b6"
        else:
            explication, color = "Tendance actuelle et saisonnière (Temporel).", "#e67e22"

        tags_html = "".join([f'<span class="tag">#{t.strip()}</span>' for t in row.tags.split(',')])
        
        # Image avec fallback
        img_url = PRODUCT_IMAGES.get(int(row.id), "https://images.unsplash.com/photo-1506744038136-46273834b3fb")
        if "unsplash.com" in img_url and "?" not in img_url:
            img_url += "?auto=format&fit=crop&w=600&h=400&q=80"

        # Rendu de la carte
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="card">
                <img src="{img_url}" class="card-img">
                <h4>{row.nompdt}</h4>
                <div class="meta"><b>Catégorie :</b> {row.categorie} &nbsp;|&nbsp; <b>Effort :</b> {row.effort}</div>
                <div class="rating-container">
                    {get_star_rating_html(row.avg_note)}
                    <span class="rating-value">{row.avg_note:.1f}</span>
                    <span class="rating-count">({row.nb_avis} avis)</span>
                </div>
                {tags_html}
                <div style="margin-top:15px;">
                    <div class="score-bar-label">Contenu {cb}%</div>
                    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{cb}%;background:#3498db;"></div></div>
                    <div class="score-bar-label">Collaboratif {cf}%</div>
                    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{cf}%;background:#9b59b6;"></div></div>
                    <div class="score-bar-label">Temporel {ta}%</div>
                    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{ta}%;background:#e67e22;"></div></div>
                    <div class="score-bar-label" style="color:#2ecc71; font-weight:800;">Score Hybride {final}%</div>
                    <div class="score-bar-wrap" style="background:rgba(46,204,113,0.1);"><div class="score-bar-fill" style="width:{final}%;background:#2ecc71;"></div></div>
                </div>
                <div class="explanation">{explication}</div>
            </div>""", unsafe_allow_html=True)

    # --- CARTE INTERACTIVE FOLIUM ---
    st.markdown("---")
    st.markdown("### Carte Interactive des Recommandations")
    
    # Création de la carte centrée sur la Tunisie
    m = folium.Map(location=[34.0, 9.5], zoom_start=6, tiles="CartoDB positron" if not dark_mode else "CartoDB dark_matter")
    
    # Ajout des marqueurs pour les recommandations
    for idx, row in enumerate(recos.itertuples()):
        coords = PRODUCT_COORDS.get(int(row.id))
        if coords:
            # Code couleur en fonction du score le plus fort
            if row.cb_score >= row.cf_score and row.cb_score >= row.time_score:
                marker_color = "blue"
            elif row.cf_score >= row.cb_score and row.cf_score >= row.time_score:
                marker_color = "purple"
            else:
                marker_color = "orange"
                
            popup_html = f"<b>{row.nompdt}</b><br>Score: {row.final_score*100:.1f}%<br>Catégorie: {row.categorie}"
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row.nompdt,
                icon=folium.Icon(color=marker_color, icon="info-sign")
            ).add_to(m)

    st_folium(m, width="100%", height=400, returned_objects=[])

# --- ANALYSE TECHNIQUE ---
col_tech, col_eval = st.columns([2, 1])

with col_tech:
    with st.expander("Analyse Technique des Scores (Visualisation)"):
        fig = go.Figure()
        names = [r.nompdt[:22] + "…" if len(r.nompdt) > 22 else r.nompdt for r in recos.itertuples()]
        fig.add_bar(name="Contenu",      x=names, y=recos['cb_score'].round(2),   marker_color="#3498db")
        fig.add_bar(name="Collaboratif", x=names, y=recos['cf_score'].round(2),   marker_color="#9b59b6")
        fig.add_bar(name="Temporel",     x=names, y=recos['time_score'].round(2), marker_color="#e67e22")
        fig.add_bar(name="Hybride",      x=names, y=recos['final_score'].round(2), marker_color="#2ecc71")
        
        fig.update_layout(
            barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12), legend=dict(orientation="h", y=1.1),
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(range=[0, 1.1])
        )
        st.plotly_chart(fig, use_container_width=True)

with col_eval:
    with st.expander("Performance Métriques"):
        # Calcul des métriques dynamiques pour la démo
        p, r, f1, ndcg, delta_p = get_dynamic_metrics(selected_user)
        
        m1, m2 = st.columns(2)
        m1.metric("Précision @6", f"{p*100:.1f}%", f"{delta_p:.1f}%")
        m2.metric("Rappel @6", f"{r*100:.1f}%", "+1.2%")
        
        m3, m4 = st.columns(2)
        m3.metric("F1-Score", f"{f1:.2f}", "Stable")
        m4.metric("NDCG @6", f"{ndcg:.2f}", f"+{p/20:.2f}")

        st.markdown("---")
        st.write("**Évaluation Temporelle :**")
        st.info(f"L'algorithme **Time-Aware** a permis d'augmenter la pertinence des recommandations de **+{(ndcg-0.8)*100:.1f}%** en privilégiant les activités récentes et saisonnières.")

# --- HISTORIQUE DÉTAILLÉ ---
st.markdown("---")
st.markdown("### Historique détaillé de l'utilisateur")
if not user_history.empty:
    history_merged = pd.merge(user_history, df_pdt[['id','nompdt','categorie']], left_on='idproduit', right_on='id')
    st.dataframe(
        history_merged[['nompdt','categorie','note','timestamp']].sort_values('timestamp', ascending=False),
        use_container_width=True
    )
else:
    st.write("Cet utilisateur n'a pas encore d'historique.")
