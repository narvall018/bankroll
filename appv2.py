import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date as date_cls
from pathlib import Path
import json
import requests
import base64


# =========================
# Configuration GitHub
# =========================
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")  # format: "username/repo-name"
GITHUB_BRANCH = "main"


def test_github_connection():
    """Test la connexion GitHub et affiche les détails"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False, "Token ou repo manquant"
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True, "Connexion GitHub OK"
        else:
            return False, f"Erreur API GitHub: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Erreur de connexion: {str(e)}"


def github_file_exists(file_path):
    """Vérifie si un fichier existe sur GitHub"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        return response.status_code == 200
    except:
        return False


def load_from_github(file_path):
    """Charge un fichier depuis GitHub"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        st.error("Configuration GitHub manquante")
        return None
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            content = response.json()["content"]
            decoded_content = base64.b64decode(content).decode('utf-8')
            return decoded_content
        else:
            st.warning(f"Impossible de charger {file_path}: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement de {file_path}: {str(e)}")
        return None


def save_to_github(file_path, content, commit_message="Update from Streamlit app"):
    """Sauvegarde un fichier sur GitHub avec gestion d'erreur améliorée"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        st.error("Configuration GitHub manquante pour la sauvegarde")
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        # Vérifier si le fichier existe déjà pour récupérer le SHA
        response = requests.get(url, headers=headers)
        sha = None
        if response.status_code == 200:
            sha = response.json()["sha"]
        elif response.status_code != 404:
            st.error(f"Erreur lors de la vérification du fichier: {response.status_code}")
            return False
        
        # Encoder le contenu
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        # Préparer les données
        data = {
            "message": commit_message,
            "content": encoded_content,
            "branch": GITHUB_BRANCH
        }
        
        if sha:
            data["sha"] = sha
        
        # Envoyer la requête
        response = requests.put(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            return True
        else:
            st.error(f"Erreur lors de la sauvegarde sur GitHub: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Exception lors de la sauvegarde: {str(e)}")
        return False


# =========================
# Fichiers & Persistance
# =========================
BETS_FILE = "data/paris.csv"
CONFIG_FILE = "data/config.json"


COLUMNS = [
    "id","date","event","market","odds","prob",
    "strategy","kelly_fraction","cap_3pct",
    "fixed_mode","fixed_value",
    "stake","status","payout","profit","notes"
]


def ensure_files():
    """Créer les fichiers par défaut s'ils n'existent pas sur GitHub"""
    try:
        if not github_file_exists(CONFIG_FILE):
            default_config = json.dumps({"bankroll_start": 1000.0})
            success = save_to_github(CONFIG_FILE, default_config, "Initialize config file")
            if success:
                st.success("Fichier de configuration créé sur GitHub")
            else:
                st.error("Échec de création du fichier de configuration")
        
        if not github_file_exists(BETS_FILE):
            empty_df = pd.DataFrame(columns=COLUMNS)
            csv_content = empty_df.to_csv(index=False)
            success = save_to_github(BETS_FILE, csv_content, "Initialize bets file")
            if success:
                st.success("Fichier de paris créé sur GitHub")
            else:
                st.error("Échec de création du fichier de paris")
    except Exception as e:
        st.error(f"Erreur lors de la création des fichiers: {str(e)}")


def load_config():
    """Charge la configuration depuis GitHub"""
    try:
        content = load_from_github(CONFIG_FILE)
        if content:
            cfg = json.loads(content)
            return float(cfg.get("bankroll_start", 1000.0))
    except Exception as e:
        st.warning(f"Erreur lors du chargement de la config: {str(e)}")
    return 1000.0


def save_config(bankroll_start):
    """Sauvegarde la configuration sur GitHub"""
    try:
        config_data = json.dumps({"bankroll_start": float(bankroll_start)})
        success = save_to_github(CONFIG_FILE, config_data, "Update bankroll config")
        if success:
            st.success("Configuration sauvegardée sur GitHub")
        else:
            st.error("Échec de sauvegarde de la configuration")
        return success
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde de la config: {str(e)}")
        return False


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]
    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in ["odds","prob","stake","payout","profit","kelly_fraction","fixed_value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cap_3pct"] = df["cap_3pct"].astype("boolean")
    for col in ["event","market","strategy","fixed_mode","status","notes"]:
        df[col] = df[col].astype("string")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df


def load_bets():
    """Charge les paris depuis GitHub"""
    try:
        content = load_from_github(BETS_FILE)
        if content:
            from io import StringIO
            df = pd.read_csv(StringIO(content))
            return ensure_schema(df)
    except Exception as e:
        st.warning(f"Erreur lors du chargement des paris: {str(e)}")
    return pd.DataFrame(columns=COLUMNS)


def save_bets(df: pd.DataFrame):
    """Sauvegarde les paris sur GitHub avec feedback utilisateur"""
    try:
        out = ensure_schema(df).copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        csv_content = out.to_csv(index=False)
        
        success = save_to_github(BETS_FILE, csv_content, "Update bets data")
        
        if success:
            st.success("✅ Paris sauvegardés sur GitHub")
            return True
        else:
            st.error("❌ Échec de sauvegarde sur GitHub")
            return False
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde des paris: {str(e)}")
        return False


def recompute_next_id():
    if st.session_state.bets.empty:
        st.session_state.next_id = 1
    else:
        max_id = pd.to_numeric(st.session_state.bets["id"], errors="coerce").fillna(0).astype(int).max()
        st.session_state.next_id = int(max_id) + 1


# =========================
# Init Session
# =========================
st.set_page_config(page_title="Bankroll & Paris", layout="wide")

# Test de connexion GitHub au démarrage
if not GITHUB_TOKEN or not GITHUB_REPO:
    st.error("⚠️ Configuration GitHub manquante. Vérifiez vos secrets Streamlit.")
    st.info("Assurez-vous d'avoir configuré GITHUB_TOKEN et GITHUB_REPO dans les secrets")
    st.stop()

# Test de connexion
github_ok, github_msg = test_github_connection()
if not github_ok:
    st.error(f"❌ Problème de connexion GitHub: {github_msg}")
    st.stop()
else:
    st.success(f"✅ {github_msg}")

ensure_files()

if "bets" not in st.session_state:
    st.session_state.bets = load_bets()
    st.session_state.bets = ensure_schema(st.session_state.bets)

if "next_id" not in st.session_state:
    recompute_next_id()

if "bankroll_start" not in st.session_state:
    st.session_state.bankroll_start = load_config()


# =========================
# Helpers métiers
# =========================
def compute_payout_profit(row):
    status = row.get("status", "En attente")
    stake = float(row.get("stake", 0.0) or 0.0)
    odds = float(row.get("odds", 0.0) or 0.0)
    if status == "Gagné":
        payout = stake * odds
        profit = payout - stake
    elif status == "Perdu":
        payout = 0.0
        profit = -stake
    elif status == "Remboursé":
        payout = stake
        profit = 0.0
    else:
        payout = np.nan
        profit = np.nan
    return payout, profit


def recompute_payouts(df):
    if df.empty:
        return df
    df = df.copy()
    payouts, profits = [], []
    for _, r in df.iterrows():
        p, pr = compute_payout_profit(r)
        payouts.append(p); profits.append(pr)
    df["payout"] = payouts
    df["profit"] = profits
    return df


def bankroll_available(df, bankroll_start):
    if df.empty:
        return bankroll_start
    settled = df[df["status"].isin(["Gagné","Perdu","Remboursé"])]
    if settled.empty:
        return bankroll_start
    total_profit = settled["profit"].fillna(0).sum()
    return bankroll_start + total_profit


def realized_pnl(df):
    if df.empty:
        return 0.0
    settled = df[df["status"].isin(["Gagné","Perdu","Remboursé"])]
    return settled["profit"].fillna(0).sum()


def kelly_fraction(p, odds):
    if odds is None or p is None:
        return 0.0
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * p - (1 - p)) / b
    return max(f, 0.0)


def suggested_stake(strategy, bankroll_avail, odds, prob, kelly_part, cap_3pct, fixed_mode, fixed_value):
    if bankroll_avail <= 0:
        return 0.0
    if strategy == "Mise fixe":
        if fixed_mode == "Montant":
            stake = max(0.0, float(fixed_value or 0.0))
        else:
            pct = max(0.0, float(fixed_value or 0.0)) / 100.0
            stake = bankroll_avail * pct
        return min(stake, bankroll_avail)
    p = None if prob is None else float(prob)
    o = None if odds is None else float(odds)
    f_full = kelly_fraction(p, o)
    f_used = f_full * float(kelly_part)
    stake = bankroll_avail * f_used
    if cap_3pct:
        stake = min(stake, bankroll_avail * 0.03)
    return min(max(stake, 0.0), bankroll_avail)


def format_money(x):
    try:
        return f"{float(x):,.2f}".replace(",", " ").replace(".", ",")
    except:
        return str(x)


# =========================
# Sidebar (paramètres + I/O)
# =========================
with st.sidebar:
    st.header("⚙️ Paramètres")
    new_bankroll = st.number_input(
        "Bankroll initiale (€)", min_value=0.0, value=float(st.session_state.bankroll_start), step=50.0
    )
    if new_bankroll != st.session_state.bankroll_start:
        st.session_state.bankroll_start = new_bankroll
        save_config(st.session_state.bankroll_start)

    strategy = st.radio("Stratégie de mise", ["Kelly","Mise fixe"], horizontal=True, key="strategy")
    if strategy == "Kelly":
        choice = st.selectbox("Fraction de Kelly", ["25%","15%","10%"], index=0, key="kelly_choice")
        kelly_part = 0.25 if choice=="25%" else (0.15 if choice=="15%" else 0.10)
        cap_3pct = st.checkbox("Plafond 3% de bankroll", value=False, key="cap_3pct")
        fixed_mode = None; fixed_value = None
    else:
        kelly_part = 0.25; cap_3pct = False
        fixed_mode = st.radio("Type de mise fixe", ["Montant","Pourcentage"], horizontal=True, key="fixed_mode")
        if fixed_mode == "Montant":
            fixed_value = st.number_input("Montant fixe (€)", min_value=0.0, value=10.0, step=5.0, key="fixed_amount")
        else:
            fixed_value = st.number_input("Pourcentage fixe (% bankroll)", min_value=0.0, max_value=100.0, value=1.0, step=0.5, key="fixed_percent")

    st.markdown("---")
    st.subheader("💾 Sauvegarde")
    
    # Bouton de test de connexion GitHub
    if st.button("🔗 Tester connexion GitHub"):
        github_ok, github_msg = test_github_connection()
        if github_ok:
            st.success(github_msg)
        else:
            st.error(github_msg)
    
    exp_df = ensure_schema(st.session_state.bets).copy()
    if not exp_df.empty:
        exp_df["date"] = pd.to_datetime(exp_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    csv = exp_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger CSV", data=csv, file_name="paris_export.csv", mime="text/csv")

    up = st.file_uploader("📤 Importer un CSV", type=["csv"])
    if up is not None:
        try:
            df_new = pd.read_csv(up)
            df_new = ensure_schema(df_new)
            df_new = recompute_payouts(df_new)
            st.session_state.bets = df_new
            recompute_next_id()
            save_bets(st.session_state.bets)
            st.success("Import réussi.")
        except Exception as e:
            st.error(f"Échec de l'import: {e}")

    # Bouton de synchronisation manuelle
    if st.button("🔄 Recharger depuis GitHub"):
        st.session_state.bets = load_bets()
        st.session_state.bankroll_start = load_config()
        recompute_next_id()
        st.success("Données rechargées depuis GitHub")
        st.rerun()


# Recompute & persist chaque run
st.session_state.bets = ensure_schema(st.session_state.bets)
st.session_state.bets = recompute_payouts(st.session_state.bets)

# =========================
# KPIs (toujours visibles)
# =========================
st.title("🎯 Gestion de Bankroll & Paris")

bankroll_start = float(st.session_state.bankroll_start)
available = bankroll_available(st.session_state.bets, bankroll_start)
pnl = realized_pnl(st.session_state.bets)
total_locked = st.session_state.bets.loc[st.session_state.bets["status"]=="En attente","stake"].sum() if not st.session_state.bets.empty else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Bankroll initiale", f"{format_money(bankroll_start)} €")
col2.metric("💵 Bankroll disponible", f"{format_money(available)} €")
col3.metric("📈 PNL réalisé", f"{format_money(pnl)} €")
col4.metric("⏳ Mises en attente", f"{format_money(total_locked)} €")

# KPIs supplémentaires
settled = st.session_state.bets[st.session_state.bets["status"].isin(["Gagné","Perdu","Remboursé"])].copy()
total_bets = len(st.session_state.bets)
wins = settled[settled["status"]=="Gagné"]
losses = settled[settled["status"]=="Perdu"]
roi_bankroll = (pnl / bankroll_start * 100.0) if bankroll_start > 0 else 0.0

col5, col6, col7, col8, col9 = st.columns(5)
col5.metric("🎲 Nb total paris", total_bets)
col6.metric("✅ Paris réglés", settled.shape[0])
col7.metric("🏆 Gagnés", wins.shape[0])
col8.metric("❌ Perdus", losses.shape[0])
col9.metric("📊 ROI (%)", f"{roi_bankroll:.2f}")

st.markdown("---")

# =========================
# Saisie de nouveau pari (toujours visible)
# =========================
st.subheader("➕ Ajouter un nouveau pari")

# Initialisation des variables de formulaire
if "form_date" not in st.session_state: st.session_state.form_date = datetime.now().date()
if "form_event" not in st.session_state: st.session_state.form_event = ""
if "form_market" not in st.session_state: st.session_state.form_market = ""
if "form_odds" not in st.session_state: st.session_state.form_odds = 1.90
if "form_prob" not in st.session_state: st.session_state.form_prob = 55.0
if "form_notes" not in st.session_state: st.session_state.form_notes = ""
if "form_status" not in st.session_state: st.session_state.form_status = "En attente"
if "stake_input" not in st.session_state: st.session_state.stake_input = 0.0
if "follow_suggested" not in st.session_state: st.session_state.follow_suggested = True

c1, c2, c3, c4 = st.columns([1.2, 0.9, 0.9, 0.9])
st.session_state.form_date = c1.date_input("Date du pari", value=st.session_state.form_date)
st.session_state.form_event = c2.text_input("Événement", value=st.session_state.form_event, placeholder="Ex: PSG - OM")
st.session_state.form_market = c3.text_input("Marché", value=st.session_state.form_market, placeholder="Ex: 1X2, Over 2.5")
st.session_state.form_odds = c4.number_input("Cote (décimale)", min_value=1.01, value=float(st.session_state.form_odds), step=0.01, format="%.2f")

c5, c6, c7 = st.columns([0.8, 1.0, 1.2])
if st.session_state.strategy == "Kelly":
    st.session_state.form_prob = c5.number_input("Probabilité estimée (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.form_prob), step=0.5, format="%.2f")
else:
    c5.write(" ")
st.session_state.form_notes = c6.text_input("Notes (optionnel)", value=st.session_state.form_notes, placeholder="Value bet, blessure, etc.")
st.session_state.form_status = c7.selectbox("Statut initial", ["En attente","Gagné","Perdu","Remboursé"], index=["En attente","Gagné","Perdu","Remboursé"].index(st.session_state.form_status))

# Mise suggérée (live)
p_float = (st.session_state.form_prob / 100.0) if st.session_state.strategy == "Kelly" else None
kelly_choice = st.session_state.get("kelly_choice", "25%")
kelly_part = 0.25 if kelly_choice=="25%" else (0.15 if kelly_choice=="15%" else 0.10)
fixed_mode = st.session_state.get("fixed_mode", None)
fixed_value = st.session_state.get("fixed_amount", None) if fixed_mode == "Montant" else st.session_state.get("fixed_percent", None)
cap_3pct_flag = bool(st.session_state.get("cap_3pct", False))

stake_preview = suggested_stake(
    strategy=st.session_state.strategy,
    bankroll_avail=available,
    odds=st.session_state.form_odds,
    prob=p_float,
    kelly_part=kelly_part,
    cap_3pct=cap_3pct_flag,
    fixed_mode=fixed_mode,
    fixed_value=fixed_value
)

follow_suggested = st.checkbox("Mise auto = suivre la suggestion", value=st.session_state.follow_suggested, key="follow_suggested")
st.markdown(f"**Mise suggérée:** {format_money(stake_preview)} € (Bankroll dispo: {format_money(available)} €)")

if follow_suggested:
    stake_value = float(round(stake_preview, 2))
else:
    stake_value = float(st.session_state.stake_input)

c8, c9 = st.columns([0.5, 1.5])
c8.write(" ")
stake_input = c9.number_input("Mise (€)", min_value=0.0, value=stake_value, step=1.0, format="%.2f", key="stake_input")

add_clicked = st.button("🎯 Ajouter le pari", use_container_width=True, type="primary")
if add_clicked:
    stake_to_use = float(stake_input or 0.0)
    odds_val = float(st.session_state.form_odds or 0.0)
    if available <= 0:
        st.error("Bankroll disponible nulle.")
    elif stake_to_use <= 0:
        st.error("La mise doit être > 0.")
    elif stake_to_use > available + 1e-9:
        st.error("La mise dépasse la bankroll disponible.")
    elif odds_val < 1.01:
        st.error("Cote invalide.")
    else:
        d = st.session_state.form_date
        date_val = pd.to_datetime(d) if isinstance(d, (date_cls, datetime)) else pd.to_datetime(datetime.now().date())
        new_row = {
            "id": int(st.session_state.next_id),
            "date": date_val,
            "event": st.session_state.form_event,
            "market": st.session_state.form_market,
            "odds": odds_val,
            "prob": None if st.session_state.strategy!="Kelly" else float(st.session_state.form_prob or 0.0),
            "strategy": st.session_state.strategy,
            "kelly_fraction": None if st.session_state.strategy!="Kelly" else kelly_part,
            "cap_3pct": None if st.session_state.strategy!="Kelly" else cap_3pct_flag,
            "fixed_mode": None if st.session_state.strategy!="Mise fixe" else fixed_mode,
            "fixed_value": None if st.session_state.strategy!="Mise fixe" else fixed_value,
            "stake": stake_to_use,
            "status": st.session_state.form_status,
            "payout": np.nan,
            "profit": np.nan,
            "notes": st.session_state.form_notes,
        }
        st.session_state.bets = pd.concat([st.session_state.bets, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.bets = ensure_schema(st.session_state.bets)
        st.session_state.bets = recompute_payouts(st.session_state.bets)
        
        # Sauvegarde avec feedback
        if save_bets(st.session_state.bets):
            st.success("Pari ajouté et sauvegardé !")
            st.session_state.next_id = int(st.session_state.next_id) + 1
        else:
            st.error("Pari ajouté mais problème de sauvegarde GitHub")

st.markdown("---")

# =========================
# ONGLETS PRINCIPAUX
# =========================
tab1, tab2, tab3 = st.tabs(["⏳ Paris en cours", "📝 Modification des paris", "📈 Évolution"])

# =========================
# ONGLET 1: Paris en cours
# =========================
with tab1:
    st.subheader("⏳ Gestion des paris en cours")
    
    pending = st.session_state.bets[st.session_state.bets["status"]=="En attente"].copy()
    if pending.empty:
        st.info("🎉 Aucun pari en cours ! Tous vos paris ont été réglés.")
    else:
        pending = pending.sort_values(by=["date","id"], ascending=[False, False])
        
        st.markdown(f"**{len(pending)} paris en attente** | Total misé: **{format_money(pending['stake'].sum())} €**")
        
        for _, r in pending.iterrows():
            bid = int(r["id"])
            
            with st.container():
                # Affichage des informations du pari
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"""
                    **#{bid}** • {r['date'].date()} • **{r['event']}** ({r['market']})  
                    Cote: **{r['odds']:.2f}** • Mise: **{format_money(r['stake'])} €** • Retour potentiel: **{format_money(r['stake'] * r['odds'])} €**
                    """)
                    if pd.notna(r['notes']) and r['notes'].strip():
                        st.caption(f"📝 {r['notes']}")
                
                with col_actions:
                    col1, col2, col3, col4 = st.columns(4)
                    win_btn = col1.button("🏆", key=f"win_{bid}", help="Gagné")
                    lose_btn = col2.button("❌", key=f"lose_{bid}", help="Perdu") 
                    void_btn = col3.button("↩️", key=f"void_{bid}", help="Remboursé")
                    del_btn = col4.button("🗑️", key=f"del_{bid}", help="Supprimer")
                    
                if win_btn or lose_btn or void_btn or del_btn:
                    if del_btn:
                        st.session_state.bets = st.session_state.bets[st.session_state.bets["id"] != bid].reset_index(drop=True)
                        st.success(f"Pari #{bid} supprimé.")
                    else:
                        df = st.session_state.bets.set_index("id")
                        new_status = "Gagné" if win_btn else ("Perdu" if lose_btn else "Remboursé")
                        df.at[bid, "status"] = new_status
                        st.session_state.bets = df.reset_index()
                        st.success(f"Pari #{bid} marqué **{new_status}**.")
                    st.session_state.bets = recompute_payouts(ensure_schema(st.session_state.bets))
                    save_bets(st.session_state.bets)
                    st.rerun()
                
                st.divider()
        
        # Actions de groupe
        st.subheader("🔧 Actions de groupe")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("↩️ Marquer tous comme remboursés", type="secondary"):
                mask = st.session_state.bets["status"]=="En attente"
                count = mask.sum()
                st.session_state.bets.loc[mask, "status"] = "Remboursé"
                st.session_state.bets = recompute_payouts(ensure_schema(st.session_state.bets))
                save_bets(st.session_state.bets)
                st.success(f"{count} paris marqués comme remboursés.")
                st.rerun()

# =========================
# ONGLET 2: Modification des paris
# =========================
with tab2:
    st.subheader("📝 Modification et gestion des paris")
    
    if st.session_state.bets.empty:
        st.info("📭 Aucun pari enregistré pour l'instant.")
    else:
        df_show = st.session_state.bets.sort_values(by=["date","id"], ascending=[False, False]).reset_index(drop=True)
        
        st.markdown(f"**{len(df_show)} paris au total**")
        
        edited_df = st.data_editor(
            df_show,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small", disabled=True),
                "status": st.column_config.SelectboxColumn("Statut", options=["En attente","Gagné","Perdu","Remboursé"], width="small"),
                "date": st.column_config.DateColumn("Date", width="small"),
                "event": st.column_config.TextColumn("Événement", width="medium"),
                "market": st.column_config.TextColumn("Marché", width="small"),
                "odds": st.column_config.NumberColumn("Cote", format="%.2f", width="small"),
                "prob": st.column_config.NumberColumn("Prob. (%)", format="%.1f", width="small"),
                "stake": st.column_config.NumberColumn("Mise (€)", format="%.2f", width="small"),
                "payout": st.column_config.NumberColumn("Retour (€)", format="%.2f", width="small", disabled=True),
                "profit": st.column_config.NumberColumn("Profit (€)", format="%.2f", width="small", disabled=True),
                "notes": st.column_config.TextColumn("Notes", width="medium"),
            },
            disabled=["id","payout","profit","strategy","kelly_fraction","cap_3pct","fixed_mode","fixed_value"]
        )
        
        # Sauvegarde automatique des modifications
        base = st.session_state.bets.copy().set_index("id")
        allowed_cols = ["date","event","market","odds","prob","stake","status","notes"]
        for _, row in edited_df.iterrows():
            rid = int(row["id"])
            if rid in base.index:
                for c in allowed_cols:
                    base.at[rid, c] = row[c]
        st.session_state.bets = recompute_payouts(ensure_schema(base.reset_index()))
        save_bets(st.session_state.bets)
        
        # Outils de suppression
        st.subheader("🗑️ Suppression de paris")
        col1, col2 = st.columns(2)
        
        with col1:
            delete_id = st.number_input("Supprimer un pari par ID", min_value=0, value=0, step=1)
            if st.button("🗑️ Supprimer ce pari"):
                if delete_id in st.session_state.bets["id"].values:
                    st.session_state.bets = st.session_state.bets[st.session_state.bets["id"] != delete_id].reset_index(drop=True)
                    st.session_state.bets = recompute_payouts(ensure_schema(st.session_state.bets))
                    save_bets(st.session_state.bets)
                    st.success(f"Pari #{delete_id} supprimé.")
                    st.rerun()
                else:
                    st.warning("ID introuvable.")

# =========================
# ONGLET 3: Évolution
# =========================
with tab3:
    st.subheader("📈 Analyse de performance et évolution")
    
    if not settled.empty:
        # Graphique d'évolution de la bankroll
        st.subheader("📊 Évolution de la bankroll réalisée")
        
        df_curve = settled.copy()
        df_curve["date_dt"] = pd.to_datetime(df_curve["date"], errors="coerce")
        df_curve = df_curve.sort_values("date_dt")
        df_curve["cumul_pnl"] = df_curve["profit"].fillna(0).cumsum()
        df_curve["bankroll_réalisée"] = bankroll_start + df_curve["cumul_pnl"]
        
        st.line_chart(df_curve.set_index("date_dt")[["bankroll_réalisée"]], height=400)
        
        # Métriques détaillées
        st.subheader("📋 Statistiques détaillées")
        
        col1, col2, col3, col4 = st.columns(4)
        
        win_rate = (len(wins) / len(settled) * 100) if len(settled) > 0 else 0
        avg_odds_wins = wins["odds"].mean() if not wins.empty else 0
        avg_odds_losses = losses["odds"].mean() if not losses.empty else 0
        avg_stake = settled["stake"].mean() if not settled.empty else 0
        
        col1.metric("🎯 Taux de réussite", f"{win_rate:.1f}%")
        col2.metric("📊 Cote moyenne (gagnés)", f"{avg_odds_wins:.2f}")
        col3.metric("📉 Cote moyenne (perdus)", f"{avg_odds_losses:.2f}") 
        col4.metric("💰 Mise moyenne", f"{format_money(avg_stake)} €")
        
        # Analyse par mois (si assez de données)
        if len(settled) >= 5:
            st.subheader("📅 Performance mensuelle")
            
            monthly_stats = settled.copy()
            monthly_stats["month"] = pd.to_datetime(monthly_stats["date"]).dt.to_period('M')
            monthly_summary = monthly_stats.groupby("month").agg({
                "profit": "sum",
                "stake": ["sum", "count"],
                "status": lambda x: (x == "Gagné").sum()
            }).round(2)
            
            monthly_summary.columns = ["PnL", "Mise totale", "Nb paris", "Paris gagnés"]
            monthly_summary["Taux réussite (%)"] = (monthly_summary["Paris gagnés"] / monthly_summary["Nb paris"] * 100).round(1)
            monthly_summary["ROI (%)"] = (monthly_summary["PnL"] / monthly_summary["Mise totale"] * 100).round(1)
            
            st.dataframe(monthly_summary, use_container_width=True)
        
        # Graphique en camembert des résultats
        if len(settled) > 0:
            st.subheader("🥧 Répartition des résultats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                result_counts = settled["status"].value_counts()
                st.bar_chart(result_counts)
            
            with col2:
                st.markdown("**Résumé des résultats :**")
                for status, count in result_counts.items():
                    percentage = count / len(settled) * 100
                    st.write(f"• {status}: {count} paris ({percentage:.1f}%)")
    
    else:
        st.info("📊 Ajoutez et réglez quelques paris pour voir les graphiques d'évolution !")
        st.markdown("Les graphiques et analyses apparaîtront une fois que vous aurez des paris réglés (Gagnés, Perdus, ou Remboursés).")