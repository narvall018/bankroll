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


def github_file_exists(file_path):
    """Vérifie si un fichier existe sur GitHub"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    return response.status_code == 200


def load_from_github(file_path):
    """Charge un fichier depuis GitHub"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        content = response.json()["content"]
        decoded_content = base64.b64decode(content).decode('utf-8')
        return decoded_content
    return None


def save_to_github(file_path, content, commit_message="Update from Streamlit app"):
    """Sauvegarde un fichier sur GitHub"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    # Vérifier si le fichier existe déjà pour récupérer le SHA
    response = requests.get(url, headers=headers)
    sha = None
    if response.status_code == 200:
        sha = response.json()["sha"]
    
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
    return response.status_code in [200, 201]


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
    # Créer les fichiers par défaut s'ils n'existent pas sur GitHub
    if not github_file_exists(CONFIG_FILE):
        default_config = json.dumps({"bankroll_start": 1000.0})
        save_to_github(CONFIG_FILE, default_config, "Initialize config file")
    
    if not github_file_exists(BETS_FILE):
        empty_df = pd.DataFrame(columns=COLUMNS)
        csv_content = empty_df.to_csv(index=False)
        save_to_github(BETS_FILE, csv_content, "Initialize bets file")


def load_config():
    try:
        content = load_from_github(CONFIG_FILE)
        if content:
            cfg = json.loads(content)
            return float(cfg.get("bankroll_start", 1000.0))
    except:
        pass
    return 1000.0


def save_config(bankroll_start):
    try:
        config_data = json.dumps({"bankroll_start": float(bankroll_start)})
        save_to_github(CONFIG_FILE, config_data, "Update bankroll config")
    except:
        pass


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
    try:
        content = load_from_github(BETS_FILE)
        if content:
            from io import StringIO
            df = pd.read_csv(StringIO(content))
            return ensure_schema(df)
    except:
        pass
    return pd.DataFrame(columns=COLUMNS)


def save_bets(df: pd.DataFrame):
    try:
        out = ensure_schema(df).copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        csv_content = out.to_csv(index=False)
        save_to_github(BETS_FILE, csv_content, "Update bets data")
    except:
        pass


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

# Vérifier la configuration GitHub
if not GITHUB_TOKEN or not GITHUB_REPO:
    st.error("⚠️ Configuration GitHub manquante. Vérifiez vos secrets Streamlit.")
    st.stop()

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
    st.header("Paramètres")
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
    st.subheader("Sauvegarde")
    exp_df = ensure_schema(st.session_state.bets).copy()
    if not exp_df.empty:
        exp_df["date"] = pd.to_datetime(exp_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    csv = exp_df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les paris (CSV)", data=csv, file_name="paris_export.csv", mime="text/csv")

    up = st.file_uploader("Importer un CSV de paris", type=["csv"])
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
# KPIs
# =========================
bankroll_start = float(st.session_state.bankroll_start)
available = bankroll_available(st.session_state.bets, bankroll_start)
pnl = realized_pnl(st.session_state.bets)
total_locked = st.session_state.bets.loc[st.session_state.bets["status"]=="En attente","stake"].sum() if not st.session_state.bets.empty else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Bankroll initiale", f"{format_money(bankroll_start)} €")
col2.metric("Bankroll disponible", f"{format_money(available)} €")
col3.metric("PNL réalisé", f"{format_money(pnl)} €")
col4.metric("Mises en attente", f"{format_money(total_locked)} €")

# =========================
# Saisie réactive du pari
# =========================
st.markdown("### Ajouter un pari")

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
st.markdown(f"Mise suggérée: {format_money(stake_preview)} € (Bankroll dispo: {format_money(available)} €)")

if follow_suggested:
    stake_value = float(round(stake_preview, 2))
else:
    stake_value = float(st.session_state.stake_input)

c8, c9 = st.columns([0.5, 1.5])
c8.write(" ")
stake_input = c9.number_input("Mise (€)", min_value=0.0, value=stake_value, step=1.0, format="%.2f", key="stake_input")

add_clicked = st.button("Ajouter le pari", use_container_width=True)
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
        save_bets(st.session_state.bets)
        st.success("Pari ajouté.")
        st.session_state.next_id = int(st.session_state.next_id) + 1

# =========================
# Paris en cours (actions rapides)
# =========================
st.markdown("### Paris en cours")
pending = st.session_state.bets[st.session_state.bets["status"]=="En attente"].copy()
if pending.empty:
    st.info("Aucun pari en cours.")
else:
    pending = pending.sort_values(by=["date","id"], ascending=[False, False])
    for _, r in pending.iterrows():
        bid = int(r["id"])
        c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5,1.0,1.0,1.0,1.0,1.2,1.0])
        c1.write(f"[{bid}] {r['date'].date()} — {r['event']} ({r['market']})")
        c2.write(f"Cote: {r['odds']:.2f}")
        c3.write(f"Mise: {format_money(r['stake'])} €")
        win_btn = c4.button("Gagné", key=f"win_{bid}")
        lose_btn = c5.button("Perdu", key=f"lose_{bid}")
        void_btn = c6.button("Remboursé", key=f"void_{bid}")
        del_btn = c7.button("Supprimer", key=f"del_{bid}")
        if win_btn or lose_btn or void_btn or del_btn:
            if del_btn:
                st.session_state.bets = st.session_state.bets[st.session_state.bets["id"] != bid].reset_index(drop=True)
                st.success(f"Pari {bid} supprimé.")
            else:
                df = st.session_state.bets.set_index("id")
                new_status = "Gagné" if win_btn else ("Perdu" if lose_btn else "Remboursé")
                df.at[bid, "status"] = new_status
                st.session_state.bets = df.reset_index()
                st.success(f"Pari {bid} marqué {new_status}.")
            st.session_state.bets = recompute_payouts(ensure_schema(st.session_state.bets))
            save_bets(st.session_state.bets)
            st.rerun()

# =========================
# Mes paris (éditeur)
# =========================
st.markdown("### Mes paris")
if st.session_state.bets.empty:
    st.info("Aucun pari pour l'instant.")
else:
    df_show = st.session_state.bets.sort_values(by=["date","id"], ascending=[False, False]).reset_index(drop=True)
    edited_df = st.data_editor(
        df_show,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": st.column_config.SelectboxColumn("Statut", options=["En attente","Gagné","Perdu","Remboursé"], width="small"),
            "date": st.column_config.DateColumn("Date"),
            "odds": st.column_config.NumberColumn("Cote", format="%.2f"),
            "prob": st.column_config.NumberColumn("Prob. (%)", format="%.2f"),
            "stake": st.column_config.NumberColumn("Mise (€)", format="%.2f"),
            "payout": st.column_config.NumberColumn("Retour (€)", format="%.2f"),
            "profit": st.column_config.NumberColumn("Profit (€)", format="%.2f"),
        },
        disabled=["id","payout","profit","strategy","kelly_fraction","cap_3pct","fixed_mode","fixed_value"]
    )
    base = st.session_state.bets.copy().set_index("id")
    allowed_cols = ["date","event","market","odds","prob","stake","status","notes"]
    for _, row in edited_df.iterrows():
        rid = int(row["id"])
        if rid in base.index:
            for c in allowed_cols:
                base.at[rid, c] = row[c]
    st.session_state.bets = recompute_payouts(ensure_schema(base.reset_index()))
    save_bets(st.session_state.bets)

    cA, cB = st.columns(2)
    with cA:
        delete_id = st.number_input("Supprimer un pari par ID", min_value=0, value=0, step=1)
        if st.button("Supprimer (ID)"):
            if delete_id in st.session_state.bets["id"].values:
                st.session_state.bets = st.session_state.bets[st.session_state.bets["id"] != delete_id].reset_index(drop=True)
                st.session_state.bets = recompute_payouts(ensure_schema(st.session_state.bets))
                save_bets(st.session_state.bets)
                st.success(f"Pari {delete_id} supprimé.")
            else:
                st.warning("ID introuvable.")
    with cB:
        if st.button("Marquer tous 'En attente' comme 'Remboursé'"):
            mask = st.session_state.bets["status"]=="En attente"
            st.session_state.bets.loc[mask, "status"] = "Remboursé"
            st.session_state.bets = recompute_payouts(ensure_schema(st.session_state.bets))
            save_bets(st.session_state.bets)
            st.success("Tous les paris en attente marqués comme remboursés.")

# =========================
# Graphiques d'évolution
# =========================
st.markdown("### Graphiques d'évolution")
settled = st.session_state.bets[st.session_state.bets["status"].isin(["Gagné","Perdu","Remboursé"])].copy()
if not settled.empty:
    df_curve = settled.copy()
    df_curve["date_dt"] = pd.to_datetime(df_curve["date"], errors="coerce")
    df_curve = df_curve.sort_values("date_dt")
    df_curve["cumul_pnl"] = df_curve["profit"].fillna(0).cumsum()
    df_curve["bankroll_réalisée"] = bankroll_start + df_curve["cumul_pnl"]
    st.line_chart(df_curve.set_index("date_dt")[["bankroll_réalisée"]], height=280)
else:
    st.info("Ajoute et règle quelques paris pour afficher la courbe de bankroll réalisée.")

# KPIs supplémentaires
total_bets = len(st.session_state.bets)
wins = settled[settled["status"]=="Gagné"]
losses = settled[settled["status"]=="Perdu"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nb de paris", total_bets)
c2.metric("Réglés", settled.shape[0])
c3.metric("Gagnés", wins.shape[0])
c4.metric("Perdus", losses.shape[0])

roi_bankroll = (pnl / bankroll_start * 100.0) if bankroll_start > 0 else 0.0
c5.metric("ROI sur bankroll (%)", f"{roi_bankroll:.2f}")
