"""Step 3: Interactive web app.  Run with:  streamlit run app.py"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LEGO Set Value Predictor", page_icon="🧱", layout="centered")
model = joblib.load("models/model.joblib")
meta = json.loads(Path("models/metadata.json").read_text())
data = pd.read_csv("data/processed/lego_model_data.csv")
ERR = meta["error_pct_points"] / 100
LICENSED_THEMES = set(data.loc[data["is_licensed"] == 1, "theme_grouped"])


def size_group(pieces: int) -> int:
    # Same cutoffs as the dataset's size groups (1 = biggest, 4 = smallest).
    return 1 if pieces >= 1204 else 2 if pieces >= 660 else 3 if pieces >= 340 else 4


def features(theme, pieces, rrp, minifigs, years_held, licensed):
    return pd.DataFrame([{
        "log_pieces": np.log(pieces), "log_rrp": np.log(rrp), "price_per_piece": rrp / pieces,
        "minifigs": minifigs, "minifigs_per_100_pieces": 100 * minifigs / pieces,
        "is_licensed": int(licensed), "size_group": size_group(pieces),
        "years_held": years_held, "theme_grouped": theme,
    }])[meta["features"]]


st.title("🧱 LEGO Set Value Predictor")
st.caption("Predicts how fast a sealed LEGO set grows in value after release, using only "
           "information known on release day. Trained on eBay resale prices for 1,800+ sets.")

mode = st.radio("Mode", ["Describe a set", "Look up a set from the data"], horizontal=True)
lo_h, hi_h = meta["years_held_range"]
years = st.slider("Years after release", int(np.ceil(lo_h)), min(15, int(hi_h)), 5)

if mode == "Describe a set":
    c1, c2 = st.columns(2)
    theme = c1.selectbox("Theme", meta["themes"], index=meta["themes"].index("Star Wars"))
    licensed = c2.checkbox("Licensed franchise (Star Wars, Marvel, etc.)", value=theme in LICENSED_THEMES)
    pieces = c1.number_input("Pieces", 10, 8000, 800)
    rrp = c2.number_input("Retail price (USD)", 5.0, 1000.0, 79.99)
    minifigs = c1.number_input("Minifigures", 0, 30, 4)
    row = features(theme, pieces, rrp, minifigs, years, licensed)
else:
    labels = (data["id"].astype(str) + " · " + data["name"].astype(str)).sort_values()
    choice = st.selectbox("Set", labels)
    rec = data[data["id"].astype(str) == choice.split(" · ")[0]].iloc[0]
    rrp = rec["rrp_usd"]
    row = features(rec["theme_grouped"], rec["pieces"], rrp, rec["minifigs"], years, rec["is_licensed"])
    actual = f"sold for **${rec['value_new_2015']:.2f}** in 2015" if rec["value_new_2015"] > 0 else "no 2015 sales recorded"
    st.write(f"**{rec['name']}** ({rec['theme']}, {int(rec['release_year'])}) · {int(rec['pieces'])} pieces · "
             f"retail ${rrp:.2f} · {actual}")

cagr = float(model.predict(row)[0])
value = rrp * (1 + cagr) ** years
low, high = rrp * (1 + cagr - ERR) ** years, rrp * (1 + cagr + ERR) ** years

c1, c2 = st.columns(2)
c1.metric("Predicted annual growth", f"{cagr:.1%}")
c2.metric(f"Estimated sealed value after {years} years", f"${value:,.0f}", f"{value / rrp - 1:+.0%} vs. retail")
st.caption(f"Typical error range: ${low:,.0f} to ${high:,.0f}. The model is better at ranking sets "
           "(which will grow faster than others) than at exact dollar values. Not investment advice.")

curve = pd.DataFrame({"years": range(0, 16)})
curve["Predicted value ($)"] = [
    rrp if y == 0 else rrp * (1 + float(model.predict(row.assign(years_held=max(y, lo_h)))[0])) ** y
    for y in curve["years"]]
st.line_chart(curve.set_index("years"))

with st.expander("How well does the model work?"):
    st.image("reports/forward_test_quintiles.png",
             caption="Sets released in 2013-2014, which the model never saw, grouped by predicted growth. "
                     "Bars show how fast each group actually grew by 2019.")
