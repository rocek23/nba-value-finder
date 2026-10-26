"""Confronto bookmaker: quote per prop, margini, scarto dalla media e copertura dei provider."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, shell
from ..labels import ITALIAN_BOOKS_WITHOUT_API, fmt_age, pct

STATUS_IT = {"ok": "✅ attivo", "manuale": "✍️ manuale", "non_configurato": "⚪ non configurato", "errore": "❌ errore",
             "non_disponibile": "⛔ non disponibile"}


def _book_summary(rows: pd.DataFrame) -> pd.DataFrame:
    piv = rows.pivot_table(index=["prop_id", "book", "bookmaker"], columns="lato", values="quota", aggfunc="max").reset_index()
    for c in ("Over", "Under"):
        if c not in piv:
            piv[c] = np.nan
    both = piv["Over"].notna() & piv["Under"].notna()
    piv["margine"] = np.where(both, 1 / piv["Over"] + 1 / piv["Under"] - 1, np.nan)
    for c in ("Over", "Under"):
        piv[f"scarto_{c}"] = piv[c] / piv.groupby("prop_id")[c].transform("mean") - 1
        piv[f"best_{c}"] = piv[c] == piv.groupby("prop_id")[c].transform("max")
    age = rows.groupby("book")["eta_min"].min()
    stale = rows.groupby("book")["obsoleta"].mean()
    out = piv.groupby(["book", "bookmaker"]).agg(
        props=("prop_id", "nunique"), margine_medio=("margine", "mean"),
        scarto_over=("scarto_Over", "mean"), scarto_under=("scarto_Under", "mean"),
        best_over=("best_Over", "mean"), best_under=("best_Under", "mean"),
    ).reset_index()
    out["quote_obsolete"] = out["book"].map(stale)
    out["quote_piu_recente_min"] = out["book"].map(age)
    return out


def render() -> None:
    ctx = shell.begin("Confronto bookmaker", "Quote a confronto, media generale tra bookmaker e copertura dei provider")
    b, board, rows = ctx.bundle, ctx.board, ctx.rows
    t1, t2, t3 = st.tabs(["Confronto per prop", "Riepilogo bookmaker", "Copertura provider"])

    with t1:
        if rows.empty:
            st.info("Nessuna quota disponibile per il confronto.")
        else:
            g = board.groups.sort_values(["giocatore", "mercato", "linea"])
            labels = {f"{r.giocatore} · {r.mercato} {r.linea:g} ({r.evento}) — {r.n_book} book": r.prop_id for r in g.itertuples()}
            q = st.text_input("Cerca giocatore", key="bk_q")
            opts = [k for k in labels if q.lower() in k.lower()] if q else list(labels)
            if not opts:
                st.info("Nessuna prop corrisponde alla ricerca.")
            else:
                pid = labels[st.selectbox("Prop", opts, key="bk_prop")]
                d = rows[rows["prop_id"] == pid]
                grp = g[g["prop_id"] == pid].iloc[0]
                piv = d.pivot_table(index=["book", "bookmaker"], columns="lato", values="quota", aggfunc="max").reset_index()
                for c in ("Over", "Under"):
                    if c not in piv:
                        piv[c] = np.nan
                both = piv["Over"].notna() & piv["Under"].notna()
                piv["Margine"] = np.where(both, 100 * (1 / piv["Over"] + 1 / piv["Under"] - 1), np.nan)
                piv["P Over senza margine"] = np.where(both, 100 * (1 / piv["Over"]) / (1 / piv["Over"] + 1 / piv["Under"]), np.nan)
                piv["Aggiornata"] = piv["book"].map(d.groupby("book")["eta_min"].min()).map(fmt_age)
                piv["Manuale"] = piv["book"].map(d.groupby("book")["manuale"].max()).map({True: "sì", False: ""})
                c1, c2 = st.columns([3, 2])
                c1.dataframe(piv.drop(columns="book").rename(columns={"bookmaker": "Bookmaker"}).style.format(
                    {"Over": "{:.2f}", "Under": "{:.2f}", "Margine": "{:.1f}%", "P Over senza margine": "{:.1f}%"}, na_rep="—"),
                    hide_index=True, width="stretch")
                c2.markdown("**Media generale tra bookmaker**")
                m = pd.DataFrame({
                    "": ["Quota media", "Quota migliore", "Prob. implicita media (grezza)", "Quota equa del modello"],
                    "Over": [f"{grp['quota_over_media']:.2f}", f"{grp['quota_over_max']:.2f} ({grp['book_over_max']})", pct(grp["prob_impl_over_media"]),
                             "—" if pd.isna(grp["quota_equa_over"]) else f"{grp['quota_equa_over']:.2f}"],
                    "Under": [f"{grp['quota_under_media']:.2f}", f"{grp['quota_under_max']:.2f} ({grp['book_under_max']})", "—",
                              "—" if pd.isna(grp["quota_equa_under"]) else f"{grp['quota_equa_under']:.2f}"],
                })
                c2.dataframe(m, hide_index=True, width="stretch")
                c2.caption(f"P Over di mercato senza margine (media): {pct(grp['p_over_mercato'])} su {int(grp['n_book_due_lati'])} book con entrambi i lati · "
                           f"P Over modello: {pct(grp['p_over_modello'])}.")
                f1, f2 = st.columns(2)
                for col, side in ((f1, "Over"), (f2, "Under")):
                    gg = piv[["bookmaker", side]].dropna().rename(columns={side: "quota"})
                    fair = grp["quota_equa_over"] if side == "Over" else grp["quota_equa_under"]
                    col.plotly_chart(charts.fig_odds_by_book(gg, None if pd.isna(fair) else float(fair),
                                                             float(gg["quota"].mean()) if len(gg) else None, side), width="stretch")

    with t2:
        if rows.empty:
            st.info("Nessuna quota disponibile.")
        else:
            s = _book_summary(rows)
            out = pd.DataFrame({
                "Bookmaker": s["bookmaker"], "Props quotate": s["props"], "Margine medio": 100 * s["margine_medio"],
                "Scarto Over vs media": 100 * s["scarto_over"], "Scarto Under vs media": 100 * s["scarto_under"],
                "Migliore Over (%prop)": 100 * s["best_over"], "Migliore Under (%prop)": 100 * s["best_under"],
                "Quote obsolete": 100 * s["quote_obsolete"], "Più recente (min)": s["quote_piu_recente_min"],
            }).sort_values("Margine medio")
            st.dataframe(out.style.format({"Margine medio": "{:.1f}%", "Scarto Over vs media": "{:+.1f}%", "Scarto Under vs media": "{:+.1f}%",
                                           "Migliore Over (%prop)": "{:.0f}%", "Migliore Under (%prop)": "{:.0f}%", "Quote obsolete": "{:.0f}%",
                                           "Più recente (min)": "{:.0f}"}, na_rep="—"), hide_index=True, width="stretch")
            st.plotly_chart(charts.fig_book_margin(s.dropna(subset=["margine_medio"])), width="stretch")
            st.caption("Margine = 1/Over + 1/Under − 1 (solo dove il bookmaker quota entrambi i lati). Scarto = quota del book rispetto alla media tra book sulla stessa prop.")

    with t3:
        st.markdown("**Provider dei dati**")
        prov = pd.DataFrame([{"Provider": p.name, "Stato": STATUS_IT.get(p.status, p.status), "Quote": p.n_quotes, "Dettaglio": p.detail} for p in b.providers])
        if prov.empty:
            st.info("Nessun provider interrogato: premi «Aggiorna dati live» (modalità live).")
        else:
            st.dataframe(prov, hide_index=True, width="stretch")
        st.markdown("**Bookmaker italiani senza copertura API verificata**")
        st.dataframe(pd.DataFrame({"Bookmaker": ITALIAN_BOOKS_WITHOUT_API, "Stato": "Non disponibile via API",
                                   "Come ottenerli": "Importa le quote da CSV (Impostazioni e API)"}), hide_index=True, width="stretch")
        st.caption("Nessuna API verificata copre le player props NBA di questi operatori (vedi docs/API_PROVIDERS.md). "
                   "Non vengono mai estratte con scraping: usa il CSV con l'orario di rilevazione.")
