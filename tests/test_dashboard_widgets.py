"""Nota di affidabilita' del dato e componenti HTML dello stile dashboard."""

from nba_vf.ui import widgets as W
from nba_vf.ui.best import build_picks, data_note, quote_age_text

from test_tracker import generous_rows


def test_quote_age_text_states_what_is_known_and_what_is_not():
    assert quote_age_text(5, True, 30) == "aggiornata 5 min fa"
    assert "da verificare" in quote_age_text(5, False, 30)  # orario del bookmaker non dichiarato
    assert quote_age_text(90, True, 30).startswith("obsoleta")
    assert quote_age_text(float("nan"), True) == "da verificare"


def test_data_note_never_invents_numbers():
    assert data_note(30, 5, True, 30) == "Storico: 30 partite · Quota aggiornata 5 min fa"
    assert data_note(float("nan"), 5, True).startswith("Storico: da verificare")


def test_every_pick_carries_a_data_note():
    rows, _ = generous_rows()
    picks = build_picks(rows.assign(obsoleta=False), 10.0, include_low=True, fresh_only=False)
    assert len(picks) and picks["nota_dato"].str.startswith("Storico:").all()
    assert picks["nota_dato"].str.contains("Quota").all()


def test_html_helpers_escape_text():
    assert "<script>" not in W.player_cell("<script>alert(1)</script>")
    assert "<b>" not in W.pill("<b>x</b>")
    assert 'title="a&quot;b"' in W.pill("x", title='a"b')
    assert W.initials("Stephen Curry") == "SC" and W.initials("") == "?"
    assert "Affidabilità alta" in W.reliability_pill("Alta", long=True)
