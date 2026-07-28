"""Test del toolkit dei grafici (niente rete, niente W&B).

    .venv/bin/python -m pytest plots/tests -q

Coprono le parti che il refactor ha unificato e che prima erano duplicate:
formattazione dei campi, scelta automatica delle serie, lettura di una run per
fonte, migrazione delle selezioni salvate e handler del selettore.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtplots import labels as L  # noqa: E402
from rtplots import rules, schema, selection, tikz  # noqa: E402
from rtplots.figure import (FigureSpec, Series, _apply_overrides,  # noqa: E402
                            auto_hue, merged_dims, split_panels)
from rtplots.sources import current, paper  # noqa: E402
from rtplots.webui import api  # noqa: E402


# --- indice finto -----------------------------------------------------------

def make_index() -> pd.DataFrame:
    rows = []
    for family, window, is_type, epoch_mult in [
        ("RT-PPO", 4.0, "N", 1.0), ("RT-PPO", 4.0, "BH", 1.0),
        ("PPO", 1.0, None, 1.0), ("PPO", 1.0, None, 2.0),
    ]:
        for seed in (1, 2):
            rows.append(dict(
                run_id=f"{family}-{is_type}-{int(epoch_mult)}-{seed}",
                name=f"{family}_{seed}", state="finished", project="rebuttal",
                source="wandb", ablation=None, campaign="c1", env="Hopper-v5",
                family=family, window=window, setting=1.0, is_type=is_type,
                opc=True, fresh_adv=False, adaptive_lr=False, sampling="balanced",
                seq=False, epoch_mult=epoch_mult, seed=seed, n_epochs=10,
                total_timesteps=1e6,
            ))
    return pd.DataFrame(rows)


# --- schema -----------------------------------------------------------------

def test_html_value_non_confonde_uno_con_vero():
    # in Python 1.0 == True: epoch_mult=1 non deve diventare "si'"
    assert schema.html_value("epoch_mult", 1.0) == "×1"
    assert schema.html_value("opc", True) == "sì"
    assert schema.html_value("window", 1.0).startswith("ω = 1")
    assert schema.html_value("setting", 2.0) == "2 · Fixed N<sub>B</sub>"
    assert schema.html_value("env", "Hopper-v5") == "Hopper"


def test_panel_title_e_legenda_dallo_stesso_campo():
    assert schema.panel_title("window", 4.0) == r"$\omega = 4$"
    assert schema.legend_bit("window", 4.0) == r"$\omega = 4$"
    # omega=1 non aggiunge nulla in legenda
    assert schema.legend_bit("window", 1.0) is None
    assert schema.legend_bit("mai_visto", 3) is None


def test_epoch_mult_e_una_dimensione_di_serie():
    # senza, le baseline PPO a epoche moltiplicate finiscono mediate insieme
    assert "epoch_mult" in schema.SERIES_FIELDS
    assert "sampling" in schema.SERIES_FIELDS
    assert "balanced" not in schema.SERIES_FIELDS


# --- scelta delle serie -----------------------------------------------------

def test_auto_hue_separa_le_baseline_per_epoche():
    ppo = make_index().query("family == 'PPO'")
    assert auto_hue(ppo) == ["epoch_mult"]
    assert merged_dims(ppo, ["epoch_mult"]) == []
    assert merged_dims(ppo, ["family"]) == ["epoch_mult"]


def test_auto_hue_ignora_le_dimensioni_su_righe_e_colonne():
    df = make_index()
    assert "is_type" not in auto_hue(df, exclude=("is_type", None))


# --- lettura delle run ------------------------------------------------------

def fake_run(config, tags=(), rid="abc"):
    return SimpleNamespace(id=rid, name="run", group=None, state="finished",
                           tags=list(tags), created_at="2026-01-01", config=config)


def test_sampling_dalla_convenzione_nuova_e_da_quella_vecchia():
    src = current.SOURCE
    nuovo = src.row(fake_run({"experiment": {"batch_sampling": "random"}}), "rebuttal")
    assert nuovo["sampling"] == "random"
    vecchio = src.row(fake_run({"experiment": {"balanced_batches": True}}), "rebuttal")
    assert vecchio["sampling"] == "balanced"
    assert src.row(fake_run({"experiment": {}}), "rebuttal")["sampling"] is None


def test_convenzione_corrente_vale_su_tutti_i_suoi_progetti():
    """Aggiungere un progetto e' una riga in PROJECTS, non una fonte nuova."""
    from rtplots import sources

    assert "rt-ppo-ablations" in sources.ALL_PROJECTS
    assert sources.for_project("rt-ppo-ablations") is current.SOURCE
    exp = {"env_name": "Hopper-v5", "window_size": 2, "n_steps": 1024,
           "batch_size": 64, "n_epochs": 10, "geppo_clip": True,
           "batch_sampling": "balanced", "adaptive_lr": True, "fresh_adv": True,
           "on_policy_critic": False, "weight_type": "naive",
           "dir_name": "/storage/fis1/01_geppo_rtppo_grid"}
    row = current.SOURCE.row(
        fake_run({"experiment": exp}, tags=["01_geppo_rtppo_grid", "b21b0ed", "geppo",
                                            "setting2"]),
        "rt-ppo-ablations")
    assert row["source"] == "wandb" and row["family"] == "GePPO"
    assert row["setting"] == 2 and row["sampling"] == "balanced"
    assert row["adaptive_lr"] is True and row["is_type"] == "N"
    assert row["campaign"] == "01_geppo_rtppo_grid"


def test_run_del_paper_setting_dai_tag_e_adaptive_lr_falso():
    src = paper.SOURCE
    exp = {"env_name": "Hopper-v5", "window_size": 4, "n_steps": 512,
           "batch_size": 64, "n_epochs": 20}
    row = src.row(fake_run({"experiment": exp}, tags=["wppo_3"]), "forzaroma-rt-ppo-hopper")
    assert row["setting"] == 2          # la numerazione dei tag non e' quella dei setting
    assert row["adaptive_lr"] is False  # la chiave non c'e', ma il valore vero e' False
    assert row["source"] == "paper"
    assert row["epoch_mult"] == 2       # 20 epoche su una base di 10


def test_setting_dedotto_dai_parametri_quando_manca_il_tag():
    exp = {"env_name": "Hopper-v5", "window_size": 4, "n_steps": 2048 // 4,
           "batch_size": 64}
    assert paper.SOURCE.setting(exp, [], "Hopper-v5") == 3
    exp["n_steps"] = 2048
    assert paper.SOURCE.setting(exp, [], "Hopper-v5") == 1
    exp["batch_size"] = 256
    assert paper.SOURCE.setting(exp, [], "Hopper-v5") == 2


def test_diagnostiche_del_paper_non_mappate_a_caso():
    # la corrispondenza semantica non e' confermata: meglio nessun dato che dati sbagliati
    assert paper.SOURCE.metric_key("diagnostics_ess/final_naive_ess_mean") is None
    # eval e train invece coincidono
    assert paper.SOURCE.metric_key("eval/mean_reward") == "eval/mean_reward"
    assert current.SOURCE.metric_key("diagnostics_kl/kl_mean") == "diagnostics_kl/kl_mean"


# --- spec e selezioni -------------------------------------------------------

def test_spec_round_trip():
    spec = FigureSpec(rows="setting", cols="window", hue=["is_type"], smooth=3)
    spec.baseline.family = "PPO"
    spec.baseline.epochs = "follow_window"
    back = FigureSpec.from_dict(spec.to_dict())
    assert back.rows == "setting" and back.hue == ["is_type"] and back.smooth == 3
    assert back.baseline.family == "PPO" and back.baseline.epochs == "follow_window"


def test_migrazione_selezione_v1(tmp_path):
    vecchia = {
        "name": "prova", "slug": "prova", "saved_at": "2026-07-27T14:10:00",
        "n_runs": 2, "filter_args": ["env=Hopper-v5"],
        "dims": {"env": ["Hopper-v5"], "balanced": ["True"]},
        "seeds": {"min": None, "max": None}, "excluded": [],
        "grid": {"rows": "setting", "cols": "window", "hue": ["balanced"],
                 "band": "std", "smooth": 3, "metric": "eval/mean_reward",
                 "baseline": {"family": "PPO", "epochs": "follow_window"}},
        "run_ids": ["a", "b"],
    }
    path = tmp_path / "vecchia.json"
    path.write_text(json.dumps(vecchia))
    spec, data = selection.spec_from(path)
    assert spec.rows == "setting" and spec.band == "std" and spec.smooth == 3
    assert spec.hue == ["sampling"]                  # balanced -> sampling
    assert data["dims"]["sampling"] == ["balanced"]  # e anche nei filtri
    assert spec.baseline.family == "PPO"
    assert spec.run_ids == ["a", "b"]
    assert spec.state == "any"


# --- handler del selettore --------------------------------------------------

def test_query_conta_run_configurazioni_e_copertura():
    df = make_index()
    res = api.query(df, {"dims": {"family": ["RT-PPO"]}})
    assert res["n_runs"] == 4
    assert res["n_configs"] == 2                     # N e BH
    assert res["filter_args"] == ["family=RT-PPO"]
    assert "IS" in res["coverage"]["columns"]
    # i conteggi di una dimensione ignorano i filtri di quella dimensione
    assert res["counts"]["family"]["PPO"] == 4


def test_operatori_dei_filtri():
    df = make_index()
    q = lambda dims: api.query(df, {"dims": dims})
    # "è" e "fra" tengono i valori scelti, "non è" e "non fra" li escludono
    assert q({"family": {"op": "is", "values": ["PPO"]}})["n_runs"] == 4
    assert q({"family": {"op": "is_not", "values": ["PPO"]}})["n_runs"] == 4
    assert q({"is_type": {"op": "in", "values": ["N", "BH"]}})["n_runs"] == 4
    assert q({"is_type": {"op": "not_in", "values": ["N", "BH"]}})["n_runs"] == 4
    # la lista nuda (selezioni salvate prima degli operatori) vale come "fra"
    assert q({"family": ["PPO"]})["n_runs"] == 4


def test_filtri_negativi_tradotti_in_sintassi_cli():
    df = make_index()
    args = api.query(df, {"dims": {"family": {"op": "not_in", "values": ["PPO"]}}})["filter_args"]
    assert args == ["family!=PPO"]
    # con l'operatore positivo, selezionare tutti i valori non filtra nulla...
    tutti = {"op": "in", "values": ["PPO", "RT-PPO"]}
    assert api.query(df, {"dims": {"family": tutti}})["filter_args"] == []
    # ...mentre escluderli tutti e' una richiesta vera e va riportata
    tutti_no = {"op": "not_in", "values": ["PPO", "RT-PPO"]}
    res = api.query(df, {"dims": {"family": tutti_no}})
    assert res["filter_args"] == ["family!=PPO,RT-PPO"] and res["n_runs"] == 0


def test_conteggi_di_un_operatore_negativo_dicono_quanto_resta_escludendo():
    df = make_index()
    counts = api.query(df, {"dims": {"family": {"op": "not_in", "values": []}}})["counts"]
    # 8 run in tutto: escludendo PPO (4 run) ne restano 4
    assert counts["family"]["PPO"] == 4
    positivi = api.query(df, {"dims": {}})["counts"]
    assert positivi["family"]["PPO"] == 4      # per caso coincide: 4 con, 4 senza
    assert positivi["is_type"]["N"] == 2 and positivi["is_type"]["BH"] == 2


def test_query_senza_filtri_non_elenca_filtri():
    assert api.query(make_index(), {})["filter_args"] == []


def test_spec_dalla_pagina():
    df = make_index()
    payload = {"dims": {}, "grid": {"rows": "setting", "cols": "", "hue": ["is_type"],
                                    "band": "iqr", "smooth": 7, "metric": "train/loss",
                                    "baseline": {"family": "PPO", "epochs": "2"}}}
    spec = api.spec_from_payload(payload, df)
    assert spec.rows == "setting" and spec.cols is None
    assert spec.band == "iqr" and spec.metric == "train/loss"
    assert spec.baseline.family == "PPO" and spec.baseline.epochs == "2"
    assert spec.state == "any" and len(spec.run_ids) == len(df)
    assert spec.row_captions == "auto"


def test_esclusioni_tolgono_run_ma_non_righe_di_copertura():
    df = make_index()
    victim = df.run_id.iloc[0]
    res = api.query(df, {"dims": {}, "excluded": [victim]})
    assert res["n_excluded"] == 1
    rows = res["coverage"]["rows"]
    assert any(r["n_kept"] < r["n_runs"] for r in rows)   # la riga resta, spuntata


# --- regole scritte a mano (style.toml) -------------------------------------

@pytest.fixture
def regole(tmp_path, monkeypatch):
    """Sostituisce style.toml con un file di prova, per il tempo del test."""
    def write(text: str):
        path = tmp_path / "style.toml"
        path.write_text(text)
        monkeypatch.setattr(rules, "RULES_FILE", path)
        rules.load(force=True)
        return path
    yield write
    rules.load(force=True)          # rimette in cache quello vero


def test_vince_la_prima_regola_che_combacia(regole):
    regole("""
[[series]]
match = { family = "RT-PPO", is_type = "BH" }
color = "#111111"
[[series]]
match = { family = "RT-PPO" }
color = "#222222"
""")
    assert rules.rule_for({"family": "RT-PPO", "is_type": "BH"})["color"] == "#111111"
    assert rules.rule_for({"family": "RT-PPO", "is_type": "N"})["color"] == "#222222"
    assert rules.rule_for({"family": "GePPO"}) == {}
    # match su una colonna che la serie non ha: non combacia, non esplode
    assert rules.rule_for({"is_type": "BH"}) == {}


def test_match_confronta_numeri_non_stringhe(regole):
    regole('[[series]]\nmatch = { window = 4 }\ncolor = "#333333"\n')
    # nell'indice window e' float: 4.0 deve combaciare con il 4 scritto nel file
    assert rules.rule_for({"window": 4.0})["color"] == "#333333"
    assert rules.rule_for({"window": 8.0}) == {}


def test_nomi_delle_serie_presi_dal_file(regole):
    regole("""
[[series]]
match = { family = "RT-PPO", is_type = "N" }
name = 'wPPO-U'
latex = '\\uppo'
""")
    row = {"family": "RT-PPO", "is_type": "N", "window": 4.0}
    # il nome scritto a mano si usa tale e quale, senza passare da mathtt()
    assert L.series_label(row, fields=("family",)) == "wPPO-U"
    assert L.series_label(row, fields=("family",), latex=True) == r"\uppo"
    # quello che varia fra i pannelli resta attaccato dopo il nome
    assert L.series_label(row, fields=("family", "window")).startswith("wPPO-U: ")


def test_senza_regola_il_nome_resta_quello_del_codice(regole):
    regole("")
    row = {"family": "RT-PPO", "is_type": "BH"}
    assert L.series_label(row, fields=("family",)) == r"$\omega\mathtt{PPO\text{-}BH}$"


def test_file_rotto_non_rompe_i_grafici(regole, capsys):
    regole("questo non e' TOML [[[")
    assert rules.series_rules() == []
    assert rules.get("lines", "width") == 1.4      # scorta di FALLBACK
    assert "non e' TOML valido" in capsys.readouterr().out


def test_opzioni_pgfplots_in_coda_alle_altre():
    code = ("\\begin{axis}[\ntick pos=left,\nxmin=0, xmax=1\n]\n"
            "\\addplot table {};\n\\end{axis}\n")
    out = tikz._add_axis_options(code, ["width=\\figurewidth", " ", "ymin=-11"])
    axis = out.split("]\n")[0]
    # le opzioni scritte a mano vengono dopo, quindi vincono su quelle generate
    assert axis.endswith("xmin=0, xmax=1,\nwidth=\\figurewidth,\nymin=-11\n")
    assert tikz._add_axis_options(code, []) == code


# --- ritocchi fatti a mano nell'anteprima -----------------------------------

def test_ritocco_rinomina_e_ricolora_in_tutti_i_pannelli():
    series = make_series()
    styles = dict(series.styles)
    styles["A"] = {**styles["A"], "latex": r"\uppo"}
    agg, order, styles, matches = _apply_overrides(
        series.agg, series.order, styles, {"A": {"family": "RT-PPO"}, "B": {}},
        {"A": {"name": "ωPPO-U: stale", "color": "#785EF0"}})
    assert order == ["ωPPO-U: stale", "B"]
    assert "A" not in set(agg.label)                 # rinominata ovunque, non a tratti
    assert styles["ωPPO-U: stale"]["color"] == "#785EF0"
    assert styles["ωPPO-U: stale"]["width"] == 1.4   # il resto resta da style.toml
    # rinominando vince il nome scelto: la macro della regola non sopravvive,
    # altrimenti il .tex direbbe una cosa e l'anteprima un'altra
    assert styles["ωPPO-U: stale"]["latex"] is None
    assert matches["ωPPO-U: stale"] == {"family": "RT-PPO"}


def test_ritocco_solo_colore_non_tocca_il_nome():
    series = make_series()
    agg, order, styles, _ = _apply_overrides(
        series.agg, series.order, dict(series.styles), {"A": {}, "B": {}},
        {"A": {"color": "#000000"}})
    assert order == ["A", "B"] and styles["A"]["color"] == "#000000"


def test_ritocco_di_una_serie_sparita_viene_ignorato():
    series = make_series()
    agg, order, styles, _ = _apply_overrides(
        series.agg, series.order, dict(series.styles), {"A": {}, "B": {}},
        {"Z": {"name": "mai vista"}})
    assert order == ["A", "B"] and "mai vista" not in styles


def test_regola_da_incollare_in_style_toml():
    snippet = api.rule_snippet(
        "ωPPO-U: stale", {"family": "RT-PPO", "is_type": "N", "window": 4.0,
                          "fresh_adv": False, "opc": None},
        {"color": "#DC267F", "latex": "\\uppo"})
    assert snippet.splitlines()[0] == "[[series]]"
    # i float dell'indice tornano interi, i booleani sono TOML, i None spariscono
    assert ('match = { family = "RT-PPO", is_type = "N", window = 4, '
            'fresh_adv = false }') in snippet
    # stringa letterale: il backslash della macro non va raddoppiato
    assert "latex = '\\uppo'" in snippet
    assert "name  = 'ωPPO-U: stale'" in snippet


# --- export LaTeX -----------------------------------------------------------

def make_series() -> Series:
    """Griglia 2x2 (env × window) gia' aggregata, con due serie per pannello."""
    rows = []
    for env in ("Hopper-v5", "Swimmer-v5"):
        for window in (2.0, 4.0):
            for label in ("A", "B"):
                for step in (0.0, 1.0):
                    rows.append(dict(step=step, mean=1.0, lo=0.0, hi=2.0, n_seeds=3,
                                     label=label, env=env, window=window))
    styles = {lab: {"color": color, "width": 1.4, "style": "solid",
                    "band_alpha": 0.18, "latex": None}
              for lab, color in (("A", "#648FFF"), ("B", "#DC267F"))}
    return Series(sel=pd.DataFrame(), agg=pd.DataFrame(rows), order=["A", "B"],
                  styles=styles, hue=["family"], matches={"A": {}, "B": {}},
                  baseline=None, ylabel="y", metric_label="Mean return", merged=[])


def test_split_panels_da_un_pannello_per_riquadro():
    spec = FigureSpec(rows="env", cols="window")
    panels = split_panels(make_series(), spec)
    assert len(panels) == 4
    # da solo un pannello non ha piu' righe/colonne su cui spezzarsi
    assert all(p.spec.rows is None and p.spec.cols is None for p in panels)
    assert [p.slug for p in panels] == [
        "env_hopper_window_2", "env_hopper_window_4",
        "env_swimmer_window_2", "env_swimmer_window_4"]
    # la caption e' mathtext, che e' gia' LaTeX valido
    assert panels[0].caption == r"Hopper, $\omega = 2$"
    # ogni pannello vede solo i propri dati, ma l'ordine delle serie e' quello
    # calcolato sull'intera griglia: i colori restano gli stessi
    assert set(panels[0].series.agg.env) == {"Hopper-v5"}
    assert panels[0].series.order == ["A", "B"]


def test_split_panels_senza_griglia_da_un_pannello_solo():
    panels = split_panels(make_series(), FigureSpec())
    assert len(panels) == 1 and panels[0].slug == "" and panels[0].caption == ""


def test_snippet_latex_monta_una_subfigure_per_pannello():
    panels = split_panels(make_series(), FigureSpec(rows="env", cols="window"))
    names = [f"fig_{p.slug}.tex" for p in panels]
    snippet = api.tex_snippet("fig", panels, names, ncol=2, caption="Didascalia.")
    assert snippet.count("\\input{figures/fig_env_") == 4
    assert snippet.count("\\begin{subfigure}") == 4
    assert "subcaption" in snippet                   # serve per le subfigure
    assert snippet.count("\\\\") == 1                # a capo solo a fine prima riga
    assert "\\label{fig:fig}" in snippet


def test_snippet_latex_di_un_pannello_solo_non_usa_subfigure():
    panels = split_panels(make_series(), FigureSpec())
    snippet = api.tex_snippet("fig", panels, ["fig.tex"], ncol=1, caption="Didascalia.")
    assert "subfigure" not in snippet
    assert "\\input{figures/fig.tex}" in snippet


def test_tikzplotlib_importabile_con_gli_alias():
    # tikzplotlib 0.10.1 non regge matplotlib 3.6+/numpy 2 senza gli alias di
    # rtplots.tikz: se questo test fallisce, l'export LaTeX e' rotto.
    # importorskip non va bene: importerebbe la libreria *senza* gli alias, cioe'
    # proprio il caso che qui deve funzionare
    if importlib.util.find_spec("tikzplotlib") is None:
        pytest.skip("tikzplotlib non installato")
    assert tikz.unavailable_reason() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
