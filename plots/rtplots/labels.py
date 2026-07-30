"""Nomenclatura: dalle colonne dell'indice alle etichette usate nei grafici.

Il nome dell'algoritmo si costruisce qui; il modo di scrivere il *valore* di una
dimensione (ω, setting, epoche...) sta in `schema.py`, unico per pagina web,
titoli dei pannelli e legenda.

Convenzioni (STATUS_EXP.md):
  - RT-PPO con IS=N  == wPPO-U  del paper
  - RT-PPO con IS=BH == wPPO-BH del paper
  - suffisso "Off" = critic off-policy (on_policy_critic=False)
  - setting1 = batch size fisso, setting2 = numero di minibatch fisso (Fixed N_B),
    setting3 = H/w steps
"""
from __future__ import annotations

from . import rules as R
from . import schema
from .schema import SETTING_CAPTIONS, SETTING_NAMES, SETTING_NAMES_HTML  # noqa: F401
from .style import mathtt

# Nome "paper" per famiglia + tipo di IS
PAPER_FAMILY = {
    ("RT-PPO", "N"): "ωPPO-U",
    ("RT-PPO", "BH"): "ωPPO-BH",
    ("GePPO", "N"): "GePPO",
    ("GePPO", "BH"): "GePPO",
}


def family_name(row, paper: bool = True, latex: bool = False) -> str:
    """Nome dell'algoritmo (senza iperparametri), gia' pronto da stampare.

    Se una regola `[[series]]` di style.toml copre questa serie, il nome scritto
    li' vince e viene usato **tale e quale**: e' il punto in cui si decide come
    si chiamano le curve, sia in anteprima (`name`) sia nel .tex (`latex`).
    """
    rule = R.rule_for(row)
    chosen = rule.get("latex") if latex else rule.get("name")
    if chosen:
        return str(chosen)
    fam = row.get("family")
    if fam in ("PPO", "SAC", "TD3"):
        return fam
    if fam == "GePPO":
        return "GePPO"
    if fam == "GePPO-original":
        return "GePPO-orig"
    is_type = row.get("is_type") or "N"
    if paper:
        return PAPER_FAMILY.get((fam, is_type), f"{fam}-{is_type}")
    return f"{fam} IS={is_type}"


def series_label(row, fields=("family", "opc", "window"), paper: bool = True,
                 latex: bool = False) -> str:
    """Etichetta di una serie, in stile figura di riferimento.

    Esempio: '$\\mathtt{\\omega PPO\\text{-}BH}$ Off: $\\omega = 4$'
    `fields` decide quali informazioni entrano nell'etichetta; l'ordine e' sempre
    quello di dichiarazione in `schema.FIELDS`, non quello di `fields`.
    """
    fields = set(fields)
    # un nome scritto a mano in style.toml si usa come sta scritto; solo quelli
    # ricavati dal codice passano per mathtt()
    written = (R.rule_for(row) or {}).get("latex" if latex else "name")
    head = str(written) if written else mathtt(family_name(row, paper=paper))
    # il critic off-policy e' un suffisso del nome, non una voce a se'
    if "opc" in fields and row.get("family") in ("RT-PPO", "GePPO") \
            and row.get("opc") is False:
        head += " Off"
    extras = []
    for f in schema.FIELDS:
        if f.col not in fields or f.col in ("family", "opc"):
            continue
        # con i nomi del paper l'IS e' gia' dentro ωPPO-U / ωPPO-BH
        if f.col == "is_type" and paper:
            continue
        bit = schema.legend_bit(f.col, row.get(f.col))
        if bit:
            extras.append(bit)
    return head + (": " + ", ".join(extras) if extras else "")


def panel_title(field: str, value, paper: bool = True) -> str:
    """Titolo di riga/colonna della griglia."""
    return schema.panel_title(field, value, paper)
