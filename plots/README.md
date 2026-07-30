# plots/ — grafici delle run

Toolkit per generare figure a partire dalle run descritte in [STATUS_EXP.md](../STATUS_EXP.md).
Gli script sono pensati per essere guidati da riga di comando: si sceglie **cosa
filtrare**, **cosa mettere sulle righe/colonne** della griglia e **cosa distinguere
con il colore**, senza scrivere codice nuovo.

## Struttura

```
plots/
├── rtplots/            libreria
│   ├── schema.py       i campi dell'indice: titoli, formattazione, ruolo
│   ├── sources/        una fonte per convenzione, non per campagna
│   │   ├── base.py     come si legge una run: parte comune
│   │   ├── current.py  convenzione corrente (rebuttal, rt-ppo-ablations, …)
│   │   └── paper.py    run del paper (tag, default, alias delle metriche)
│   ├── envbase.py      n_steps/batch_size/n_epochs letti da run/config/ppo/
│   ├── paths.py        percorsi (cache in /storage, output in plots/output)
│   ├── index.py        indice dei run: metadati W&B -> parquet in cache
│   ├── curves.py       curve di eval (evaluations.npz locali, fallback W&B) + aggregazione
│   ├── select.py       filtri e conteggi di copertura
│   ├── figure.py       FigureSpec + pipeline unica: selezione -> figura
│   ├── tikz.py         export .tex (pgfplots), un file per pannello
│   ├── selection.py    selezioni salvate (lettura, scrittura, migrazione)
│   ├── labels.py       nomi delle serie (wPPO-U / wPPO-BH, setting, ...)
│   ├── rules.py        legge style.toml (le regole scritte a mano)
│   ├── style.py        traduce le regole in rcParams e colori
│   ├── grid.py         disegno della griglia di pannelli
│   └── webui/          selettore: api.py (logica) + server.py (HTTP)
├── scripts/            eseguibili
│   ├── build_index.py  costruisce/aggiorna la cache dei metadati
│   ├── selector.py     selettore interattivo delle run (server locale)
│   ├── list_runs.py    che run/seed esistono per una data combinazione
│   ├── plot_curves.py  curve di una metrica vs step (--metric)
│   └── plot_final.py   prestazione finale a barre
├── style.toml          regole dei grafici: si modifica a mano
├── selector/           pagina del selettore (html/css/js, nessuna dipendenza)
├── tests/              test senza rete (`.venv/bin/python -m pytest plots/tests -q`)
└── output/             figure generate (ignorata da git)
```

**Una sola pipeline.** Riga di comando e selettore costruiscono lo stesso oggetto
— `FigureSpec` — e lo passano a `rtplots.figure`: la stessa selezione da' la
stessa figura da tutte e due le strade, baseline e ordine dei colori compresi.
Prima erano due implementazioni parallele e divergevano.

## Selettore interattivo

```bash
.venv/bin/python plots/scripts/selector.py     # http://127.0.0.1:8770
```

Pagina locale con una sezione di filtri per ogni dimensione di ablation
(environment, famiglia, ω, setting, IS, critic, fresh advantages, adaptive LR,
sampling, stato, campagna) più l'intervallo dei seed. Nessuna casella selezionata
in un gruppo = quella dimensione non è filtrata. Mostra in tempo reale quante run
restano, la tabella di copertura (evidenziando le combinazioni con meno seed) e
l'anteprima del grafico nello stesso stile della figura finale.

Ogni dimensione ha un **operatore**, nella tendina accanto al titolo:

| operatore | significato | valori selezionabili |
|---|---|---|
| **è** | uguale a quel valore | uno |
| **non è** | diverso da quel valore | uno |
| **fra** (default) | uno dei valori scelti | molti |
| **non fra** | nessuno dei valori scelti | molti |

Con «non è» e «non fra» le pillole scelte diventano rosse e barrate: sono
esclusioni. La stringa di «Copia filtri» usa la stessa sintassi degli script
(`family!=PPO,SAC`), quindi resta incollabile in `--filter`.

Il numero accanto a ogni valore è **quante configurazioni distinte
resterebbero scegliendolo** (combinazioni di iperparametri, non run: i seed
della stessa configurazione contano una volta sola, altrimenti una
combinazione con dieci seed sembrerebbe dieci volte più presente di una con un
seed solo). Tiene conto di tutti gli altri filtri attivi ma non di quelli
della sua stessa dimensione (come nelle ricerche a faccette), e i valori che
porterebbero a zero configurazioni sono sbiaditi. Con un operatore negativo il
numero risponde alla domanda giusta — quante configurazioni resterebbero
**escludendo** quel valore.

Le selezioni si salvano **con un nome** e restano nella lista «Selezioni salvate»:
un click le riapplica (filtri, seed e impostazioni di griglia), la ✕ le elimina.
I file stanno in `/storage/fis1/plots_cache/selections/<nome>.json`; l'ultima
salvata o riaperta è anche `selection.json`, quella che gli script usano con
`--runs-file`.

### Ritoccare le serie a mano

Sotto l'anteprima c'è l'elenco delle **serie** disegnate, con il loro colore. Si
clicca una serie e si cambiano **nome** e **colore** (fra quelli di `[palette]`
in `style.toml`): il ritocco vale in tutti i pannelli in cui quella serie
compare, perché la voce di legenda è una sola. Tutto il resto — spessore, tratto,
banda, font, legenda — continua ad arrivare da `style.toml`.

I ritocchi vivono **nella figura**: entrano nella `FigureSpec`, quindi si salvano
con la selezione, si riaprono uguali e finiscono nel `.tex` esportato. Sono
indicizzati per l'etichetta *di partenza*, così rinominare non fa perdere il
collegamento; se cambi i filtri e quella serie non c'è più, il ritocco viene
ignorato in silenzio.

Quando una scelta è definitiva, **«Copia regola per style.toml»** dà il blocco
`[[series]]` già scritto — `match`, colore e nome — da incollare nel file. Da lì
in poi vale per tutte le figure, non solo per questa.

Rinominando a mano il nome scelto vince anche nel `.tex`: la macro `latex` della
regola non sopravvive al rinomino, altrimenti l'anteprima e il sorgente
direbbero due cose diverse. Se vuoi una macro, scrivila in `style.toml`.

Due pulsanti scaricano direttamente la figura mostrata:

- **JPEG** — la stessa figura a 300 dpi, sfondo bianco, per slide e bozze.
- **LaTeX** — i sorgenti `pgfplots`: **un `.tex` per pannello**, con i dati dentro
  (`\addplot table {...}`, bande come `\path[fill=...]`), non un'immagine inclusa.
  Una griglia m×n scarica uno zip con m×n file; un pannello solo scarica il `.tex`.
  Sotto l'anteprima compare lo snippet `figure` già montato — un `\input` per
  pannello, in `subfigure` quando sono più di uno — copiato anche negli appunti.

I pannelli si compongono in LaTeX, non in matplotlib: per questo l'export dà i
riquadri separati e non un unico `.tex` con m×n assi. Colori, ordine delle serie e
legende sono quelli calcolati sull'intera griglia, quindi un pannello estratto è
identico a come si vedeva nell'anteprima. Serve `\usepackage{pgfplots,amsmath}`
(più `subcaption` per le griglie) e `\pgfplotsset{compat=1.18}`.

Il nome del file riassume le dimensioni fissate nella selezione, più quelle che
distinguono il pannello (`rt_hopper_rtppo_s1_env_hopper_window_2.tex`). Il file
viene servito da `/download/<token>/<nome>` con
`Content-Disposition: attachment`, quindi lo scarica il browser sulla macchina da
cui guardi la pagina. Una copia resta comunque in `plots/output/`, utile se stai
usando il browser interno di VSCode (che blocca i download) — il percorso compare
sotto l'anteprima insieme a un link manuale.

Due modi per passare la selezione agli script:

- **«Salva selezione»** scrive `/storage/fis1/plots_cache/selection.json`, che
  contiene la figura intera (run, righe/colonne, colori, banda, metrica,
  baseline) e non solo l'elenco delle run; poi `plot_curves.py --runs-file`
  (senza argomento usa quel file) rifa' esattamente quella figura. Quello che si
  passa da riga di comando scavalca la selezione, il resto lo eredita. In chat
  basta chiedere «fai il plot della selezione».
- **«Copia filtri»** copia la stringa `--filter env=Hopper-v5 family=RT-PPO …`.

L'indice viene riletto a ogni interrogazione: dopo un `build_index.py` le nuove
run compaiono senza riavviare il server.

## Dati

- **Curve di eval**: `<campagna>/logs/<run_id>/evaluations.npz` in `/storage/fis1/…`,
  scritti da `EvalCallback` (100 punti, 50 episodi per punto). È la fonte primaria:
  locale, completa, veloce.
- **Metadati** (env, famiglia, ω, setting, IS, critic, seed, stato…): dalla config
  W&B, messa in cache in `/storage/fis1/plots_cache/run_index.parquet`.
- **Diagnostiche** (KL, ESS, clip fraction, …): solo da W&B, con cache per run.

Il `setting` viene letto dai tag W&B (`setting1/2/3`); se mancano è dedotto da
`n_steps`/`batch_size` rispetto ai valori base di `run/config/ppo/ppo_<env>.yaml`.

Prima esecuzione (o dopo una nuova campagna):

```bash
.venv/bin/python plots/scripts/build_index.py          # incrementale
.venv/bin/python plots/scripts/build_index.py --force  # riscarica tutto
```

## Due fonti: campagne nuove e run del paper

L'indice unisce **otto progetti W&B**, distinti dalla colonna `source`:

| `source` | progetti | contenuto |
|---|---|---|
| `wandb` | `rebuttal`, `rt-ppo-ablations` | campagne attuali (sottospazi W1, W3, baseline SAC/TD3, ablation) |
| `paper` | `forzaroma-rt-ppo-{ant,hopper,reacher,swimmer,walker}`, `erghosting-rt-ppo-half-cheetah` | run del paper (P1, P2, B2, B3) |
| `geppo-orig` | `rt-ppo-ablations`, tag `geppo_original` | run della codebase originale di GePPO |

Un progetto per environment: è una convenzione dei lanci del paper, l'env sta
comunque nella config. Per HalfCheetah vale `erghosting-…`, non
`forzaroma-rt-ppo-half-cheetah`.

**Trappola da conoscere.** `wandb.Api().runs(<progetto>)` restituisce
`run.config == {}` per le run del paper: i parametri arrivano **solo** dopo
`run.load(force=True)`. Chi sonda quei progetti senza `load()` conclude a torto
che siano senza metadati e che serva ricostruirli dal nome del run. Non serve:
`build_index.py` chiama sempre `load()` e le run del paper finiscono nell'indice
con lo stesso schema delle altre — `env`, `family`, `window`, `is_type`, `setting`,
`opc`, `seed`, `fresh_adv`, `sampling` tutti popolati, zero valori mancanti.

La fonte `geppo-orig` sta **nello stesso progetto** delle campagne attuali, quindi
non si riconosce dal progetto ma dal tag `geppo_original` (`claim_tags`): la sua
config è piatta (`env_kwargs/env_name`, `runner_kwargs/M`, `ac_kwargs/eps_ppo`)
invece di stare sotto `experiment`. `family = GePPO-original`,
`window = runner_kwargs/M`, `setting` vuoto (configurazioni tunate, fuori dalla
griglia 1/2/3), `ablation = geppo_original` — quindi `ablation=none` continua a
isolare P1+P2+B2+B3 e le figure esistenti non cambiano. `fresh_adv`, `opc`,
`sampling`, `seq`, `is_type` restano vuoti: nella codebase originale non esistono
come opzioni. Nessun `.npz` locale: le curve vengono dalla history W&B.

Tutto ciò che distingue una fonte dall'altra sta in `rtplots/sources/`: un file
per *convenzione* (`current.py`, `paper.py`) con i tag che contano, i default
per le chiavi assenti e gli alias delle metriche. `index.py` non sa nulla di
paper e campagne: scarica, unisce, mette in cache. Quindi **un progetto nuovo
sulle convenzioni correnti è una riga in `PROJECTS`** (`current.py`), e solo una
convenzione diversa richiede un file nuovo.

Cosa il codice aggiunge sopra la config, e perché:

- **`setting`** — le run del paper non hanno i tag `setting1/2/3` ma
  `wppo_1`, `wppo_3`, `wppo_3_v2`, che valgono rispettivamente setting **1**,
  **2**, **3**: la numerazione dei tag *non* è quella dei setting. La mappa è
  `SETTING_TAGS` in `sources/paper.py`. È ridondante ma esplicita: su tutte le
  run che hanno il tag, la deduzione dai soli `n_steps`/`batch_size` arriva allo
  stesso valore.
- **`sampling`** — l'asse del campionamento dei minibatch a tre valori
  (`random`, `balanced`, `weighted`), come `batch_sampling` nei config. Le run
  più vecchie hanno il booleano `balanced_batches` e vengono mappate qui: prima
  l'indice leggeva solo il booleano, quindi le run nuove restavano senza valore.
- **`ablation`** — i tag `clip_range_ablation`, `clip_range_ablation2`, `tuned_1`,
  `tuned_2` marcano run fuori dai sottospazi di STATUS_EXP. Filtrare
  `ablation=none` isola esattamente P1+P2+B2+B3 (1860 run, 162 celle RT-PPO e 24
  PPO tutte a 10 seed).
- **`adaptive_lr`** — assente dalla config del paper perché precedente alla
  feature: viene messo a `False`, che è il valore vero, non "ignoto".
- **`epoch_mult`** — le epoche in multipli del valore base dell'environment
  (`n_epochs` di `run/config/ppo/ppo_<env>.yaml`): 1, 2, 4, 8. È ciò che separa
  le due baseline PPO di STATUS_EXP: **B2 è `epoch_mult=1`**, **B3 è
  `epoch_mult=2,4,8`**. Senza, le quattro celle PPO di un environment
  differiscono solo per `n_epochs` e in figura collassano in un'unica serie.
  Il valore base cambia per environment (Reacher 5, Ant/Hopper/Swimmer 10,
  HalfCheetah/Walker2d 20) e **si legge da `run/config/ppo/ppo_<env>.yaml`**
  (`envbase.py`), non è ricopiato: se un config cambia, cambia anche la
  classificazione. `epoch_mult` è confrontabile fra environment, `n_epochs` no.
- **doppioni** — i rilanci identici sono stati ripuliti su W&B, quindi l'indice
  non scarta più niente da solo. Se ricompaiono run omonime nello stesso progetto
  (dove il nome codifica tutti i parametri più il seed) `build_index.py` lo
  stampa come avviso: gonfiano `n_seeds` e restringono la banda senza cambiare la
  media, ma toglierle è una decisione, non un automatismo.

**Ant-v5 gira su 3M step**, tutti gli altri environment su 1M: `total_timesteps`
lo dice ed è filtrabile. Vale per l'intero progetto Ant, non per un solo setting,
quindi Ant non è confrontabile a parità di step con gli altri in nessuna figura
che non tagli l'asse x.

**Curve.** Le run del paper non hanno `.npz` locali: passano sempre dalla history
W&B. I `run_id` sono unici per progetto e non per entity, quindi sia la richiesta
sia il file di cache sono qualificati col progetto
(`<progetto>__<run_id>__<metrica>.parquet`). Le curve hanno 100–300 punti, sotto
il tetto di `samples=2000`: i valori sono esatti e `scan_history` darebbe lo stesso
risultato 20× più lentamente. Per non pagare una richiesta per run a ogni figura:

```bash
.venv/bin/python plots/scripts/prefetch_curves.py      # ~25 MB, ~10 minuti
```

**Non ancora allineate: le diagnostiche.** Le run del paper usano chiavi diverse
(`diagnostics_ess/post_naive_mean`, `diagnostics_kl/post_naive_mean`, …) da quelle
del catalogo in `metrics.py` (`diagnostics_ess/final_naive_ess_mean`, …), e non
hanno le controparti `initial_*`. Le metriche `train/*` e `rollout/*` invece
coincidono. Finché la corrispondenza semantica non è confermata **non c'è alcun
alias attivo**: le candidate stanno in `UNCONFIRMED_ALIASES`
(`sources/paper.py`), promuoverne una è una scelta sperimentale.

Chiedere una diagnostica su una selezione che contiene run del paper non le fa
più sparire in silenzio dalla media: il caricamento stampa quante run mancano,
per quale progetto e perché.

## Cosa si può plottare

`--metric` (CLI) e la tendina **«cosa plottare»** (selettore) usano lo stesso
catalogo, definito in `rtplots/metrics.py`:

| gruppo | esempi | fonte |
|---|---|---|
| Eval | mean return, lunghezza episodio | `.npz` locali, immediato |
| Rollout | `rollout/ep_rew_mean`, `time/fps` | W&B |
| Ottimizzazione | loss, clip fraction, explained variance, learning rate | W&B |
| Diagnostiche IS | KL, ESS, varianza dei ratio, `|ratio|` medio | W&B |

`python plots/scripts/plot_curves.py --list-metrics` stampa l'elenco completo;
qualsiasi altra chiave loggata su W&B è comunque accettata. Le metriche W&B
costano una richiesta per run: vengono scaricate in parallelo e messe in cache su
disco, quindi solo la prima volta è lenta (nel selettore c'è un tetto per evitare
attese di minuti su selezioni enormi).

## Filtri

Ogni script accetta `--filter chiave=valore …` (in AND):

| forma | significato |
|---|---|
| `env=Hopper-v5` (o `env=Hopper`) | uguaglianza |
| `window=2,4` | uno di questi valori |
| `setting!=1` | diverso |
| `window>=4` | confronto numerico (`>`, `>=`, `<`, `<=`) |
| `opc=false` | booleani |

Colonne disponibili: `family` (PPO/RT-PPO/GePPO/SAC/TD3), `env`, `window`, `setting`,
`is_type` (N/BH), `opc`, `fresh_adv`, `adaptive_lr`, `sampling`, `seq`, `seed`,
`n_steps`, `batch_size`, `n_minibatch`, `n_epochs`, `epoch_mult`, `clip_range`,
`lr`, `gamma`, `total_timesteps`, `campaign`, `state`, `tags`,
`source` (wandb/paper), `project`, `ablation`.

Le baseline PPO si separano per numero di epoche con `epoch_mult`:

```bash
# solo il PPO base (B2)
--filter family=PPO epoch_mult=1
# le baseline a epoche moltiplicate (B3), una serie per moltiplicatore
--filter family=PPO epoch_mult=2,4,8 --hue epoch_mult --label-fields epoch_mult
```

Per default vengono usati solo i run `finished` (`--state any` per includere gli altri).

## Esempi

Cosa c'è a disposizione:

```bash
.venv/bin/python plots/scripts/list_runs.py --by family env window setting
.venv/bin/python plots/scripts/list_runs.py --filter family=RT-PPO env=Hopper-v5 \
    --by setting window is_type --check-curves
```

Figura in stile paper (righe = setting, colonne = ω, colori = IS, ω anche in legenda):

```bash
.venv/bin/python plots/scripts/plot_curves.py \
    --filter family=RT-PPO env=Hopper-v5 \
    --rows setting --cols window --hue is_type --label-fields is_type window \
    --row-captions auto --titles off --legend-loc "lower right" \
    --name hopper_rtppo
```

**Serie automatiche**: senza `--hue` ogni configurazione diversa presente nella
selezione diventa una curva a sé, sovrapposta alle altre nello stesso pannello;
l'unica aggregazione è sui seed. Le dimensioni assegnate a righe/colonne e quelle
ridondanti (p.es. `adaptive_lr`, determinato da `family`) non vengono ripetute in
legenda. Con `--hue` si forza la scelta a mano: se così facendo qualche dimensione
che varia resta senza colore, lo script avvisa che quelle configurazioni finiscono
mediate insieme.

`--hue` decide i **colori** (restano gli stessi in tutti i pannelli), `--label-fields`
decide cosa compare in **legenda**: mettendoci anche `window` si ottiene
«ωPPO-BH: ω = 4» come nella figura del paper. Con `--baseline` si sovrappongono in
nero una o più baseline (`--baseline family=SAC env=Hopper-v5`): le famiglie si
elencano con la virgola — `--baseline family=PPO,GePPO-original` — e si
distinguono per tratteggio (continua, `-.`, `:`, vedi `[lines].baseline_styles`).
Nel selettore sono pillole, quindi se ne accendono quante se ne vuole.

Confronto GePPO vs RT-PPO sui quattro environment W&B:

```bash
.venv/bin/python plots/scripts/plot_curves.py \
    --filter family=GePPO,RT-PPO setting=1 is_type=N \
    --cols env --hue family window --share col --name geppo_vs_rtppo
```

Diagnostica (ESS) e prestazione finale:

```bash
.venv/bin/python plots/scripts/plot_curves.py --metric diagnostics_ess/final_naive_ess_mean \
    --filter family=RT-PPO env=Hopper-v5 setting=1 --cols window --name hopper_ess

.venv/bin/python plots/scripts/plot_final.py \
    --filter family=RT-PPO setting=2 --panels env --x window --hue is_type \
    --name final_setting2
```

## Stile

Le regole stanno in **[`style.toml`](style.toml)**, un file solo, dichiarativo, da
modificare a mano: colori, spessori, tratto, banda, font, legenda, nomi delle
serie e opzioni `pgfplots`. Vale sia per l'anteprima sia per l'export `.tex` —
quello che vedi è quello che scarichi — e viene riletto a ogni figura, quindi si
salva e si ridisegna senza riavviare il selettore. Se il file non è TOML valido
il selettore lo dice nel terminale e usa i default, invece di non disegnare.

Le sezioni:

| sezione | cosa decide |
|---|---|
| `[figure]` | dimensione dei pannelli, etichette degli assi, scala x, font |
| `[lines]` | spessore, opacità della banda, tipo di banda, smoothing, baseline |
| `[legend]` | dove sta, posizione, colonne, cornice, corpo del font |
| `[palette]` | i colori usati dalle serie che nessuna regola copre |
| `[[series]]` | una regola per curva: `match` + colore, spessore, tratto, nome |
| `[latex]` | solo nel `.tex`: opzioni dell'asse, preambolo, macro dei nomi |

Ogni blocco `[[series]]` dice «se la serie ha questi valori, disegnala così».
Vince la **prima regola che combacia**, quindi le più specifiche vanno in alto:

```toml
[[series]]
match = { family = "RT-PPO", is_type = "BH" }
color = "#648FFF"
style = "solid"
name  = 'ωPPO-BH'     # in anteprima
latex = '\bhppo'      # nel .tex: la macro del paper
```

`name` e `latex` sono la stessa serie scritta nei due mondi: matplotlib non sa
disegnare `\bhppo`, quindi l'anteprima mostra `name` e nel `.tex` finisce
`\addlegendentry{\bhppo}`. Le macro elencate in `[latex].macros` vengono ricopiate
come commento in testa a ogni `.tex`, per ricordarsi cosa serve nel documento.

Quello che il file decide sono i **default**: `--band`, `--smooth`, `--panel-size`
e gli altri argomenti degli script continuano a scavalcarli figura per figura, e
così fanno i controlli del selettore.

Resta invece fissato nel codice (`rtplots/style.py`), perché non è mai cambiato:

- media sui seed + **banda ombreggiata** attorno alla curva;
- font serif, nomi degli algoritmi in monospace, ω come lettera greca;
- pannelli con box completo, tick esterni, nessuna griglia;
- caption per riga in stile paper con `--row-captions auto` («(b) Fixed $N_\mathcal{B}$ setting.»);
- output in `png` + `pdf` a 300 dpi (`--formats`).

Naming: `RT-PPO` + `is_type=N` → `ωPPO-U`, `is_type=BH` → `ωPPO-BH`; il suffisso
`Off` indica `on_policy_critic=false`. Con `--raw-names` si usano i nomi del codice.

## Note

- Le figure vanno in `plots/output/`; per i report di un lancio usare
  `--outdir experiments/expNNN/x/data`.
- Cache e artefatti stanno in `/storage/fis1/plots_cache` (override con `RTPLOTS_CACHE`).
- `--dump-csv` salva accanto alla figura i dati aggregati usati per disegnarla.
