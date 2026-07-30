/* Selettore run: filtri -> /api/query (copertura) e /api/preview (grafico). */

// risoluzione dell'anteprima (vedi PREVIEW_DPI in webui/api.py): converte i px
// scelti nella casella "dimensione" nei pollici che FigureSpec.panel_size vuole
const PREVIEW_DPI = 110;
const DEFAULT_SIZE_PX = 500;

const state = {
  // colonna -> {op, values}: l'operatore decide se i valori scelti si tengono
  // (è / fra) o si escludono (non è / non fra), e se se ne puo' scegliere piu' di uno
  dims: {},
  seeds: { min: null, max: null },
  // run tolte a mano dalla tabella di copertura: restano visibili in tabella
  // ma non entrano nel grafico
  excluded: new Set(),
  // run pinnate dalla copertura: entrano nel grafico anche quando i filtri
  // correnti le taglierebbero fuori, e sopravvivono ai cambi di filtro —
  // servono a confrontare configurazioni che nessun filtro terrebbe insieme
  pinned: new Set(),
  // hue vuoto = automatico: una serie per ogni configurazione che varia
  // baseline: curva nera sovrapposta, cercata nell'indice completo e non nella
  // selezione. epoch_mult sceglie la variante di PPO (B2 = x1, B3 = x2/x4/x8)
  // senza filtrare le curve principali, che hanno un loro epoch_mult.
  grid: { rows: "", cols: "", hue: [], band: "ci95", smooth: 5, metric: "eval/mean_reward",
          panel_size: [DEFAULT_SIZE_PX / PREVIEW_DPI, DEFAULT_SIZE_PX / PREVIEW_DPI],
          baseline: { family: [], epochs: "" } },
  // ritocchi a mano delle serie: etichetta di partenza -> {name, color}
  series_overrides: {},
  series: [],            // ultimo elenco di serie disegnate
  seriesSel: null,       // quale e' selezionata nell'elenco
  palette: [],
  meta: null,
};

const $ = (sel) => document.querySelector(sel);
let queryTimer = null;
let selectionItems = [];   // ultimo elenco di selezioni salvate, per ridisegnarlo

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

/* --- filtri --------------------------------------------------------------- */

const pillIndex = {};   // colonna -> valore -> {input, label, countEl}
const opIndex = {};     // colonna -> <select> dell'operatore

/* Le selezioni salvate prima degli operatori tengono la sola lista di valori. */
function asFilter(raw) {
  if (Array.isArray(raw)) return { op: state.meta?.default_op || "in", values: [...raw] };
  return { op: raw?.op || state.meta?.default_op || "in", values: [...(raw?.values || [])] };
}

function filterOf(col) {
  if (!state.dims[col]) state.dims[col] = asFilter(null);
  return state.dims[col];
}

function isMulti(op) {
  const found = (state.meta?.ops || []).find((o) => o.op === op);
  return found ? found.multi : true;
}

function isNegative(op) {
  return op === "is_not" || op === "not_in";
}

function renderDimensions(dims) {
  const box = $("#dimensions");
  box.innerHTML = "";
  const ops = state.meta?.ops || [];
  for (const dim of dims) {
    const sec = document.createElement("section");
    sec.className = "dim";
    const head = document.createElement("div");
    head.className = "dim-head";
    head.innerHTML = `<h2>${esc(dim.title)}</h2>` +
      `<select class="op" aria-label="operatore per ${esc(dim.title)}">` +
      ops.map((o) => `<option value="${esc(o.op)}">${esc(o.label)}</option>`).join("") +
      `</select>`;
    const opSel = head.querySelector("select");
    opSel.value = filterOf(dim.col).op;
    opIndex[dim.col] = opSel;
    sec.appendChild(head);

    const pills = document.createElement("div");
    pills.className = "pills";
    pillIndex[dim.col] = {};
    for (const v of dim.values) {
      const label = document.createElement("label");
      label.className = "pill";
      // v.label arriva da schema.html_value e puo' contenere markup (N<sub>B</sub>);
      // v.value invece e' un valore della config W&B, va sempre passato da esc()
      label.innerHTML = `<input type="checkbox" value="${esc(v.value)}">` +
        `<span>${v.label}</span><span class="count">${v.count}</span>`;
      const input = label.querySelector("input");
      input.addEventListener("change", () => {
        const f = filterOf(dim.col);
        const cur = new Set(f.values);
        input.checked ? cur.add(v.value) : cur.delete(v.value);
        // con «è» / «non è» il valore e' uno solo: sceglierne un altro sostituisce
        f.values = (!isMulti(f.op) && input.checked) ? [v.value] : [...cur];
        syncPills(dim.col);
        // le esclusioni valgono per la selezione corrente: cambiando i filtri
        // si riparte con tutte le combinazioni incluse
        state.excluded.clear();
        scheduleQuery();
      });
      pills.appendChild(label);
      pillIndex[dim.col][v.value] = { input, label, countEl: label.querySelector(".count") };
    }
    sec.appendChild(pills);
    syncPills(dim.col);      // stato iniziale coerente (anche dopo un ricaricamento)

    opSel.addEventListener("change", () => {
      const f = filterOf(dim.col);
      f.op = opSel.value;
      // passando a un operatore a valore singolo si tiene solo il primo scelto
      if (!isMulti(f.op) && f.values.length > 1) f.values = [f.values[0]];
      syncPills(dim.col);
      state.excluded.clear();
      scheduleQuery();
    });
    box.appendChild(sec);
  }
}

/* Allinea le caselle allo stato: quali sono spuntate e se la dimensione esclude
   invece di includere (le pillole scelte si colorano di rosso). */
function syncPills(col) {
  const f = filterOf(col);
  const chosen = new Set(f.values.map(String));
  const negative = isNegative(f.op);
  for (const [value, ref] of Object.entries(pillIndex[col] || {})) {
    ref.input.checked = chosen.has(String(value));
    ref.label.classList.toggle("on", ref.input.checked && !negative);
    ref.label.classList.toggle("off", ref.input.checked && negative);
  }
  if (opIndex[col]) {
    opIndex[col].value = f.op;
    // la tendina chiusa dice comunque se la dimensione esclude
    opIndex[col].classList.toggle("negative", negative);
  }
}

/* Conteggi aggiornati sulle run rimaste: per ogni dimensione si ignorano i suoi
   stessi filtri, cosi' il numero dice quante run resterebbero scegliendo quel valore. */
function applyCounts(counts) {
  if (!counts) return;
  for (const [col, values] of Object.entries(pillIndex)) {
    const colCounts = counts[col] || {};
    for (const [value, ref] of Object.entries(values)) {
      const n = colCounts[value] || 0;
      ref.countEl.textContent = n;
      ref.label.classList.toggle("zero", n === 0);
    }
  }
}

/* Tendina "cosa plottare": eval dai file locali, il resto dalla history W&B. */
function renderMetrics(groups, dflt) {
  const sel = $("#grid-metric");
  sel.innerHTML = groups.map((g) =>
    `<optgroup label="${esc(g.group)}">` +
    g.options.map((o) =>
      `<option value="${esc(o.key)}">${esc(o.label)}` +
      `${o.source === "wandb" ? " · W&B" : ""}</option>`).join("") +
    `</optgroup>`).join("");
  state.grid.metric = dflt || state.grid.metric;
  sel.value = state.grid.metric;
  sel.addEventListener("change", () => {
    state.grid.metric = sel.value;
    maybeAutoPreview();
  });
}

/* Baseline: si scelgono le famiglie (PPO/SAC/TD3/GePPO-original), disegnate in
   nero con le epoche di base — se ne possono accendere piu' d'una insieme, e si
   distinguono per tratteggio. La tendina aggiunge una baseline nera tratteggiata
   a epoche moltiplicate: «segui ω» usa in ogni pannello il PPO con lo stesso
   moltiplicatore dell'ω del pannello, come nelle figure del paper. */
const BASELINE_FAMILIES = ["PPO", "SAC", "TD3", "GePPO-original"];
const baselineIndex = {};   // famiglia -> {input, label}, per il ripristino

function baselineFamilies() {
  const fam = state.grid.baseline.family;
  return Array.isArray(fam) ? fam : (fam ? [fam] : []);
}

function renderBaselineControls(dimensions) {
  const famDim = dimensions.find((d) => d.col === "family");
  const multDim = dimensions.find((d) => d.col === "epoch_mult");
  const available = BASELINE_FAMILIES
    .filter((f) => (famDim ? famDim.values.map((v) => v.value) : []).includes(f));
  state.grid.baseline.family = baselineFamilies();

  // solo i moltiplicatori veri (x2, x4, x8): x1 e' gia' la baseline continua
  const mults = (multDim ? multDim.values.map((v) => v.value) : [])
    .filter((m) => parseFloat(m) > 1)
    .sort((a, b) => parseFloat(a) - parseFloat(b));
  const epSel = $("#grid-baseline-epochs");
  epSel.innerHTML = `<option value="">nessuna</option>` +
    `<option value="follow_window">segui ω (ω K̃ epochs)</option>` +
    mults.map((m) => `<option value="${m}">×${Math.round(parseFloat(m))} epoche</option>`).join("");
  epSel.value = state.grid.baseline.epochs || "";

  const syncEpochsVisibility = () => {
    const show = baselineFamilies().includes("PPO") && mults.length > 0;
    $("#grid-baseline-epochs-field").hidden = !show;
    if (!show) state.grid.baseline.epochs = "";
  };

  const box = $("#grid-baseline");
  box.className = "pills";
  box.innerHTML = "";
  for (const key of Object.keys(baselineIndex)) delete baselineIndex[key];
  for (const fam of available) {
    const on = baselineFamilies().includes(fam);
    const label = document.createElement("label");
    label.className = "pill" + (on ? " on" : "");
    label.innerHTML = `<input type="checkbox" value="${fam}"${on ? " checked" : ""}>` +
      `<span>${fam}</span>`;
    const input = label.querySelector("input");
    baselineIndex[fam] = { input, label };
    input.addEventListener("change", () => {
      label.classList.toggle("on", input.checked);
      // l'ordine delle famiglie e' quello di BASELINE_FAMILIES: cosi' il
      // tratteggio di una baseline non cambia a seconda di come l'hai accesa
      const cur = new Set(baselineFamilies());
      input.checked ? cur.add(fam) : cur.delete(fam);
      state.grid.baseline.family = BASELINE_FAMILIES.filter((f) => cur.has(f));
      syncEpochsVisibility();
      maybeAutoPreview();
    });
    box.appendChild(label);
  }
  if (!available.length) {
    box.innerHTML = `<span class="hint">nessuna famiglia disponibile</span>`;
  }
  syncEpochsVisibility();

  epSel.addEventListener("change", () => {
    state.grid.baseline.epochs = epSel.value;
    maybeAutoPreview();
  });
}

function renderGridControls(fields) {
  for (const [id, key] of [["#grid-rows", "rows"], ["#grid-cols", "cols"]]) {
    const sel = $(id);
    sel.innerHTML = `<option value="">—</option>` +
      fields.map((f) => `<option value="${f.col}">${f.title}</option>`).join("");
    sel.value = state.grid[key];
    sel.addEventListener("change", () => {
      state.grid[key] = sel.value;
      maybeAutoPreview();
    });
  }
  $("#grid-band").addEventListener("change", (e) => {
    state.grid.band = e.target.value; maybeAutoPreview();
  });
  $("#grid-smooth").addEventListener("change", (e) => {
    state.grid.smooth = parseInt(e.target.value, 10) || 1; maybeAutoPreview();
  });
  syncSizeInputs();
  const onSizeChange = () => {
    const w = parseInt($("#grid-width-px").value, 10) || DEFAULT_SIZE_PX;
    const h = parseInt($("#grid-height-px").value, 10) || DEFAULT_SIZE_PX;
    state.grid.panel_size = [w / PREVIEW_DPI, h / PREVIEW_DPI];
    maybeAutoPreview();
  };
  $("#grid-width-px").addEventListener("change", onSizeChange);
  $("#grid-height-px").addEventListener("change", onSizeChange);
}

/* Dimensione del pannello: la pagina la mostra in px (piu' intuitiva di
   pollici), FigureSpec.panel_size la vuole in pollici — PREVIEW_DPI converte. */
function syncSizeInputs() {
  const [w, h] = state.grid.panel_size || [DEFAULT_SIZE_PX / PREVIEW_DPI, DEFAULT_SIZE_PX / PREVIEW_DPI];
  $("#grid-width-px").value = Math.round(w * PREVIEW_DPI);
  $("#grid-height-px").value = Math.round(h * PREVIEW_DPI);
}

/* --- query ---------------------------------------------------------------- */

function payload() {
  return { dims: state.dims, seeds: state.seeds, grid: state.grid,
           excluded: [...state.excluded],
           pinned: [...state.pinned],
           series_overrides: state.series_overrides };
}

function scheduleQuery() {
  clearTimeout(queryTimer);
  queryTimer = setTimeout(runQuery, 180);
}

async function runQuery() {
  const data = await post("/api/query", payload());
  if (data.error) { toast(data.error); return; }
  $("#n-runs").textContent = data.n_runs;
  $("#n-configs").textContent = data.n_configs;
  $("#states").innerHTML = Object.entries(data.states || {})
    .map(([k, v]) => `<span class="badge ${k}">${k}: ${v}</span>`).join("") +
    (data.n_excluded ? `<span class="badge excluded">escluse: ${data.n_excluded}</span>` : "") +
    (state.pinned.size ? `<span class="badge pinned">📌 ${state.pinned.size}</span>` : "");
  state.filterArgs = data.filter_args || [];
  applyCounts(data.counts);
  renderCoverage(data.coverage);
  maybeAutoPreview();
}

function renderCoverage(cov) {
  const box = $("#coverage");
  if (!cov || !cov.rows.length) {
    box.innerHTML = `<p class="placeholder">Nessuna run per questi filtri.</p>`;
    $("#coverage-note").textContent = "";
    return;
  }
  const maxSeeds = Math.max(...cov.rows.map((r) => r.n_seeds));
  const allOn = cov.rows.every((r) => r.on);
  const allPinned = cov.rows.every((r) => r.pinned);
  const head =
    `<th class="pick"><input type="checkbox" id="cov-all"${allOn ? " checked" : ""}` +
    ` title="tutte / nessuna"></th>` +
    `<th class="pin"><button id="cov-pin-all" class="pin-btn${allPinned ? " on" : ""}"` +
    ` title="pinna tutte le presenti / spinna tutte">📌</button></th>` +
    cov.columns.map((c) => `<th>${c}</th>`).join("") +
    `<th>run</th><th>seed</th><th>quali seed</th>`;
  const body = cov.rows.map((r, i) => {
    const partial = r.n_seeds < maxSeeds;
    const cls = [partial ? "partial" : "", r.on ? "" : "off", r.pinned ? "pinned" : ""]
      .filter(Boolean).join(" ");
    return `<tr class="${cls}">` +
      `<td class="pick"><input type="checkbox" data-row="${i}"${r.on ? " checked" : ""}></td>` +
      `<td class="pin"><button class="pin-btn${r.pinned ? " on" : ""}" data-pin="${i}"` +
      ` title="${r.pinned ? "spinna: torna a seguire i filtri" :
                 "pinna: resta nel grafico anche cambiando i filtri"}">📌</button></td>` +
      r.cells.map((c) => `<td>${c}</td>`).join("") +
      `<td class="num">${r.n_runs}</td>` +
      `<td class="num seeds-missing">${r.n_seeds}</td>` +
      `<td class="num">${r.seeds}</td></tr>`;
  }).join("");
  box.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

  const setRow = (row, on) => {
    for (const id of row.run_ids) on ? state.excluded.delete(id) : state.excluded.add(id);
  };
  for (const input of box.querySelectorAll("input[data-row]")) {
    input.addEventListener("change", () => {
      setRow(cov.rows[Number(input.dataset.row)], input.checked);
      runQuery();   // rilancia da se' anche l'anteprima
    });
  }
  $("#cov-all").addEventListener("change", (e) => {
    for (const row of cov.rows) setRow(row, e.target.checked);
    runQuery();
  });

  // pin: una riga e' pinnata solo se lo sono tutte le sue run; il pulsante in
  // testata pinna tutte le righe presenti, o le spinna se lo sono gia' tutte
  const setPin = (row, on) => {
    for (const id of row.run_ids) on ? state.pinned.add(id) : state.pinned.delete(id);
  };
  for (const btn of box.querySelectorAll("button[data-pin]")) {
    btn.addEventListener("click", () => {
      const row = cov.rows[Number(btn.dataset.pin)];
      setPin(row, !row.pinned);
      runQuery();
    });
  }
  $("#cov-pin-all").addEventListener("click", () => {
    for (const row of cov.rows) setPin(row, !allPinned);
    runQuery();
  });

  const note = $("#coverage-note");
  const off = cov.rows.filter((r) => !r.on).length;
  const notes = [];
  if (cov.truncated) notes.push("(elenco troncato)");
  if (off) notes.push(`${off} combinazioni escluse dal grafico`);
  note.textContent = notes.join(" · ");
  if (off) {
    const btn = document.createElement("button");
    btn.className = "ghost small";
    btn.textContent = "includi tutte";
    btn.addEventListener("click", () => { state.excluded.clear(); runQuery(); });
    note.append(" ", btn);
  }
}

/* --- anteprima ------------------------------------------------------------ */

function maybeAutoPreview() {
  if ($("#auto-preview").checked) runPreview();
}

async function runPreview() {
  const box = $("#preview-box");
  box.innerHTML = `<p class="spinner">Disegno in corso…</p>`;
  const data = await post("/api/preview", payload());
  if (data.error) { box.innerHTML = `<p class="error">${data.error}</p>`; return; }
  const hue = (data.hue || []).join(" × ");
  // dimensioni che variano ma non separano le curve: finiscono mediate insieme,
  // ed e' il modo piu' facile di guardare una figura che dice il falso
  const merged = (data.merged || []).join(", ");
  box.innerHTML = `<img src="data:image/png;base64,${data.png}" alt="anteprima">` +
    `<p class="hint">${esc(data.metric || "")} · ${data.series} serie · ` +
    `${data.panels} pannelli · ${data.elapsed}s` +
    (hue ? ` · colori${data.auto_hue ? " (auto)" : ""}: ${esc(hue)}` : "") + `</p>` +
    (merged ? `<p class="error merged-warning">${esc(merged)} ${
      data.merged.length > 1 ? "variano" : "varia"} senza separare le curve: ` +
      `configurazioni diverse sono mediate insieme.</p>` : "");
  renderSeries(data.series_list || [], data.palette || []);
}

/* --- ritocchi alle serie -------------------------------------------------- */

/* Nome, colore e tratteggio di una serie, scelti a mano qui e applicati in
   tutti i pannelli — vale sia per le curve normali sia per le baseline (nere
   per convenzione, ma anche loro ritoccabili). Restano nella selezione (quindi
   anche nel .tex esportato); per renderli permanenti si copia la regola
   [[series]] e la si incolla in style.toml. La chiave e' sempre l'etichetta di
   partenza: rinominare non la cambia. */
function renderSeries(items, palette) {
  state.series = items;
  state.palette = palette;
  const box = $("#series-box");
  box.hidden = items.length === 0;
  const list = $("#series-list");
  list.innerHTML = "";
  for (const s of items) {
    const el = document.createElement("button");
    el.className = "series-item" + (state.seriesSel === s.key ? " on" : "");
    el.innerHTML = `<span class="swatch" style="background:${esc(s.color)}"></span>` +
      `<span class="series-name">${esc(s.label)}</span>` +
      (s.renamed || s.recolored || s.restyled
        ? `<span class="badge tweak">ritoccata</span>` : "");
    el.addEventListener("click", () => {
      state.seriesSel = state.seriesSel === s.key ? null : s.key;
      renderSeries(items, palette);
    });
    list.appendChild(el);
  }
  renderSeriesEdit();
}

function renderSeriesEdit() {
  const chosen = (state.series || []).find((s) => s.key === state.seriesSel);
  $("#series-edit").hidden = !chosen;
  if (!chosen) return;
  $("#series-name").value = chosen.label;
  $("#series-style").value = chosen.style || "solid";
  const box = $("#series-palette");
  box.innerHTML = "";
  for (const color of state.palette || []) {
    const el = document.createElement("button");
    el.className = "swatch pick" + (color === chosen.color ? " on" : "");
    el.style.background = color;
    el.title = color;
    el.addEventListener("click", () => tweakSeries(chosen.key, { color }));
    box.appendChild(el);
  }
}

function tweakSeries(key, change) {
  const cur = state.series_overrides[key] || {};
  state.series_overrides[key] = { ...cur, ...change };
  runPreview();
}

function resetSeries() {
  if (!state.seriesSel) return;
  delete state.series_overrides[state.seriesSel];
  runPreview();
}

function copySeriesRule() {
  const chosen = (state.series || []).find((s) => s.key === state.seriesSel);
  if (!chosen) return;
  navigator.clipboard.writeText(chosen.rule)
    .then(() => toast("Regola copiata: incollala in plots/style.toml"))
    .catch(() => toast("Copia non riuscita"));
}

/* --- export --------------------------------------------------------------- */

/* Il file e' servito da /download/<token> con Content-Disposition: attachment, così
   il download funziona anche dove i blob sono bloccati (browser interno di VSCode).
   In quel caso resta il link manuale sotto l'anteprima. */
function download(url, filename) {
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.rel = "noopener";
  document.body.appendChild(a); a.click(); a.remove();
}

function showDownloadLink(data) {
  const box = $("#download-box");
  box.hidden = false;
  box.innerHTML = `Se il download non parte: ` +
    `<a href="${data.url}" download="${esc(data.filename)}">${esc(data.filename)}</a>` +
    ` · copia sul server: <code>${esc(data.saved_path)}</code>`;
}

async function exportFigure(format) {
  const btn = format === "tex" ? $("#export-latex") : $("#export-jpeg");
  const old = btn.textContent;
  btn.disabled = true; btn.textContent = "…";
  try {
    const data = await post("/api/export", { ...payload(), format });
    if (data.error) { toast(data.error); return; }
    download(data.url, data.filename);
    showDownloadLink(data);
    toast(format === "tex"
      ? `${data.n_panels} pannelli + snippet · ${data.filename}`
      : data.filename);
  } finally {
    btn.disabled = false; btn.textContent = old;
  }
}

/* --- azioni --------------------------------------------------------------- */

async function save() {
  const name = $("#save-name").value.trim();
  const data = await post("/api/save", { ...payload(), name });
  if (data.error) { toast(data.error); return; }
  renderSelections(data.items);
  toast(`«${data.name}» salvata: ${data.n_runs} run`);
}

/* --- selezioni salvate ----------------------------------------------------- */

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* Di ogni selezione restano solo il nome (rinominabile) e i parametri scelti. */
function renderSelections(items) {
  selectionItems = items || [];
  const box = $("#selections");
  if (!selectionItems.length) {
    box.innerHTML = `<p class="placeholder">Nessuna selezione salvata.</p>`;
    return;
  }
  box.innerHTML = selectionItems.map((it) => `
    <div class="sel-item" data-slug="${esc(it.slug)}">
      <div class="sel-load" role="button" tabindex="0">
        <strong>${esc(it.name)}</strong>
        <span class="hint filters">${esc(it.summary)}</span>
      </div>
      <button class="sel-rename ghost small" title="rinomina">✎</button>
      <button class="sel-del ghost small" title="elimina">✕</button>
    </div>`).join("");
  box.querySelectorAll(".sel-item").forEach((item) => {
    const slug = item.dataset.slug;
    const load = item.querySelector(".sel-load");
    load.addEventListener("click", () => loadSelection(slug));
    load.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); loadSelection(slug); }
    });
    item.querySelector(".sel-rename")
      .addEventListener("click", () => startRename(item, slug));
    item.querySelector(".sel-del")
      .addEventListener("click", () => deleteSelection(slug));
  });
}

/* Rinomina in linea: il nome diventa un campo di testo, Invio conferma, Esc annulla. */
function startRename(item, slug) {
  const nameEl = item.querySelector("strong");
  if (!nameEl) return;                    // rinomina gia' aperta
  const old = nameEl.textContent;
  const input = document.createElement("input");
  input.type = "text";
  input.className = "sel-name-edit";
  input.value = old;
  input.maxLength = 60;
  input.addEventListener("click", (e) => e.stopPropagation());
  nameEl.replaceWith(input);
  input.focus();
  input.select();

  let done = false;
  const finish = async (commit) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (!commit || !name || name === old) {
      renderSelections(selectionItems);
      return;
    }
    const data = await post("/api/selections/rename", { slug, name });
    if (data.error) { toast(data.error); renderSelections(selectionItems); return; }
    renderSelections(data.items);
    toast(`Rinominata: «${data.name}»`);
  };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

/* La sezione si puo' chiudere; la scelta resta tra una visita e l'altra. */
function setPanel(open) {
  $("#sel-card").classList.toggle("closed", !open);
  $("#sel-toggle").setAttribute("aria-expanded", open ? "true" : "false");
  try { localStorage.setItem("selpanel", open ? "1" : "0"); } catch (e) { /* ignora */ }
}

async function loadSelection(slug) {
  const data = await post("/api/selections/load", { slug });
  if (data.error) { toast(data.error); return; }
  applySelection(data);
  toast(`«${data.name}» applicata: ${data.n_runs} run`);
}

async function deleteSelection(slug) {
  const data = await post("/api/selections/delete", { slug });
  renderSelections(data.items);
  toast("Selezione eliminata");
}

/* Riporta filtri, seed e impostazioni di griglia allo stato salvato. */
/* La selezione salvata contiene lo `spec` completo della figura (rtplots/figure.py):
   qui se ne riprendono i campi che la pagina sa mostrare. Le selezioni vecchie
   sono gia' state convertite dal server, quindi `spec` c'e' sempre. */
function gridFromSpec(spec) {
  const out = {};
  for (const k of ["rows", "cols", "band", "smooth", "metric", "panel_size"]) {
    if (spec[k] !== undefined && spec[k] !== null) out[k] = spec[k];
  }
  out.hue = spec.hue || [];
  // le selezioni vecchie hanno una famiglia sola come stringa
  const fam = (spec.baseline || {}).family;
  out.baseline = { family: Array.isArray(fam) ? fam : (fam ? [fam] : []),
                   epochs: (spec.baseline || {}).epochs || "" };
  return out;
}

function applySelection(entry) {
  state.dims = {};
  for (const [col, raw] of Object.entries(entry.dims || {})) state.dims[col] = asFilter(raw);
  state.seeds = entry.seeds || { min: null, max: null };
  state.excluded = new Set(entry.excluded || []);
  state.pinned = new Set(entry.pinned || []);
  state.series_overrides = entry.series_overrides || (entry.spec || {}).series_overrides || {};
  state.seriesSel = null;
  state.grid = { ...state.grid, ...gridFromSpec(entry.spec || {}) };
  $("#save-name").value = entry.name || "";

  for (const col of Object.keys(pillIndex)) syncPills(col);
  $("#seed-min").value = state.seeds.min ?? state.meta.seed_min;
  $("#seed-max").value = state.seeds.max ?? state.meta.seed_max;
  $("#grid-metric").value = state.grid.metric || "eval/mean_reward";
  syncSizeInputs();
  $("#grid-rows").value = state.grid.rows || "";
  $("#grid-cols").value = state.grid.cols || "";
  $("#grid-band").value = state.grid.band || "ci95";
  $("#grid-smooth").value = state.grid.smooth || 5;
  // ripristino senza dispatch: un evento qui rilancerebbe l'anteprima a meta' setup
  state.grid.baseline = state.grid.baseline || { family: [], epochs: "" };
  state.grid.baseline.family = baselineFamilies();
  for (const [fam, ref] of Object.entries(baselineIndex)) {
    ref.input.checked = state.grid.baseline.family.includes(fam);
    ref.label.classList.toggle("on", ref.input.checked);
  }
  $("#grid-baseline-epochs-field").hidden = !state.grid.baseline.family.includes("PPO");
  $("#grid-baseline-epochs").value = state.grid.baseline.epochs || "";
  runQuery();
}

function copyFilters() {
  const text = (state.filterArgs || []).join(" ");
  navigator.clipboard.writeText(text)
    .then(() => toast(text ? `Copiato: ${text}` : "Nessun filtro attivo"))
    .catch(() => toast(text || "nessun filtro"));
}

function reset() {
  state.dims = {};
  state.seeds = { min: null, max: null };
  state.excluded.clear();
  state.pinned.clear();
  state.series_overrides = {};
  state.seriesSel = null;
  for (const col of Object.keys(pillIndex)) syncPills(col);
  $("#seed-min").value = state.meta.seed_min;
  $("#seed-max").value = state.meta.seed_max;
  runQuery();
}

/* --- avvio ---------------------------------------------------------------- */

async function init() {
  const meta = await (await fetch("/api/dimensions")).json();
  state.meta = meta;
  renderDimensions(meta.dimensions);
  renderMetrics(meta.metrics, meta.default_metric);
  renderGridControls(meta.grid_fields);
  renderBaselineControls(meta.dimensions);
  renderSelections(meta.selections);
  $("#selection-path").textContent = meta.selection_path;
  $("#save-name").addEventListener("keydown", (e) => { if (e.key === "Enter") save(); });
  for (const [id, key] of [["#seed-min", "min"], ["#seed-max", "max"]]) {
    const el = $(id);
    el.value = key === "min" ? meta.seed_min : meta.seed_max;
    el.min = meta.seed_min; el.max = meta.seed_max;
    el.addEventListener("change", () => {
      const v = parseInt(el.value, 10);
      state.seeds[key] = Number.isFinite(v) ? v : null;
      scheduleQuery();
    });
  }
  $("#preview").addEventListener("click", runPreview);
  $("#series-reset").addEventListener("click", resetSeries);
  $("#series-rule").addEventListener("click", copySeriesRule);
  // il nome si applica quando si esce dal campo o si preme Invio: ridisegnare a
  // ogni tasto vorrebbe dire una figura per lettera
  $("#series-name").addEventListener("change", (e) => {
    if (state.seriesSel) tweakSeries(state.seriesSel, { name: e.target.value });
  });
  $("#series-style").addEventListener("change", (e) => {
    if (state.seriesSel) tweakSeries(state.seriesSel, { style: e.target.value });
  });
  $("#export-jpeg").addEventListener("click", () => exportFigure("jpeg"));
  $("#export-latex").addEventListener("click", () => exportFigure("tex"));
  $("#sel-toggle").addEventListener("click", () =>
    setPanel($("#sel-card").classList.contains("closed")));
  let open = true;
  try { open = localStorage.getItem("selpanel") !== "0"; } catch (e) { /* ignora */ }
  setPanel(open);
  $("#save").addEventListener("click", save);
  $("#copy").addEventListener("click", copyFilters);
  $("#reset").addEventListener("click", reset);
  runQuery();
}

init();
