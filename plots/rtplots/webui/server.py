"""Server locale del selettore: solo trasporto e instradamento.

Nessuna logica di selezione o di disegno qui dentro — sta in `api.py`, che non
sa nemmeno che esiste HTTP. Le rotte sono una tabella: aggiungerne una e' una
riga, non un ramo in fondo a una catena di `if`.

Libreria standard soltanto (piu' quello che i plot gia' usano): il selettore
gira su una macchina di calcolo, senza dipendenze web da installare.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import uuid
import zipfile
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

from .. import selection  # noqa: E402
from ..index import load_index  # noqa: E402
from ..paths import OUTPUT_DIR, PLOTS_ROOT, SELECTION_JSON  # noqa: E402
from . import api  # noqa: E402

# progetti fuori dal selettore web (restano nell'indice, disponibili da CLI
# con --filter project=...): "rebuttal" e' superata da rt-ppo-ablations, tranne
# le baseline SAC/TD3 che stanno solo li'.
SITE_HIDDEN_PROJECTS = {"rebuttal"}
SITE_HIDDEN_EXCEPT_FAMILIES = {"rebuttal": {"SAC", "TD3"}}


def site_index():
    """L'indice come lo vede il sito: senza i progetti nascosti (con eccezioni)."""
    df = load_index()
    hidden = df.project.isin(SITE_HIDDEN_PROJECTS)
    for project, families in SITE_HIDDEN_EXCEPT_FAMILIES.items():
        hidden &= ~((df.project == project) & df.family.isin(families))
    return df[~hidden]


STATIC_DIR = PLOTS_ROOT / "selector"
STATIC = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}
EXPORT_FORMATS = {"jpeg": ("jpg", "image/jpeg"), "pdf": ("pdf", "application/pdf"),
                  "png": ("png", "image/png")}
MAX_EXPORTS = 12

_plot_lock = threading.Lock()          # matplotlib non e' rientrante
_exports: OrderedDict[str, dict] = OrderedDict()   # figure servite da /download
_exports_lock = threading.Lock()


def _remember_export(filename: str, mime: str, raw: bytes) -> str:
    token = uuid.uuid4().hex[:12]
    with _exports_lock:
        _exports[token] = {"filename": filename, "mime": mime, "raw": raw}
        while len(_exports) > MAX_EXPORTS:
            _exports.popitem(last=False)
    return token


# --- handler degli endpoint POST (df, payload) -> dict ----------------------

def _export_tex(df, payload: dict, sub, name: str) -> dict:
    """Sorgenti pgfplots: un `.tex` per pannello piu' la figura montata, in zip."""
    res = api.tex_panels(df, sub, payload, name, plot_lock=_plot_lock)
    if "error" in res:
        return res
    files = res.pop("files")
    # lo snippet non si mostra a schermo: scende insieme ai pannelli, come file
    files.append((f"{name}_figure.tex", res.pop("latex") + "\n"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, code in files:
        (OUTPUT_DIR / filename).write_text(code)
    filename, mime = f"{name}_tex.zip", "application/zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for member, code in files:
            zf.writestr(member, code)
    raw = buf.getvalue()
    # anche lo zip su disco: `saved_path` deve indicare un file che esiste
    (OUTPUT_DIR / filename).write_bytes(raw)
    token = _remember_export(filename, mime, raw)
    res.update(filename=filename, mime=mime, url=f"/download/{token}/{filename}",
               saved_path=str(OUTPUT_DIR / filename),
               files=[f for f, _ in files])
    print(f"[selector] export {filename} ({len(files) - 1} pannelli + snippet)"
          f" -> {OUTPUT_DIR}")
    return res


def _export(df, payload: dict) -> dict:
    sub = api.selected_runs(df, payload)
    if sub.empty:
        return {"error": "Nessuna run selezionata."}
    fmt = (payload.get("format") or "jpeg").lower()
    if fmt == "tex":
        err = api.wandb_cost_guard(sub, (payload.get("grid") or {}).get("metric"))
        if err:
            return {"error": err}
        return _export_tex(df, payload, sub, api.figure_name(sub))
    if fmt not in EXPORT_FORMATS:
        return {"error": f"Formato non supportato: {fmt}"}
    grid = payload.get("grid") or {}
    err = api.wandb_cost_guard(sub, grid.get("metric"))
    if err:
        return {"error": err}
    ext, mime = EXPORT_FORMATS[fmt]
    res = api.render(df, sub, payload, fmt=ext if ext != "jpg" else "jpg",
                     dpi=int(payload.get("dpi") or 300), plot_lock=_plot_lock)
    if "error" in res:
        return res
    name = api.figure_name(sub)
    filename = f"{name}.{ext}"
    raw = res.pop("raw")
    # copia su disco: resta a disposizione anche se il browser blocca il download
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / filename).write_bytes(raw)
    token = _remember_export(filename, mime, raw)
    res.update(filename=filename, mime=mime, url=f"/download/{token}/{filename}",
               saved_path=str(OUTPUT_DIR / filename),
               latex=api.latex_snippet(name, sub, res, grid.get("band", "se")))
    print(f"[selector] export {filename} ({len(sub)} run) -> {OUTPUT_DIR}")
    return res


def _preview(df, payload: dict) -> dict:
    res = api.preview(df, payload, plot_lock=_plot_lock)
    if "raw" in res:
        res["png"] = base64.b64encode(res.pop("raw")).decode()
    return res


def _load_selection(df, payload: dict) -> dict:
    slug = payload.get("slug") or ""
    if not selection.path_for(slug).exists():
        return {"error": "Selezione non trovata.", "_code": 404}
    return selection.activate(slug)


def _rename(df, payload: dict) -> dict:
    slug = payload.get("slug") or ""
    name = (payload.get("name") or "").strip()
    if not selection.path_for(slug).exists():
        return {"error": "Selezione non trovata.", "_code": 404}
    if not name:
        return {"error": "Il nome non puo' essere vuoto."}
    data = selection.rename(slug, name)
    return {"ok": True, "name": data["name"], "slug": selection.slugify(slug),
            "items": selection.listing()}


def _delete(df, payload: dict) -> dict:
    selection.delete(payload.get("slug") or "")
    return {"ok": True, "items": selection.listing()}


POST_ROUTES = {
    "/api/query": api.query,
    "/api/preview": _preview,
    "/api/export": _export,
    "/api/save": api.save,
    "/api/selections/load": _load_selection,
    "/api/selections/rename": _rename,
    "/api/selections/delete": _delete,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "rtplots-selector"

    def log_message(self, fmt, *a):  # log piu' silenzioso
        if "/api/" in (a[0] if a else ""):
            return
        super().log_message(fmt, *a)

    # --- trasporto ---
    def _send(self, code, body: bytes, ctype="application/json", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        code = obj.pop("_code", code) if isinstance(obj, dict) else code
        self._send(code, json.dumps(obj, default=str).encode())

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/download/"):
            with _exports_lock:
                item = _exports.get(path.split("/")[2])
            if item is None:
                return self._send(404, b"export scaduto", "text/plain")
            # Content-Disposition: attachment -> il browser scarica invece di aprire
            return self._send(200, item["raw"], item["mime"],
                              [("Content-Disposition",
                                f'attachment; filename="{item["filename"]}"')])
        if path in STATIC:
            name, ctype = STATIC[path]
            return self._send(200, (STATIC_DIR / name).read_bytes(), ctype)
        if path == "/api/dimensions":
            return self._json(api.dimensions(site_index()))
        if path == "/api/selections":
            return self._json({"items": selection.listing()})
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        handler = POST_ROUTES.get(path)
        if handler is None:
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            # l'indice viene riletto a ogni richiesta: dopo un build_index.py le
            # run nuove compaiono senza riavviare il server
            return self._json(handler(site_index(), payload))
        except Exception as exc:  # errore visibile nella pagina, server vivo
            import traceback
            traceback.print_exc()
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, code=500)


def serve(host: str = "127.0.0.1", port: int = 8770) -> None:
    load_index()  # fallisce subito se l'indice non c'e'
    try:
        srv = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        if exc.errno != 98:
            raise
        raise SystemExit(
            f"La porta {port} e' gia' occupata (un altro selettore e' attivo).\n"
            f"  chi la usa:  ss -ltnp | grep {port}\n"
            f"  altra porta: python plots/scripts/selector.py --port {port + 1}"
        )
    print(f"[selector] http://{host}:{port}  (Ctrl-C per fermare)")
    print(f"[selector] selezione salvata in {SELECTION_JSON}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[selector] chiuso")
