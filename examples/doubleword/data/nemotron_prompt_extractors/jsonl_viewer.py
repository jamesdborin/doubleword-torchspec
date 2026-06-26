#!/usr/bin/env python3
"""Local paginated viewer for extracted prompt JSONL files."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_OUTPUT_ROOT = Path("/tmp/nemotron_prompt_only_exports")
PROMPT_FILENAMES = ("prompts.jsonl", "prompt.jsonl")
MAX_PAGE_SIZE = 200


def line_count(path: Path) -> int | None:
    try:
        output = subprocess.check_output(["wc", "-l", str(path)], text=True, timeout=10)
        return int(output.strip().split()[0])
    except Exception:
        return None


def human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


def find_prompt_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for filename in PROMPT_FILENAMES:
        files.extend(path for path in root.rglob(filename) if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def dataset_label(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.name
    if len(rel.parts) >= 2:
        return rel.parts[-2].replace("__", "/")
    return rel.as_posix()


def build_index(root: Path, include_counts: bool = False) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for index, path in enumerate(find_prompt_files(root)):
        stat = path.stat()
        datasets.append(
            {
                "id": str(index),
                "label": dataset_label(root, path),
                "path": str(path),
                "relative_path": path.relative_to(root).as_posix(),
                "rows": line_count(path) if include_counts else None,
                "size": stat.st_size,
                "size_label": human_bytes(stat.st_size),
                "mtime": stat.st_mtime,
            }
        )
    return datasets


def resolve_dataset(root: Path, dataset_id: str) -> Path | None:
    try:
        index = int(dataset_id)
    except ValueError:
        return None
    files = find_prompt_files(root)
    if index < 0 or index >= len(files):
        return None
    return files[index]


def collect_headings(records: list[dict[str, Any]]) -> list[str]:
    headings: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                headings.append(key)
                seen.add(key)
    return headings


def normalize_page(page: int, page_size: int) -> tuple[int, int]:
    page = max(1, page)
    page_size = min(MAX_PAGE_SIZE, max(1, page_size))
    return page, page_size


def read_page(path: Path, page: int, page_size: int) -> dict[str, Any]:
    page, page_size = normalize_page(page, page_size)
    start = (page - 1) * page_size
    end = start + page_size
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    headings: list[str] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number <= start:
                continue
            if line_number > end:
                break
            text = line.rstrip("\n")
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                record = {"__raw__": text}
                errors.append({"line": line_number, "error": str(exc)})
            if not isinstance(record, dict):
                record = {"value": record}
            for key in record:
                if key not in seen:
                    headings.append(key)
                    seen.add(key)
            records.append({"line": line_number, "values": record})

    total_rows = line_count(path)
    if total_rows is not None:
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
    else:
        total_pages = None

    return {
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "headings": headings or collect_headings([row["values"] for row in records]),
        "records": records,
        "errors": errors,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt JSONL Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1b2430;
      --muted: #687385;
      --line: #d9dee8;
      --accent: #2563eb;
      --accent-soft: #e8f0ff;
      --bad: #b42318;
      --code: #f3f5f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      min-height: 100vh;
      position: sticky;
      top: 0;
      align-self: start;
      display: flex;
      flex-direction: column;
    }
    .side-head { padding: 16px; border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
    input, select, button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--text);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; font-weight: 650; }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button:disabled { cursor: not-allowed; opacity: .45; }
    #filter { width: 100%; }
    .dataset-list { overflow: auto; padding: 8px; }
    .dataset {
      width: 100%;
      min-height: 58px;
      display: block;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      padding: 8px;
      margin-bottom: 4px;
    }
    .dataset.active { background: var(--accent-soft); border-color: #b9ccff; }
    .dataset strong { display: block; overflow-wrap: anywhere; }
    .muted { color: var(--muted); font-size: 12px; }
    main { min-width: 0; padding: 16px; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .title { min-width: 240px; }
    .title h2 { margin: 0 0 2px; font-size: 20px; letter-spacing: 0; overflow-wrap: anywhere; }
    .heading-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0 0 12px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      font: 12px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      max-height: calc(100vh - 140px);
    }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
      min-width: 220px;
    }
    th:first-child, td:first-child {
      min-width: 74px;
      width: 74px;
      position: sticky;
      left: 0;
      z-index: 2;
      background: #f1f4f8;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 3;
      background: #f1f4f8;
      text-align: left;
      color: var(--muted);
      font-size: 12px;
    }
    th:first-child { z-index: 4; }
    td { background: white; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 320px;
      overflow: auto;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .empty {
      padding: 28px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: white;
      color: var(--muted);
    }
    .error { color: var(--bad); }
    @media (max-width: 900px) {
      body { display: block; }
      aside { position: relative; min-height: 0; max-height: 42vh; }
      .table-wrap { max-height: none; }
    }
  </style>
</head>
<body>
  <aside>
    <div class="side-head">
      <h1>Prompt JSONL Viewer</h1>
      <input id="filter" placeholder="Filter categories" />
      <div class="muted" id="rootLabel"></div>
    </div>
    <div class="dataset-list" id="datasets"></div>
  </aside>
  <main>
    <div class="toolbar">
      <div class="title">
        <h2 id="datasetTitle">No dataset selected</h2>
        <div class="muted" id="datasetMeta"></div>
      </div>
      <div class="controls">
        <button id="prev">Previous</button>
        <input id="page" type="number" min="1" value="1" style="width:84px" />
        <select id="pageSize">
          <option>10</option>
          <option selected>25</option>
          <option>50</option>
          <option>100</option>
          <option>200</option>
        </select>
        <button class="primary" id="go">Go</button>
        <button id="next">Next</button>
      </div>
    </div>
    <div class="heading-strip" id="headings"></div>
    <div id="content" class="empty">Choose a category on the left.</div>
  </main>
  <script>
    const datasetsEl = document.getElementById('datasets');
    const filterEl = document.getElementById('filter');
    const titleEl = document.getElementById('datasetTitle');
    const metaEl = document.getElementById('datasetMeta');
    const headingsEl = document.getElementById('headings');
    const contentEl = document.getElementById('content');
    const pageEl = document.getElementById('page');
    const pageSizeEl = document.getElementById('pageSize');
    const prevEl = document.getElementById('prev');
    const nextEl = document.getElementById('next');
    const goEl = document.getElementById('go');
    let datasets = [];
    let active = null;
    let activePage = null;

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function formatValue(value) {
      if (value === null || value === undefined) return '';
      if (typeof value === 'string') return value;
      return JSON.stringify(value, null, 2);
    }

    function renderDatasets() {
      const q = filterEl.value.trim().toLowerCase();
      const visible = datasets.filter(d => d.label.toLowerCase().includes(q) || d.relative_path.toLowerCase().includes(q));
      datasetsEl.innerHTML = visible.map(d => `
        <button class="dataset ${active && active.id === d.id ? 'active' : ''}" data-id="${esc(d.id)}">
          <strong>${esc(d.label)}</strong>
          <span class="muted">${esc(d.rows ?? '?')} rows · ${esc(d.size_label)}</span>
        </button>
      `).join('');
    }

    function renderPage(payload) {
      activePage = payload;
      const totalPages = payload.total_pages || '?';
      pageEl.value = payload.page;
      prevEl.disabled = payload.page <= 1;
      nextEl.disabled = payload.total_pages ? payload.page >= payload.total_pages : payload.records.length < payload.page_size;
      titleEl.textContent = active ? active.label : 'No dataset selected';
      metaEl.textContent = active ? `${active.relative_path} · page ${payload.page} of ${totalPages} · ${payload.total_rows ?? '?'} rows` : '';
      headingsEl.innerHTML = payload.headings.map(h => `<span class="chip">${esc(h)}</span>`).join('');
      if (!payload.records.length) {
        contentEl.className = 'empty';
        contentEl.textContent = 'No rows on this page.';
        return;
      }
      const header = ['Line', ...payload.headings];
      const rows = payload.records.map(row => `
        <tr>
          <td><pre>${row.line}</pre></td>
          ${payload.headings.map(h => `<td><pre>${esc(formatValue(row.values[h]))}</pre></td>`).join('')}
        </tr>
      `).join('');
      const errors = payload.errors.length
        ? `<div class="error">${payload.errors.length} JSON parse error(s) on this page.</div>`
        : '';
      contentEl.className = 'table-wrap';
      contentEl.innerHTML = `${errors}<table><thead><tr>${header.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
    }

    async function loadIndex() {
      const response = await fetch('/api/datasets', {cache: 'no-store'});
      const payload = await response.json();
      datasets = payload.datasets;
      document.getElementById('rootLabel').textContent = payload.root;
      renderDatasets();
      if (datasets.length) {
        selectDataset(datasets[0].id);
      } else {
        contentEl.textContent = 'No prompt.jsonl or prompts.jsonl files found under the output root.';
      }
    }

    async function selectDataset(id) {
      active = datasets.find(d => d.id === String(id));
      pageEl.value = 1;
      renderDatasets();
      await loadPage();
    }

    async function loadPage() {
      if (!active) return;
      const params = new URLSearchParams({
        dataset: active.id,
        page: pageEl.value || '1',
        page_size: pageSizeEl.value || '25',
      });
      const response = await fetch(`/api/page?${params}`, {cache: 'no-store'});
      if (!response.ok) {
        contentEl.className = 'empty';
        contentEl.textContent = await response.text();
        return;
      }
      renderPage(await response.json());
    }

    datasetsEl.addEventListener('click', event => {
      const button = event.target.closest('.dataset');
      if (button) selectDataset(button.dataset.id);
    });
    filterEl.addEventListener('input', renderDatasets);
    goEl.addEventListener('click', loadPage);
    pageEl.addEventListener('keydown', event => { if (event.key === 'Enter') loadPage(); });
    pageSizeEl.addEventListener('change', () => { pageEl.value = 1; loadPage(); });
    prevEl.addEventListener('click', () => { pageEl.value = Math.max(1, Number(pageEl.value || 1) - 1); loadPage(); });
    nextEl.addEventListener('click', () => { pageEl.value = Number(pageEl.value || 1) + 1; loadPage(); });
    loadIndex();
  </script>
</body>
</html>
"""


class JsonlViewerHandler(BaseHTTPRequestHandler):
    output_root: Path

    def log_message(self, format: str, *args: Any) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}")

    def send_json(self, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, status: int, message: str) -> None:
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_html()
            return
        if parsed.path == "/api/datasets":
            self.send_json({"root": str(self.output_root), "datasets": build_index(self.output_root)})
            return
        if parsed.path == "/api/page":
            query = parse_qs(parsed.query)
            dataset_id = query.get("dataset", [""])[0]
            try:
                page = int(query.get("page", ["1"])[0])
                page_size = int(query.get("page_size", ["25"])[0])
            except ValueError:
                self.send_text(400, "page and page_size must be integers")
                return
            path = resolve_dataset(self.output_root, dataset_id)
            if path is None:
                self.send_text(404, "dataset not found")
                return
            payload = read_page(path, page, page_size)
            payload["dataset"] = dataset_label(self.output_root, path)
            payload["path"] = str(path)
            payload["relative_path"] = path.relative_to(self.output_root).as_posix()
            self.send_json(payload)
            return
        self.send_text(404, "not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a paginated JSONL prompt dataset viewer.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    JsonlViewerHandler.output_root = args.output_root.expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), JsonlViewerHandler)
    print(f"Serving JSONL viewer at http://{args.host}:{args.port}")
    print(f"Scanning for prompt JSONL files under {JsonlViewerHandler.output_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
