# =========================================================================
# SYNAPSE NOTEBOOK CELL - Comparison Dashboard
# Reads summary + detail parquet from ADLS, renders interactive HTML dashboard.
# No live computation — purely a viewer cell.
# =========================================================================

from IPython.display import display, HTML
from pyspark.sql import functions as F
import json

# ── ADLS paths (must match comparison_pipeline.py) ────────────────────────
RESULTS_BASE_PATH = "abfss://<container>@<storage>.dfs.core.windows.net/comparison_results/"


def load_results(spark):
    """Load latest-run summary and full detail from ADLS parquet."""
    summary_path = RESULTS_BASE_PATH + "summary/"
    detail_path  = RESULTS_BASE_PATH + "detail/"

    summary_df = spark.read.parquet(summary_path)
    detail_df  = spark.read.parquet(detail_path)

    # Keep only the latest run per dataset
    from pyspark.sql.window import Window
    w = Window.partitionBy("dataset_name").orderBy(F.col("run_timestamp").desc())
    latest_summary = (
        summary_df
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # Join detail to only latest run_ids
    latest_run_ids = latest_summary.select("dataset_name", "run_id")
    latest_detail  = detail_df.join(latest_run_ids, on=["dataset_name", "run_id"], how="inner")

    return latest_summary.toPandas(), latest_detail.toPandas()


def render_dashboard(spark):
    """Main entry point — loads data and renders the interactive dashboard."""
    print("Loading comparison results from ADLS...")
    summary_pd, detail_pd = load_results(spark)

    # Serialize to JSON for embedding in HTML
    summary_records = summary_pd.to_dict(orient="records")
    detail_records  = detail_pd.to_dict(orient="records")

    # Parse sample_mismatches back to list for rendering
    for r in detail_records:
        try:
            r["sample_mismatches"] = json.loads(r.get("sample_mismatches") or "[]")
        except Exception:
            r["sample_mismatches"] = []

    summary_json = json.dumps(summary_records, default=str)
    detail_json  = json.dumps(detail_records,  default=str)

    html = _build_html(summary_json, detail_json)
    display(HTML(html))


# =========================================================================
# HTML / JS DASHBOARD
# =========================================================================

def _build_html(summary_json: str, detail_json: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Tahoma, Verdana, sans-serif; background: #f0f2f5; color: #333; }}

  .db-wrap {{ max-width: 1300px; margin: 0 auto; padding: 24px 16px; }}

  /* ── Header ── */
  .db-header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 28px 32px; border-radius: 12px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px;
  }}
  .db-header h1 {{ font-size: 22px; font-weight: 600; }}
  .db-header .sub {{ font-size: 13px; opacity: 0.85; margin-top: 4px; }}

  /* ── Summary cards ── */
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: white; border-radius: 10px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-top: 4px solid #667eea;
  }}
  .card.pass {{ border-top-color: #28a745; }}
  .card.fail {{ border-top-color: #dc3545; }}
  .card.warn {{ border-top-color: #ffc107; }}
  .card-label {{ font-size: 11px; text-transform: uppercase; color: #888; letter-spacing: 0.5px; }}
  .card-value {{ font-size: 32px; font-weight: 700; margin-top: 6px; }}
  .card.pass .card-value {{ color: #28a745; }}
  .card.fail .card-value {{ color: #dc3545; }}
  .card.warn .card-value {{ color: #f0a500; }}

  /* ── Filter bar ── */
  .filter-bar {{
    background: white; border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }}
  .filter-bar input {{
    border: 1px solid #ddd; border-radius: 6px; padding: 8px 12px;
    font-size: 14px; outline: none; flex: 1; min-width: 180px;
  }}
  .filter-bar input:focus {{ border-color: #667eea; }}
  .filter-group {{ display: flex; gap: 8px; }}
  .filter-btn {{
    padding: 7px 16px; border: 1px solid #ddd; border-radius: 20px;
    background: white; cursor: pointer; font-size: 13px; transition: all 0.15s;
  }}
  .filter-btn.active {{ background: #667eea; color: white; border-color: #667eea; }}
  .filter-btn:hover:not(.active) {{ background: #f0f2f5; }}

  /* ── Dataset table ── */
  .table-wrap {{
    background: white; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); overflow: hidden; margin-bottom: 24px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  thead th {{
    background: #f8f9fa; padding: 12px 16px; text-align: left;
    font-weight: 600; color: #555; border-bottom: 2px solid #eee;
    cursor: pointer; user-select: none; white-space: nowrap;
  }}
  thead th:hover {{ background: #eef0f3; }}
  thead th .sort-arrow {{ margin-left: 4px; color: #aaa; font-size: 10px; }}
  tbody tr {{ border-bottom: 1px solid #f0f0f0; cursor: pointer; transition: background 0.1s; }}
  tbody tr:hover {{ background: #f7f8ff; }}
  tbody tr.expanded {{ background: #f0f3ff; }}
  tbody td {{ padding: 12px 16px; vertical-align: middle; }}

  .badge {{
    display: inline-block; padding: 4px 12px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
  }}
  .badge-pass {{ background: #d4edda; color: #155724; }}
  .badge-fail {{ background: #f8d7da; color: #721c24; }}
  .badge-error {{ background: #fff3cd; color: #856404; }}
  .badge-skip {{ background: #e2e3e5; color: #383d41; }}

  .pct-bar-wrap {{ width: 120px; }}
  .pct-bar {{ height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; }}
  .pct-fill {{ height: 100%; border-radius: 4px; }}
  .pct-label {{ font-size: 11px; color: #666; margin-top: 2px; }}

  /* ── Detail panel ── */
  .detail-panel {{ background: #f7f8ff; border-top: none; }}
  .detail-panel td {{ padding: 0 !important; }}
  .detail-inner {{ padding: 20px 24px; }}
  .detail-title {{ font-weight: 600; font-size: 15px; margin-bottom: 16px; color: #444; }}

  .col-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px;
  }}
  .col-card {{
    background: white; border-radius: 8px; padding: 14px;
    border-left: 4px solid #28a745; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .col-card.mismatch {{ border-left-color: #dc3545; }}
  .col-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .col-name {{ font-weight: 600; font-size: 13px; }}
  .col-type {{ font-size: 11px; color: #888; }}
  .col-pct {{ font-size: 13px; font-weight: 700; }}
  .col-pct.ok {{ color: #28a745; }}
  .col-pct.bad {{ color: #dc3545; }}

  .mini-bar {{ height: 6px; background: #e9ecef; border-radius: 3px; overflow: hidden; margin: 6px 0; }}
  .mini-fill {{ height: 100%; border-radius: 3px; }}

  .stat-row {{ display: flex; justify-content: space-between; font-size: 12px; color: #666; margin: 3px 0; }}
  .stat-val {{ font-weight: 600; color: #333; }}

  .sample-list {{ margin-top: 8px; }}
  .sample-item {{
    font-family: monospace; font-size: 11px; background: #fff5f5;
    border-left: 2px solid #dc3545; padding: 4px 8px; margin: 3px 0; border-radius: 2px;
  }}

  .no-results {{ padding: 48px; text-align: center; color: #888; font-size: 15px; }}

  .schema-issues {{
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
    padding: 10px 14px; margin-bottom: 12px; font-size: 13px;
  }}
</style>
</head>
<body>
<div class="db-wrap">

  <div class="db-header">
    <div>
      <h1>DataFrame Comparison Dashboard</h1>
      <div class="sub" id="hdr-sub">Loading...</div>
    </div>
    <div style="text-align:right; font-size:13px; opacity:0.85;">
      <div id="hdr-ts"></div>
    </div>
  </div>

  <div class="cards">
    <div class="card"><div class="card-label">Total Datasets</div><div class="card-value" id="c-total">—</div></div>
    <div class="card pass"><div class="card-label">Passed</div><div class="card-value" id="c-pass">—</div></div>
    <div class="card fail"><div class="card-label">Failed</div><div class="card-value" id="c-fail">—</div></div>
    <div class="card warn"><div class="card-label">Avg Match %</div><div class="card-value" id="c-avg">—</div></div>
  </div>

  <div class="filter-bar">
    <input type="text" id="search-input" placeholder="Search dataset name..." oninput="applyFilters()">
    <div class="filter-group">
      <button class="filter-btn active" onclick="setFilter('all', this)">All</button>
      <button class="filter-btn" onclick="setFilter('fail', this)">Failures</button>
      <button class="filter-btn" onclick="setFilter('pass', this)">Passes</button>
    </div>
  </div>

  <div class="table-wrap">
    <table id="main-table">
      <thead>
        <tr>
          <th onclick="sortBy('dataset_name')">Dataset <span class="sort-arrow">⇅</span></th>
          <th onclick="sortBy('overall_match')">Status <span class="sort-arrow">⇅</span></th>
          <th onclick="sortBy('match_pct')">Data Match <span class="sort-arrow">⇅</span></th>
          <th onclick="sortBy('matching_columns')">Columns</th>
          <th onclick="sortBy('df1_row_count')">DF1 Rows</th>
          <th onclick="sortBy('df2_row_count')">DF2 Rows</th>
          <th onclick="sortBy('run_timestamp')">Last Run <span class="sort-arrow">⇅</span></th>
        </tr>
      </thead>
      <tbody id="table-body"></tbody>
    </table>
    <div class="no-results" id="no-results" style="display:none">No datasets match your filters.</div>
  </div>

</div>

<script>
const SUMMARY = {summary_json};
const DETAIL  = {detail_json};

// Index detail by dataset_name for fast lookup
const DETAIL_MAP = {{}};
DETAIL.forEach(d => {{
  if (!DETAIL_MAP[d.dataset_name]) DETAIL_MAP[d.dataset_name] = [];
  DETAIL_MAP[d.dataset_name].push(d);
}});

let currentFilter = 'all';
let sortKey = 'dataset_name';
let sortAsc = true;
let expandedRow = null;

// ── Init ──────────────────────────────────────────────────────────────────
function init() {{
  const total = SUMMARY.length;
  const passed = SUMMARY.filter(r => r.overall_match).length;
  const failed = total - passed;
  const pcts = SUMMARY.filter(r => r.match_pct != null).map(r => r.match_pct);
  const avg = pcts.length ? (pcts.reduce((a,b)=>a+b,0)/pcts.length).toFixed(1) : '—';

  document.getElementById('c-total').textContent = total;
  document.getElementById('c-pass').textContent  = passed;
  document.getElementById('c-fail').textContent  = failed;
  document.getElementById('c-avg').textContent   = avg + (pcts.length ? '%' : '');

  const ts = SUMMARY.map(r => r.run_timestamp).filter(Boolean).sort().pop();
  document.getElementById('hdr-ts').textContent = ts ? 'Last run: ' + ts.replace('T',' ').slice(0,19) : '';
  document.getElementById('hdr-sub').textContent = total + ' datasets compared';

  applyFilters();
}}

// ── Filters & sort ────────────────────────────────────────────────────────
function setFilter(f, btn) {{
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}}

function applyFilters() {{
  const q = document.getElementById('search-input').value.toLowerCase();
  let rows = SUMMARY.filter(r => {{
    const nameOk = r.dataset_name.toLowerCase().includes(q);
    const filterOk = currentFilter === 'all'
      ? true
      : currentFilter === 'pass' ? r.overall_match : !r.overall_match;
    return nameOk && filterOk;
  }});

  rows.sort((a,b) => {{
    let va = a[sortKey], vb = b[sortKey];
    if (va == null) return 1; if (vb == null) return -1;
    if (typeof va === 'boolean') {{ va = va?1:0; vb = vb?1:0; }}
    return sortAsc ? (va>vb?1:-1) : (va<vb?1:-1);
  }});

  renderTable(rows);
}}

function sortBy(key) {{
  if (sortKey === key) sortAsc = !sortAsc;
  else {{ sortKey = key; sortAsc = true; }}
  applyFilters();
}}

// ── Table rendering ───────────────────────────────────────────────────────
function renderTable(rows) {{
  const tbody = document.getElementById('table-body');
  const noRes = document.getElementById('no-results');
  tbody.innerHTML = '';
  expandedRow = null;

  if (!rows.length) {{
    noRes.style.display = 'block';
    return;
  }}
  noRes.style.display = 'none';

  rows.forEach((r, i) => {{
    const tr = document.createElement('tr');
    tr.id = 'row-' + i;
    tr.onclick = () => toggleDetail(i, r);

    const pct = r.match_pct != null ? r.match_pct : 0;
    const pctColor = pct >= 95 ? '#28a745' : pct >= 70 ? '#ffc107' : '#dc3545';

    let statusBadge;
    if (r.error) statusBadge = '<span class="badge badge-error">ERROR</span>';
    else if (r.data_status === 'SKIPPED') statusBadge = '<span class="badge badge-skip">SKIPPED</span>';
    else if (r.overall_match) statusBadge = '<span class="badge badge-pass">PASS</span>';
    else statusBadge = '<span class="badge badge-fail">FAIL</span>';

    const ts = r.run_timestamp ? r.run_timestamp.replace('T',' ').slice(0,19) : '—';
    const colText = r.total_columns != null
      ? r.matching_columns + ' / ' + r.total_columns
      : '—';

    tr.innerHTML = `
      <td><strong>${{r.dataset_name}}</strong></td>
      <td>${{statusBadge}}</td>
      <td>
        <div class="pct-bar-wrap">
          <div class="pct-bar"><div class="pct-fill" style="width:${{pct}}%;background:${{pctColor}}"></div></div>
          <div class="pct-label">${{pct != null ? pct + '%' : '—'}}</div>
        </div>
      </td>
      <td>${{colText}}</td>
      <td>${{fmt(r.df1_row_count)}}</td>
      <td>${{fmt(r.df2_row_count)}}</td>
      <td style="font-size:12px;color:#666">${{ts}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

// ── Detail panel ──────────────────────────────────────────────────────────
function toggleDetail(i, r) {{
  const existingPanel = document.getElementById('panel-' + i);
  if (existingPanel) {{
    existingPanel.remove();
    document.getElementById('row-' + i).classList.remove('expanded');
    expandedRow = null;
    return;
  }}

  // Close any open panel
  if (expandedRow !== null) {{
    const prev = document.getElementById('panel-' + expandedRow);
    if (prev) prev.remove();
    const prevRow = document.getElementById('row-' + expandedRow);
    if (prevRow) prevRow.classList.remove('expanded');
  }}
  expandedRow = i;
  document.getElementById('row-' + i).classList.add('expanded');

  const cols = DETAIL_MAP[r.dataset_name] || [];
  const panelTr = document.createElement('tr');
  panelTr.id = 'panel-' + i;
  panelTr.className = 'detail-panel';

  let schemaWarn = '';
  const schema = r.schema_match === false
    ? '<div class="schema-issues">⚠️ Schema mismatch detected — some columns have different data types between DF1 and DF2.</div>'
    : '';

  let colHtml = '';
  if (!cols.length) {{
    colHtml = '<div style="color:#888;font-size:13px">No column detail available (data comparison may have been skipped).</div>';
  }} else {{
    const sorted = [...cols].sort((a,b) => a.all_values_match === b.all_values_match ? 0 : a.all_values_match ? 1 : -1);
    colHtml = '<div class="col-grid">' + sorted.map(c => renderColCard(c)).join('') + '</div>';
  }}

  panelTr.innerHTML = `<td colspan="7">
    <div class="detail-inner">
      <div class="detail-title">Column-level breakdown — ${{r.dataset_name}}</div>
      ${{schema}}
      ${{colHtml}}
    </div>
  </td>`;

  document.getElementById('row-' + i).insertAdjacentElement('afterend', panelTr);
}}

function renderColCard(c) {{
  const ok = c.all_values_match;
  const pct = c.match_pct != null ? c.match_pct : 0;
  const fillColor = pct >= 95 ? '#28a745' : pct >= 70 ? '#ffc107' : '#dc3545';

  let statsHtml = '';
  if (c.comparison_type === 'numeric') {{
    statsHtml = `
      <div class="stat-row"><span>Exact matches</span><span class="stat-val">${{fmt(c.exact_matches)}}</span></div>
      <div class="stat-row"><span>Mismatches</span><span class="stat-val">${{fmt(c.mismatches)}}</span></div>
      <div class="stat-row"><span>Avg diff</span><span class="stat-val">${{fmtF(c.avg_diff)}}</span></div>
      <div class="stat-row"><span>Min diff</span><span class="stat-val">${{fmtF(c.min_diff)}}</span></div>
      <div class="stat-row"><span>Max diff</span><span class="stat-val">${{fmtF(c.max_diff)}}</span></div>
    `;
  }} else if (c.comparison_type === 'string_fuzzy') {{
    statsHtml = `
      <div class="stat-row"><span>Exact matches</span><span class="stat-val">${{fmt(c.exact_matches)}}</span></div>
      <div class="stat-row"><span>Mismatches</span><span class="stat-val">${{fmt(c.mismatches)}}</span></div>
      <div class="stat-row"><span>Avg fuzzy score</span><span class="stat-val">${{fmtF(c.avg_fuzzy_score)}}</span></div>
      <div class="stat-row"><span>Min fuzzy score</span><span class="stat-val">${{fmtF(c.min_fuzzy_score)}}</span></div>
    `;
  }}

  let samplesHtml = '';
  if (!ok && c.sample_mismatches && c.sample_mismatches.length) {{
    samplesHtml = '<div class="sample-list">';
    c.sample_mismatches.slice(0,3).forEach((s, idx) => {{
      if (c.comparison_type === 'numeric') {{
        samplesHtml += `<div class="sample-item">#${{idx+1}} DF1: ${{s.df1_value}} → DF2: ${{s.df2_value}} (Δ ${{fmtF(s.difference)}})</div>`;
      }} else {{
        samplesHtml += `<div class="sample-item">#${{idx+1}} "${{s.df1_value}}" → "${{s.df2_value}}" (score: ${{fmtF(s.fuzzy_score)}})</div>`;
      }}
    }});
    samplesHtml += '</div>';
  }}

  return `
    <div class="col-card ${{ok ? '' : 'mismatch'}}">
      <div class="col-card-header">
        <div>
          <div class="col-name">${{c.column_name}}</div>
          <div class="col-type">${{c.data_type || ''}} · ${{c.comparison_type || ''}}</div>
        </div>
        <div class="col-pct ${{ok ? 'ok' : 'bad'}}">${{pct}}%</div>
      </div>
      <div class="mini-bar"><div class="mini-fill" style="width:${{pct}}%;background:${{fillColor}}"></div></div>
      ${{statsHtml}}
      ${{samplesHtml}}
    </div>
  `;
}}

// ── Formatters ────────────────────────────────────────────────────────────
function fmt(v)  {{ return v != null ? Number(v).toLocaleString() : '—'; }}
function fmtF(v) {{ return v != null ? parseFloat(v).toFixed(4) : '—'; }}

init();
</script>
</body>
</html>
"""


# =========================================================================
# RUN
# =========================================================================
# render_dashboard(spark)
