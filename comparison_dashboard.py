# =========================================================================
# SYNAPSE NOTEBOOK CELL - Multi-Dataset Comparison Dashboard
#
# Pass your ready DataFrames directly — no ADLS reads, no pipeline needed.
#
# USAGE:
#   datasets = [
#       {
#           "name":     "claims",          # display name for this dataset pair
#           "df1":      sas_claims_df,     # your SAS-sourced DataFrame
#           "df1_name": "SAS_claims",      # label shown in the UI for df1
#           "df2":      spark_claims_df,   # your PySpark DataFrame
#           "df2_name": "PySpark_claims",  # label shown in the UI for df2
#           "key_cols": ["claim_id"],      # join keys (or [] for row-order match)
#       },
#       {
#           "name":     "members",
#           "df1":      sas_members_df,
#           "df1_name": "SAS_members",
#           "df2":      spark_members_df,
#           "df2_name": "PySpark_members",
#           "key_cols": ["member_id", "effective_date"],
#       },
#   ]
#   render_dashboard(spark, datasets)
# =========================================================================

from IPython.display import display, HTML
import json
from datetime import datetime

# ── Paste (or import) your DataFrameComparator class before this cell ─────
# from dataframe_comparator import DataFrameComparator


def run_comparisons(spark, datasets: list) -> tuple:
    """
    Run DataFrameComparator for each dataset dict and return
    (summary_records, detail_records) as plain Python lists — no Spark writes.

    Each dataset dict must have keys:
        name, df1, df1_name, df2, df2_name, key_cols
    """
    summary_records = []
    detail_records  = []
    run_ts = datetime.now().isoformat()

    for ds in datasets:
        name     = ds["name"]
        df1      = ds["df1"]
        df2      = ds["df2"]
        df1_name = ds.get("df1_name", "DF1")
        df2_name = ds.get("df2_name", "DF2")
        key_cols = ds.get("key_cols") or None

        print(f"\n{'='*60}")
        print(f"Comparing: {name}  ({df1_name} vs {df2_name})")
        print(f"{'='*60}")

        try:
            comparator = DataFrameComparator(
                spark=spark,
                df1=df1,
                df2=df2,
                key_columns=key_cols,
                df1_name=df1_name,
                df2_name=df2_name,
            )
            report = comparator.compare()

            # ── flatten to summary row ───────────────────────────────────
            rc       = report.get("row_count", {})
            cols_rpt = report.get("columns", {})
            data_rpt = report.get("data_comparison", {})
            col_det  = data_rpt.get("column_details", {})

            total_cols    = len(col_det) if col_det else cols_rpt.get("column_count_df1", 0)
            matching_cols = sum(1 for d in col_det.values() if d.get("all_values_match", False))
            match_pct     = round(matching_cols / total_cols * 100, 2) if total_cols > 0 else 0.0

            summary_records.append({
                "dataset_name":    name,
                "df1_name":        df1_name,
                "df2_name":        df2_name,
                "run_timestamp":   run_ts,
                "overall_match":   report.get("overall_match", False),
                "row_count_match": rc.get("match", False),
                "columns_match":   report.get("columns_match", False),
                "schema_match":    report.get("schema", {}).get("schemas_match", False),
                "data_status":     data_rpt.get("status", "SKIPPED"),
                "df1_row_count":   rc.get("df1_count", 0),
                "df2_row_count":   rc.get("df2_count", 0),
                "row_diff":        rc.get("difference", 0),
                "total_columns":   total_cols,
                "matching_columns": matching_cols,
                "match_pct":       match_pct,
                "extra_in_df1":    cols_rpt.get("only_in_df1", []),
                "extra_in_df2":    cols_rpt.get("only_in_df2", []),
                "error":           None,
            })

            # ── flatten to detail rows ───────────────────────────────────
            for col_name, d in col_det.items():
                stats = d.get("statistics", {})
                detail_records.append({
                    "dataset_name":    name,
                    "df1_name":        df1_name,
                    "df2_name":        df2_name,
                    "column_name":     col_name,
                    "data_type":       d.get("data_type", "unknown"),
                    "comparison_type": d.get("comparison_type", "unknown"),
                    "all_values_match": d.get("all_values_match", False),
                    "match_pct":       d.get("match_percentage", 0.0),
                    "exact_matches":   d.get("exact_matches", 0),
                    "mismatches":      d.get("mismatches", 0),
                    "only_in_df1":     d.get("only_in_df1", 0),
                    "only_in_df2":     d.get("only_in_df2", 0),
                    "avg_diff":        stats.get("average_difference"),
                    "min_diff":        stats.get("min_difference"),
                    "max_diff":        stats.get("max_difference"),
                    "avg_fuzzy_score": stats.get("average_fuzzy_score"),
                    "min_fuzzy_score": stats.get("min_fuzzy_score"),
                    "sample_mismatches":    d.get("sample_mismatches", []),
                    "sample_only_in_df1":   d.get("sample_only_in_df1", []),
                    "sample_only_in_df2":   d.get("sample_only_in_df2", []),
                })

            status = "PASS" if report.get("overall_match") else "FAIL"
            print(f"→ {status}  |  {match_pct}% data match  |  {matching_cols}/{total_cols} columns matched\n")

        except Exception as e:
            import traceback
            print(f"ERROR comparing [{name}]: {e}")
            traceback.print_exc()
            summary_records.append({
                "dataset_name": name, "df1_name": df1_name, "df2_name": df2_name,
                "run_timestamp": run_ts, "overall_match": False,
                "row_count_match": False, "columns_match": False,
                "schema_match": False, "data_status": "ERROR",
                "df1_row_count": None, "df2_row_count": None, "row_diff": None,
                "total_columns": None, "matching_columns": None, "match_pct": None,
                "extra_in_df1": [], "extra_in_df2": [], "error": str(e),
            })

    return summary_records, detail_records


def render_dashboard(spark, datasets: list):
    """
    Main entry point.
    Runs all comparisons then renders the interactive multi-dataset dashboard.
    """
    summary_records, detail_records = run_comparisons(spark, datasets)

    summary_json = json.dumps(summary_records, default=str)
    detail_json  = json.dumps(detail_records,  default=str)

    display(HTML(_build_html(summary_json, detail_json)))


# =========================================================================
# HTML / JS DASHBOARD
# =========================================================================

def _build_html(summary_json: str, detail_json: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Tahoma, Verdana, sans-serif; background: #f0f2f5; color: #333; }}

  .db-wrap {{ max-width: 1400px; margin: 0 auto; padding: 24px 16px; }}

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
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-top: 4px solid #667eea;
  }}
  .card.pass  {{ border-top-color: #28a745; }}
  .card.fail  {{ border-top-color: #dc3545; }}
  .card.warn  {{ border-top-color: #ffc107; }}
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

  /* ── Main table ── */
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
  thead th .sa {{ margin-left: 4px; color: #aaa; font-size: 10px; }}
  tbody tr {{ border-bottom: 1px solid #f0f0f0; cursor: pointer; transition: background 0.1s; }}
  tbody tr:hover {{ background: #f7f8ff; }}
  tbody tr.expanded {{ background: #f0f3ff; }}
  tbody td {{ padding: 12px 16px; vertical-align: middle; }}

  .badge {{
    display: inline-block; padding: 4px 12px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
  }}
  .badge-pass  {{ background: #d4edda; color: #155724; }}
  .badge-fail  {{ background: #f8d7da; color: #721c24; }}
  .badge-error {{ background: #fff3cd; color: #856404; }}
  .badge-skip  {{ background: #e2e3e5; color: #383d41; }}

  .pct-bar-wrap {{ width: 120px; }}
  .pct-bar  {{ height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; }}
  .pct-fill {{ height: 100%; border-radius: 4px; }}
  .pct-label {{ font-size: 11px; color: #666; margin-top: 2px; }}

  .extra-tag {{
    display: inline-block; background: #fff3cd; color: #856404;
    font-size: 11px; padding: 2px 7px; border-radius: 10px; margin: 2px 2px 0 0;
  }}

  /* ── Detail panel ── */
  .detail-panel {{ background: #f7f8ff; }}
  .detail-panel td {{ padding: 0 !important; }}
  .detail-inner {{ padding: 20px 24px; }}

  .detail-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 16px; flex-wrap: wrap; gap: 8px;
  }}
  .detail-title {{ font-weight: 700; font-size: 16px; color: #333; }}
  .detail-names {{ font-size: 13px; color: #667eea; font-weight: 600; }}

  /* column filter inside detail panel */
  .col-filter-bar {{
    display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; align-items: center;
  }}
  .col-filter-bar input {{
    border: 1px solid #ddd; border-radius: 6px; padding: 6px 10px;
    font-size: 13px; outline: none; width: 200px;
  }}
  .col-filter-bar input:focus {{ border-color: #667eea; }}
  .cfilter-btn {{
    padding: 5px 12px; border: 1px solid #ddd; border-radius: 14px;
    background: white; cursor: pointer; font-size: 12px; transition: all 0.15s;
  }}
  .cfilter-btn.active {{ background: #667eea; color: white; border-color: #667eea; }}

  .schema-warn {{
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
    padding: 10px 14px; margin-bottom: 14px; font-size: 13px;
  }}
  .extra-cols-warn {{
    background: #e8f4f8; border: 1px solid #bee5eb; border-radius: 6px;
    padding: 10px 14px; margin-bottom: 14px; font-size: 13px;
  }}

  .col-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 12px;
  }}
  .col-card {{
    background: white; border-radius: 8px; padding: 14px;
    border-left: 4px solid #28a745; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .col-card.mismatch {{ border-left-color: #dc3545; }}
  .col-card-header {{
    display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;
  }}
  .col-name  {{ font-weight: 700; font-size: 13px; }}
  .col-type  {{ font-size: 11px; color: #888; margin-top: 2px; }}
  .col-pct   {{ font-size: 13px; font-weight: 700; white-space: nowrap; }}
  .col-pct.ok  {{ color: #28a745; }}
  .col-pct.bad {{ color: #dc3545; }}

  .mini-bar {{ height: 6px; background: #e9ecef; border-radius: 3px; overflow: hidden; margin: 6px 0 8px; }}
  .mini-fill {{ height: 100%; border-radius: 3px; }}

  .stat-row {{
    display: flex; justify-content: space-between;
    font-size: 12px; color: #666; margin: 3px 0;
  }}
  .stat-val {{ font-weight: 600; color: #333; }}

  .sample-section {{ margin-top: 10px; }}
  .sample-section-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    color: #888; letter-spacing: 0.4px; margin-bottom: 4px;
  }}
  .sample-item {{
    font-family: monospace; font-size: 11px; padding: 4px 8px;
    margin: 3px 0; border-radius: 3px;
  }}
  .sample-item.diff     {{ background: #fff5f5; border-left: 2px solid #dc3545; }}
  .sample-item.only-df1 {{ background: #fff8e1; border-left: 2px solid #ffc107; }}
  .sample-item.only-df2 {{ background: #e8f5e9; border-left: 2px solid #28a745; }}

  .no-results {{ padding: 48px; text-align: center; color: #888; font-size: 15px; }}
  .no-cols    {{ color: #888; font-size: 13px; padding: 12px 0; }}
</style>
</head>
<body>
<div class="db-wrap">

  <div class="db-header">
    <div>
      <h1>📊 DataFrame Comparison Dashboard</h1>
      <div class="sub" id="hdr-sub">Loading...</div>
    </div>
    <div style="text-align:right;font-size:13px;opacity:0.85">
      <div id="hdr-ts"></div>
    </div>
  </div>

  <div class="cards">
    <div class="card">       <div class="card-label">Total Datasets</div><div class="card-value" id="c-total">—</div></div>
    <div class="card pass">  <div class="card-label">Passed</div>        <div class="card-value" id="c-pass">—</div></div>
    <div class="card fail">  <div class="card-label">Failed</div>        <div class="card-value" id="c-fail">—</div></div>
    <div class="card warn">  <div class="card-label">Avg Match %</div>   <div class="card-value" id="c-avg">—</div></div>
  </div>

  <div class="filter-bar">
    <input type="text" id="search-input" placeholder="Search dataset name..." oninput="applyFilters()">
    <div class="filter-group">
      <button class="filter-btn active" onclick="setFilter('all',  this)">All</button>
      <button class="filter-btn"        onclick="setFilter('fail', this)">Failures only</button>
      <button class="filter-btn"        onclick="setFilter('pass', this)">Passes only</button>
    </div>
  </div>

  <div class="table-wrap">
    <table id="main-table">
      <thead>
        <tr>
          <th onclick="sortBy('dataset_name')">Dataset <span class="sa">⇅</span></th>
          <th onclick="sortBy('overall_match')">Status <span class="sa">⇅</span></th>
          <th onclick="sortBy('match_pct')">Data Match <span class="sa">⇅</span></th>
          <th onclick="sortBy('matching_columns')">Columns Matched</th>
          <th>Source Rows</th>
          <th>Target Rows</th>
          <th>Extra Columns</th>
          <th onclick="sortBy('run_timestamp')">Run Time <span class="sa">⇅</span></th>
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

// index detail by dataset_name for fast drill-down
const DETAIL_MAP = {{}};
DETAIL.forEach(d => {{
  if (!DETAIL_MAP[d.dataset_name]) DETAIL_MAP[d.dataset_name] = [];
  DETAIL_MAP[d.dataset_name].push(d);
}});

let currentFilter = 'all';
let sortKey = 'dataset_name';
let sortAsc  = true;
let expandedRow = null;

// active column filter per open panel
let colFilter = 'all';
let colSearch  = '';

// ── Init ──────────────────────────────────────────────────────────────────
function init() {{
  const total  = SUMMARY.length;
  const passed = SUMMARY.filter(r => r.overall_match).length;
  const failed = total - passed;
  const pcts   = SUMMARY.filter(r => r.match_pct != null).map(r => r.match_pct);
  const avg    = pcts.length ? (pcts.reduce((a,b)=>a+b,0)/pcts.length).toFixed(1) : '—';

  document.getElementById('c-total').textContent = total;
  document.getElementById('c-pass').textContent  = passed;
  document.getElementById('c-fail').textContent  = failed;
  document.getElementById('c-avg').textContent   = avg + (pcts.length ? '%' : '');

  const ts = SUMMARY.map(r => r.run_timestamp).filter(Boolean).sort().pop();
  document.getElementById('hdr-ts').textContent  = ts ? 'Run: ' + ts.replace('T',' ').slice(0,19) : '';
  document.getElementById('hdr-sub').textContent = total + ' dataset pair' + (total!==1?'s':'') + ' compared';

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
    const nameOk   = r.dataset_name.toLowerCase().includes(q);
    const filterOk = currentFilter === 'all' ? true
                   : currentFilter === 'pass' ? r.overall_match : !r.overall_match;
    return nameOk && filterOk;
  }});

  rows.sort((a, b) => {{
    let va = a[sortKey], vb = b[sortKey];
    if (va == null) return 1; if (vb == null) return -1;
    if (typeof va === 'boolean') {{ va = va?1:0; vb = vb?1:0; }}
    if (typeof va === 'string')  return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
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

  if (!rows.length) {{ noRes.style.display = 'block'; return; }}
  noRes.style.display = 'none';

  rows.forEach((r, i) => {{
    const tr = document.createElement('tr');
    tr.id      = 'row-' + i;
    tr.onclick = () => toggleDetail(i, r);

    const pct      = r.match_pct != null ? r.match_pct : 0;
    const pctColor = pct >= 95 ? '#28a745' : pct >= 70 ? '#ffc107' : '#dc3545';

    let badge;
    if      (r.error)                    badge = '<span class="badge badge-error">ERROR</span>';
    else if (r.data_status === 'ERROR')  badge = '<span class="badge badge-error">ERROR</span>';
    else if (r.data_status === 'SKIPPED')badge = '<span class="badge badge-skip">SKIPPED</span>';
    else if (r.overall_match)            badge = '<span class="badge badge-pass">PASS</span>';
    else                                 badge = '<span class="badge badge-fail">FAIL</span>';

    const ts       = r.run_timestamp ? r.run_timestamp.replace('T',' ').slice(0,19) : '—';
    const colText  = r.total_columns != null ? r.matching_columns + ' / ' + r.total_columns : '—';
    const srcLabel = r.df1_name || 'Source';
    const tgtLabel = r.df2_name || 'Target';

    // extra columns tags
    let extraHtml = '';
    (r.extra_in_df1 || []).forEach(c => extraHtml += `<span class="extra-tag" title="only in ${{srcLabel}}">${{c}} (${{srcLabel}})</span>`);
    (r.extra_in_df2 || []).forEach(c => extraHtml += `<span class="extra-tag" title="only in ${{tgtLabel}}">${{c}} (${{tgtLabel}})</span>`);
    if (!extraHtml) extraHtml = '<span style="color:#aaa;font-size:12px">—</span>';

    tr.innerHTML = `
      <td>
        <strong>${{r.dataset_name}}</strong>
        <div style="font-size:11px;color:#888;margin-top:2px">${{srcLabel}} vs ${{tgtLabel}}</div>
      </td>
      <td>${{badge}}</td>
      <td>
        <div class="pct-bar-wrap">
          <div class="pct-bar"><div class="pct-fill" style="width:${{pct}}%;background:${{pctColor}}"></div></div>
          <div class="pct-label">${{r.match_pct != null ? pct + '%' : '—'}}</div>
        </div>
      </td>
      <td>${{colText}}</td>
      <td>${{fmt(r.df1_row_count)}}</td>
      <td>${{fmt(r.df2_row_count)}}</td>
      <td style="max-width:220px;white-space:normal">${{extraHtml}}</td>
      <td style="font-size:12px;color:#666">${{ts}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

// ── Detail panel ──────────────────────────────────────────────────────────
function toggleDetail(i, r) {{
  const existing = document.getElementById('panel-' + i);
  if (existing) {{
    existing.remove();
    document.getElementById('row-' + i).classList.remove('expanded');
    expandedRow = null;
    return;
  }}
  if (expandedRow !== null) {{
    const prev = document.getElementById('panel-' + expandedRow);
    if (prev) prev.remove();
    const prevRow = document.getElementById('row-' + expandedRow);
    if (prevRow) prevRow.classList.remove('expanded');
  }}
  expandedRow = i;
  colFilter = 'all';
  colSearch  = '';
  document.getElementById('row-' + i).classList.add('expanded');
  renderDetailPanel(i, r);
}}

function renderDetailPanel(i, r) {{
  const old = document.getElementById('panel-' + i);
  if (old) old.remove();

  const cols    = DETAIL_MAP[r.dataset_name] || [];
  const src     = r.df1_name || 'Source';
  const tgt     = r.df2_name || 'Target';
  const panelTr = document.createElement('tr');
  panelTr.id        = 'panel-' + i;
  panelTr.className = 'detail-panel';

  // warnings
  let warns = '';
  if (!r.schema_match) warns += `<div class="schema-warn">⚠️ Schema mismatch — some common columns have different data types between <strong>${{src}}</strong> and <strong>${{tgt}}</strong>.</div>`;
  const extras1 = r.extra_in_df1 || [];
  const extras2 = r.extra_in_df2 || [];
  if (extras1.length || extras2.length) {{
    let extMsg = `<div class="extra-cols-warn">ℹ️ Column count differs — only common columns were compared.<br>`;
    if (extras1.length) extMsg += `<strong>Extra in ${{src}}:</strong> ${{extras1.join(', ')}}<br>`;
    if (extras2.length) extMsg += `<strong>Extra in ${{tgt}}:</strong> ${{extras2.join(', ')}}`;
    extMsg += `</div>`;
    warns += extMsg;
  }}

  // column filter bar
  const filterId  = 'cf-search-'  + i;
  const filterAll = 'cf-all-'     + i;
  const filterFail= 'cf-fail-'    + i;
  const filterPass= 'cf-pass-'    + i;
  const gridId    = 'col-grid-'   + i;

  const filterBar = `
    <div class="col-filter-bar">
      <input id="${{filterId}}" type="text" placeholder="Filter columns..." oninput="filterCols(${{i}})">
      <button id="${{filterAll}}"  class="cfilter-btn active" onclick="setCFilter(${{i}},'all',  this)">All</button>
      <button id="${{filterFail}}" class="cfilter-btn"        onclick="setCFilter(${{i}},'fail', this)">Mismatches</button>
      <button id="${{filterPass}}" class="cfilter-btn"        onclick="setCFilter(${{i}},'pass', this)">Matches</button>
    </div>
  `;

  let colsHtml = '<div class="no-cols">No column detail available (comparison may have been skipped or errored).</div>';
  if (cols.length) {{
    const sorted = [...cols].sort((a,b) => a.all_values_match===b.all_values_match ? 0 : a.all_values_match?1:-1);
    colsHtml = `<div class="col-grid" id="${{gridId}}">${{sorted.map(c => renderColCard(c, src, tgt)).join('')}}</div>`;
  }}

  panelTr.innerHTML = `<td colspan="8">
    <div class="detail-inner">
      <div class="detail-header">
        <div class="detail-title">Column breakdown — ${{r.dataset_name}}</div>
        <div class="detail-names">${{src}} &nbsp;vs&nbsp; ${{tgt}}</div>
      </div>
      ${{warns}}
      ${{filterBar}}
      ${{colsHtml}}
    </div>
  </td>`;

  document.getElementById('row-' + i).insertAdjacentElement('afterend', panelTr);
}}

function setCFilter(i, f, btn) {{
  colFilter = f;
  btn.closest('.col-filter-bar').querySelectorAll('.cfilter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyColFilter(i);
}}

function filterCols(i) {{
  const inp = document.getElementById('cf-search-' + i);
  colSearch = inp ? inp.value.toLowerCase() : '';
  applyColFilter(i);
}}

function applyColFilter(i) {{
  const grid = document.getElementById('col-grid-' + i);
  if (!grid) return;
  grid.querySelectorAll('.col-card').forEach(card => {{
    const name   = (card.dataset.colname  || '').toLowerCase();
    const match  =  card.dataset.match === 'true';
    const nameOk = name.includes(colSearch);
    const filterOk = colFilter === 'all' ? true : colFilter === 'pass' ? match : !match;
    card.style.display = (nameOk && filterOk) ? '' : 'none';
  }});
}}

function renderColCard(c, src, tgt) {{
  const ok  = c.all_values_match;
  const pct = c.match_pct != null ? c.match_pct : 0;
  const fillColor = pct >= 95 ? '#28a745' : pct >= 70 ? '#ffc107' : '#dc3545';

  let statsHtml = '';
  if (c.comparison_type === 'numeric') {{
    statsHtml = `
      <div class="stat-row"><span>Exact matches</span>   <span class="stat-val">${{fmt(c.exact_matches)}}</span></div>
      <div class="stat-row"><span>Mismatches</span>       <span class="stat-val">${{fmt(c.mismatches)}}</span></div>
      <div class="stat-row"><span>Only in ${{src}}</span>  <span class="stat-val">${{fmt(c.only_in_df1)}}</span></div>
      <div class="stat-row"><span>Only in ${{tgt}}</span>  <span class="stat-val">${{fmt(c.only_in_df2)}}</span></div>
      <div class="stat-row"><span>Avg diff</span>         <span class="stat-val">${{fmtF(c.avg_diff)}}</span></div>
      <div class="stat-row"><span>Min diff</span>         <span class="stat-val">${{fmtF(c.min_diff)}}</span></div>
      <div class="stat-row"><span>Max diff</span>         <span class="stat-val">${{fmtF(c.max_diff)}}</span></div>
    `;
  }} else if (c.comparison_type === 'string_fuzzy') {{
    statsHtml = `
      <div class="stat-row"><span>Exact matches</span>    <span class="stat-val">${{fmt(c.exact_matches)}}</span></div>
      <div class="stat-row"><span>Mismatches</span>        <span class="stat-val">${{fmt(c.mismatches)}}</span></div>
      <div class="stat-row"><span>Only in ${{src}}</span>  <span class="stat-val">${{fmt(c.only_in_df1)}}</span></div>
      <div class="stat-row"><span>Only in ${{tgt}}</span>  <span class="stat-val">${{fmt(c.only_in_df2)}}</span></div>
      <div class="stat-row"><span>Avg fuzzy score</span>  <span class="stat-val">${{fmtF(c.avg_fuzzy_score)}}</span></div>
      <div class="stat-row"><span>Min fuzzy score</span>  <span class="stat-val">${{fmtF(c.min_fuzzy_score)}}</span></div>
    `;
  }}

  // sample rows — 3 sections: value differences, only-in-src, only-in-tgt
  let samplesHtml = '';

  if (!ok) {{
    const diffs = c.sample_mismatches || [];
    if (diffs.length) {{
      samplesHtml += `<div class="sample-section"><div class="sample-section-title">Value differences</div>`;
      diffs.slice(0,3).forEach((s,idx) => {{
        if (c.comparison_type === 'numeric') {{
          const d = s.difference;
          const dStr = (d != null && typeof d === 'number') ? d.toFixed(4) : String(d);
          samplesHtml += `<div class="sample-item diff">#${{idx+1}} Key:${{s.key}}  ${{src}}:${{s.df1_value}} → ${{tgt}}:${{s.df2_value}}  (Δ ${{dStr}})</div>`;
        }} else {{
          samplesHtml += `<div class="sample-item diff">#${{idx+1}} Key:${{s.key}}  ${{src}}:"${{s.df1_value}}" → ${{tgt}}:"${{s.df2_value}}"  (score:${{fmtF(s.fuzzy_score)}})</div>`;
        }}
      }});
      samplesHtml += `</div>`;
    }}

    const df1only = c.sample_only_in_df1 || [];
    if (df1only.length) {{
      samplesHtml += `<div class="sample-section"><div class="sample-section-title">Only in ${{src}}</div>`;
      df1only.slice(0,3).forEach((s,idx) => {{
        samplesHtml += `<div class="sample-item only-df1">#${{idx+1}} Key:${{s.key}}  value:${{s.df1_value}}</div>`;
      }});
      samplesHtml += `</div>`;
    }}

    const df2only = c.sample_only_in_df2 || [];
    if (df2only.length) {{
      samplesHtml += `<div class="sample-section"><div class="sample-section-title">Only in ${{tgt}}</div>`;
      df2only.slice(0,3).forEach((s,idx) => {{
        samplesHtml += `<div class="sample-item only-df2">#${{idx+1}} Key:${{s.key}}  value:${{s.df2_value}}</div>`;
      }});
      samplesHtml += `</div>`;
    }}
  }}

  return `
    <div class="col-card ${{ok?'':'mismatch'}}" data-colname="${{c.column_name}}" data-match="${{ok}}">
      <div class="col-card-header">
        <div>
          <div class="col-name">${{c.column_name}}</div>
          <div class="col-type">${{c.data_type||''}} · ${{c.comparison_type||''}}</div>
        </div>
        <div class="col-pct ${{ok?'ok':'bad'}}">${{pct}}%</div>
      </div>
      <div class="mini-bar"><div class="mini-fill" style="width:${{pct}}%;background:${{fillColor}}"></div></div>
      ${{statsHtml}}
      ${{samplesHtml}}
    </div>
  `;
}}

// ── Formatters ────────────────────────────────────────────────────────────
function fmt(v)  {{ return v != null ? Number(v).toLocaleString() : '—'; }}
function fmtF(v) {{ return v != null ? parseFloat(v).toFixed(4)   : '—'; }}

init();
</script>
</body>
</html>
"""


# =========================================================================
# USAGE EXAMPLE
# =========================================================================
# datasets = [
#     {
#         "name":     "claims",
#         "df1":      sas_claims_df,
#         "df1_name": "SAS_claims",
#         "df2":      spark_claims_df,
#         "df2_name": "PySpark_claims",
#         "key_cols": ["claim_id"],
#     },
#     {
#         "name":     "members",
#         "df1":      sas_members_df,
#         "df1_name": "SAS_members",
#         "df2":      spark_members_df,
#         "df2_name": "PySpark_members",
#         "key_cols": ["member_id"],
#     },
# ]
# render_dashboard(spark, datasets)
