

import React, { useMemo, useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import * as XLSX from "xlsx";
import { jsPDF } from "jspdf";
import "jspdf-autotable";
import Swal from "sweetalert2";
import RefineWizard from "./RefineWizard";
import { Button } from "@mui/material";



export default function ExtractionReviewPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { uploadedFilename, sku_matrix = [], aggregated_matrix = {} } =
    location.state || {};


  const [viewMode, setViewMode] = useState("config");

  // local UI state
  const [attributeSearch, setAttributeSearch] = useState("");
  const [valueSearch, setValueSearch] = useState("");
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [localAggregated, setLocalAggregated] = useState(
    aggregated_matrix || { columns: [], rows: [] }
  );
  const [selectedRows, setSelectedRows] = useState([]);
  const [selectedAttributes, setSelectedAttributes] = useState([]);

  // Rules state
  const [configRules, setConfigRules] = useState({ rules: [], generated_at: null });
  const [pricingRules, setPricingRules] = useState({ rules: [], message: "Not implemented" });
  const [rulesLoading, setRulesLoading] = useState(false);

  // preview state
  const [previewExamples, setPreviewExamples] = useState([]);
  const [previewRule, setPreviewRule] = useState(null);

  const storedUser = JSON.parse(localStorage.getItem("user") || "null");
  const username = storedUser?.username || "User";

  // derive attribute list from sku_matrix (per-SKU attributes)
  const allAttributes = useMemo(() => {
    const attrs = new Set();
    sku_matrix.forEach((s) =>
      (s.attributes || []).forEach(([a]) => {
        if (a) attrs.add(a);
      })
    );
    return Array.from(attrs);
  }, [sku_matrix]);

  // Build matrixRows (per SKU) - each row object: { SKU: "text", Attr1: val, Attr2: val, ... }
  const matrixRows = useMemo(() => {
    return sku_matrix.map((skuObj) => {
      const row = { SKU: skuObj.sku };
      allAttributes.forEach((a) => {
        const found = (skuObj.attributes || []).find(([attr]) => attr === a);
        row[a] = found ? found[1] : "";
      });
      return row;
    });
  }, [sku_matrix, allAttributes]);

  // When switching to aggregated view, refresh aggregated data from backend if empty or requested
  useEffect(() => {
    if (viewMode === "aggregated" && (!localAggregated || !localAggregated.rows || !localAggregated.rows.length)) {
      fetchAggregatedFromGraph();
    }
    // when entering config_rules view, fetch rules
    if (viewMode === "config_rules") {
      fetchConfigRules();
    }
    // when entering pricing_rules view, fetch pricing rules
    if (viewMode === "pricing_rules") {
      fetchPricingRules();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode]);

  // Load latest graph + rules on full page refresh
  useEffect(() => {
    fetchAggregatedFromGraph();   // always load latest graph state
    fetchConfigRules();          // always load latest rules
  }, []); // empty dependencies → runs only on mount/refresh


  function fetchAggregatedFromGraph() {
    fetch("http://localhost:5000/graph/aggregated")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.columns && data.rows) {
          setLocalAggregated(data);
        }
      })
      .catch((err) => {
        console.error("❌ Error loading aggregated from graph:", err);
      });
  }

  async function fetchConfigRules() {
    setRulesLoading(true);
    try {
      const res = await fetch("http://localhost:5000/rules/configuration");
      const data = await res.json();
      if (res.ok) {
        setConfigRules(data || { rules: [], generated_at: null });
      } else {
        console.warn("Non-ok rules/configuration response:", data);
        setConfigRules({ rules: [], generated_at: null });
      }
    } catch (e) {
      console.error("❌ Error fetching configuration rules:", e);
      setConfigRules({ rules: [], generated_at: null });
    } finally {
      setRulesLoading(false);
    }
  }

  async function fetchPricingRules() {
    setRulesLoading(true);
    try {
      const res = await fetch("http://localhost:5000/rules/pricing");
      const data = await res.json();
      if (res.ok) {
        setPricingRules(data || { rules: [], message: "No pricing rules" });
      } else {
        setPricingRules({ rules: [], message: "No pricing rules" });
      }
    } catch (e) {
      console.error("❌ Error fetching pricing rules:", e);
      setPricingRules({ rules: [], message: "Error fetching pricing rules" });
    } finally {
      setRulesLoading(false);
    }
  }

  function toggleSelectAll(checked) {
    setSelectedRows(checked ? filteredRows.map((r) => r.SKU) : []);
  }

  // Filtering for both config and aggregated
  const filteredRows = useMemo(() => {
    if (viewMode === "config" || viewMode === "config_rules" || viewMode === "pricing_rules") {
      // config matrix filtering
      return matrixRows.filter((row) => {
        const attrMatch = Object.keys(row)
          .join(" ")
          .toLowerCase()
          .includes(attributeSearch.toLowerCase());
        const valMatch = Object.values(row)
          .join(" ")
          .toLowerCase()
          .includes(valueSearch.toLowerCase());
        return attrMatch && valMatch;
      });
    } else {
      // aggregated view filtering
      const columns = localAggregated.columns || [];
      const rows = localAggregated.rows || [];
      return rows.filter((row) => {
        const headerText = columns.join(" ").toLowerCase();
        const rowText = row.join(" ").toLowerCase();
        const attrMatch = !attributeSearch || headerText.includes(attributeSearch.toLowerCase());
        const valMatch = !valueSearch || rowText.includes(valueSearch.toLowerCase());
        return attrMatch && valMatch;
      });
    }
  }, [matrixRows, localAggregated, attributeSearch, valueSearch, viewMode]);

  // Export handlers
  const handleExport = (format) => {
    try {
      let data, filename, columns, rows;

      if (viewMode === "config" || viewMode === "config_rules" || viewMode === "pricing_rules") {
        // export configuration matrix (per SKU)
        data = matrixRows;
        filename = uploadedFilename
          ? `${uploadedFilename.replace(/\.[^/.]+$/, "")}_Configuration_Matrix`
          : "Configuration_Matrix";
      } else {
        // aggregated
        columns = localAggregated.columns || [];
        rows = localAggregated.rows || [];
        if (!columns.length || !rows.length) {
          Swal.fire("No data available to export.", "", "info");
          return;
        }
        data = rows.map((r) => Object.fromEntries(columns.map((c, i) => [c, r[i]])));
        filename = uploadedFilename
          ? `${uploadedFilename.replace(/\.[^/.]+$/, "")}_Aggregated_Attributes`
          : "Aggregated_Attributes";
      }

      if (format === "xlsx") {
        const ws = XLSX.utils.json_to_sheet(data);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(
          wb,
          ws,
          viewMode === "config" || viewMode === "config_rules" || viewMode === "pricing_rules"
            ? "Configuration"
            : "Aggregated"
        );
        XLSX.writeFile(wb, `${filename}.xlsx`);
        Swal.fire({
          icon: "success",
          title: "Excel Exported Successfully!",
          timer: 1500,
          showConfirmButton: false,
        });
      } else if (format === "pdf") {
        const doc = new jsPDF({
          orientation: "landscape",
          unit: "pt",
          format: "a4",
        });
        doc.setFontSize(14);
        doc.text(filename, 40, 40);
        if (viewMode === "config" || viewMode === "config_rules" || viewMode === "pricing_rules") {
          const headers = Object.keys(matrixRows[0] || {});
          const body = filteredRows.map((r) => Object.values(r));
          doc.autoTable({
            head: [headers],
            body,
            startY: 60,
            styles: { fontSize: 8, overflow: "linebreak" },
            headStyles: { fillColor: [99, 102, 241] },
          });
        } else {
          doc.autoTable({
            head: [localAggregated.columns || []],
            body: localAggregated.rows || [],
            startY: 60,
            styles: { fontSize: 8, overflow: "linebreak" },
            headStyles: { fillColor: [59, 130, 246] },
          });
        }
        doc.save(`${filename}.pdf`);
        Swal.fire({
          icon: "success",
          title: "PDF Exported Successfully!",
          timer: 1500,
          showConfirmButton: false,
        });
      }

      setExportMenuOpen(false);
    } catch (err) {
      Swal.fire("Error", err.message, "error");
    }
  };

  // Rule preview handler
  async function previewRuleHandler(ruleId) {
    try {
      const res = await fetch(`http://localhost:5000/rules/${encodeURIComponent(ruleId)}/preview`);
      const data = await res.json();
      if (!res.ok) {
        Swal.fire("Error", data?.error || "Failed to fetch preview");
        return;
      }
      setPreviewRule(data.rule || null);
      setPreviewExamples(data.examples || []);
      // show modal with examples
      const exampleText =
        (data.examples || []).map((ex, idx) => `${idx + 1}. ${ex}`).join("\n\n") || "No examples available.";
      await Swal.fire({
        title: `Preview`,
        html: `<pre style="text-align:left;white-space:pre-wrap;">${escapeHtml(exampleText)}</pre>`,
        width: "70%",
      });
    } catch (e) {
      console.error("❌ Error previewing rule:", e);
      Swal.fire("Error", "Failed to fetch rule preview", "error");
    }
  }

  // Utility: escape html for displaying pre text in Swal
  function escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // UI: Determine the label for the config/aggregated toggle
  const configAggregatedLabel = viewMode === "config" ? "View Aggregated Attributes" : "← Back to Configuration Matrix";

  // UI: Determine the label for the rules toggle (the second button)
  // Desired behavior:
  // - When in config or aggregated: show "View Configuration Rules"
  // - When in config_rules: show "View Pricing Rules"
  // - When in pricing_rules: show "← Back to Configuration Rules"
  function rulesToggleLabel() {
    if (viewMode === "config_rules") return "View Pricing Rules";
    if (viewMode === "pricing_rules") return "← Back to Configuration Rules";
    // otherwise (config or aggregated)
    return "View Configuration Rules";
  }

  // Handler for rules toggle button click
  function handleRulesToggleClick() {
    if (viewMode === "config_rules") {
      // move to pricing rules
      setViewMode("pricing_rules");
    } else if (viewMode === "pricing_rules") {
      // back to config rules
      setViewMode("config_rules");
    } else {
      // from config or aggregated -> open configuration rules
      setViewMode("config_rules");
    }
  }

  // Handler for config/aggregated toggle
  function handleConfigAggregatedToggle() {
    if (viewMode === "config") {
      setViewMode("aggregated");
    } else if (viewMode === "aggregated") {
      setViewMode("config");
    } else {
      // if we're in rules pages and user toggles main, bring them back to config
      setViewMode("config");
    }
  }

  // Refresh graph aggregated data (exposed to RefineWizard)
  const refreshGraph = () => {
    fetchAggregatedFromGraph();
  };

  // Recompute rules from frontend for manual recompute (calls backend /rules/recompute)
  async function recomputeRulesFromFrontend() {
    try {
      setRulesLoading(true);
      const res = await fetch("http://localhost:5000/rules/recompute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sku_matrix }),
      });
      const data = await res.json();
      if (res.ok) {
        setConfigRules(data);
        Swal.fire({ icon: "success", title: "Rules recomputed", timer: 1200, showConfirmButton: false });
      } else {
        Swal.fire("Error recomputing", data?.error || "Unknown error", "error");
      }
    } catch (e) {
      console.error("❌ Error recomputing rules:", e);
      Swal.fire("Error", "Failed to recompute rules", "error");
    } finally {
      setRulesLoading(false);
    }
  }

  // Row render helpers for rules table
  function renderConfigRulesTable() {
    const rules = configRules?.rules || [];
    if (rulesLoading) {
      return <div style={{ padding: 20 }}>Loading configuration rules...</div>;
    }
    if (!rules.length) {
      return <div style={{ padding: 20 }}>No configuration rules found.</div>;
    }

    return (
      <div style={{ overflow: "auto", borderRadius: 8 }}>
        <table style={tableStyle}>
          <thead style={theadStyle}>
            <tr>
              <th style={{ ...thStyle, width: 40 }}>#</th>
              <th style={thStyle}>Rule</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((r, idx) => (
              <tr key={r.id || idx}>
                <td style={tdStyle}>{idx + 1}</td>
                <td style={tdStyle}>
                  <strong>{r.rule_text || formatRuleReadable(r)}</strong>
                </td>

                 <td style={{ ...tdStyle, textAlign: "center" }}>
                <button
                  style={smallButtonStyle}
                  onClick={() => previewRuleHandler(r.id)}
                >
                  Preview examples
                </button>

                <button
                  style={{ ...smallButtonStyle, marginLeft: 10 }}
                  onClick={() => {
                    navigator.clipboard?.writeText(
                      r.rule_text || formatRuleReadable(r)
                    );
                    Swal.fire({
                      icon: "success",
                      title: "Copied!",
                      timer: 900,
                      showConfirmButton: false,
                    });
                  }}
                >
                  Copy
                </button>
              </td>
              
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  function renderPricingRulesTable() {
    const rules = pricingRules?.rules || [];
    if (rulesLoading) {
      return <div style={{ padding: 20 }}>Loading pricing rules...</div>;
    }
    if (!rules.length) {
      return <div style={{ padding: 20 }}>{pricingRules?.message || "No pricing rules available yet."}</div>;
    }

    return (
      <div style={{ overflow: "auto", borderRadius: 8 }}>
        <table style={tableStyle}>
          <thead style={theadStyle}>
            <tr>
              <th style={{ ...thStyle, width: 40 }}>#</th>
              <th style={thStyle}>Pricing Rule</th>
              <th style={{ ...thStyle, width: 140 }}>Notes</th>
              <th style={{ ...thStyle, width: 160 }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((r, idx) => (
              <tr key={r.id || idx}>
                <td style={tdStyle}>{idx + 1}</td>
                <td style={tdStyle}>{r.rule_text || JSON.stringify(r)}</td>
                <td style={tdStyle}>{r.note || ""}</td>
                <td style={{ ...tdStyle, textAlign: "center" }}>
                  <button
                    style={smallButtonStyle}
                    onClick={() => {
                      navigator.clipboard?.writeText(r.rule_text || JSON.stringify(r));
                      Swal.fire({ icon: "success", title: "Copied to clipboard", timer: 900, showConfirmButton: false });
                    }}
                  >
                    Copy
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  function formatRuleReadable(rule) {
  try {
    // Case 1: Your backend uses "if" and "then" keys
    if (rule.if && rule.then) {
      const left = Object.entries(rule.if)
        .map(([k, v]) => `${k} = ${v}`)
        .join(" AND ");

      const right = Object.entries(rule.then)
        .map(([k, v]) => `${k} = ${v}`)
        .join(" AND ");

      return `IF ${left} THEN ${right}`;
    }

    // Case 2: antecedent / consequent structure
    if (rule.antecedent && rule.consequent) {
      const left = Object.entries(rule.antecedent)
        .map(([k, v]) => `${k} = ${v}`)
        .join(" AND ");

      const right = Object.entries(rule.consequent)
        .map(([k, v]) => `${k} = ${v}`)
        .join(" AND ");

      return `IF ${left} THEN ${right}`;
    }

    // fallback
    return JSON.stringify(rule);
  } catch (err) {
    return "Invalid rule";
  }
}


  function formatPct(v) {
    if (v === null || v === undefined) return "-";
    if (typeof v === "number") {
      return `${(v * 100).toFixed(1)}%`;
    }
    if (!isNaN(Number(v))) {
      return `${(Number(v) * 100).toFixed(1)}%`;
    }
    return v;
  }

  // ----------------------------
  // Render main component
  // ----------------------------
  return (
    <div style={pageStyle}>
      <header style={headerStyle}>
        <img
          src="https://img.icons8.com/fluency/96/000000/bot.png"
          alt="bot"
          style={{ width: 48, height: 48, marginRight: 16 }}
        />
        <h2 style={{ margin: 0, fontWeight: 700 }}>
          {viewMode === "config" || viewMode.includes("config")
            ? "Configuration Matrix (Per SKU)"
            : viewMode === "aggregated"
            ? "Aggregated Attributes"
            : viewMode === "config_rules"
            ? "Configuration Rules"
            : "Pricing Rules"}
        </h2>
        <span style={headerUserStyle}>Hi {username} 👋</span>
      </header>

      {/* RefineWizard visible only on aggregated */}
      {viewMode === "aggregated" && (
        <div style={{ position: "sticky", top: "85px", zIndex: 15 }}>
          <RefineWizard onRefresh={refreshGraph} />
        </div>
      )}

      <main style={mainStyle}>
        <div style={cardStyle}>
          {/* Controls */}
          <div style={controlsRowStyle}>
            <div style={{ display: "flex", gap: 10, flex: 1 }}>
              <input
                placeholder="Search by Attribute..."
                value={attributeSearch}
                onChange={(e) => setAttributeSearch(e.target.value)}
                style={searchInputStyle}
              />
              <input
                placeholder="Search by Value..."
                value={valueSearch}
                onChange={(e) => setValueSearch(e.target.value)}
                style={searchInputStyle}
              />
            </div>

            <div style={{ position: "relative" }}>
              <button
                style={buttonStyle}
                onClick={() => setExportMenuOpen((s) => !s)}
              >
                Export ▼
              </button>
              {exportMenuOpen && (
                <div style={exportMenuStyle}>
                  {["xlsx", "pdf"].map((fmt) => (
                    <div
                      key={fmt}
                      onClick={() => handleExport(fmt)}
                      style={exportMenuItemStyle}
                    >
                      Export as {fmt.toUpperCase()}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: 10, marginLeft: 8 }}>
              <button
                style={{
                  ...buttonStyle,
                  background: "linear-gradient(135deg, #16a34a, #22c55e)",
                }}
                onClick={handleConfigAggregatedToggle}
              >
                {configAggregatedLabel}
              </button>

              <button
                style={{
                  ...buttonStyle,
                  background: "linear-gradient(135deg, #1629f9ff, #fb923c)",
                }}
                onClick={handleRulesToggleClick}
              >
                {rulesToggleLabel()}
              </button>

              {/* Variant button when aggregated */}
              {viewMode === "aggregated" && (
                <button
                  style={{
                    ...buttonStyle,
                    background: "linear-gradient(135deg, #3b82f6, #2563eb)",
                  }}
                  onClick={() => navigate("/variant")}
                >
                  Variant Attribute Analysis
                </button>
              )}
            </div>
          </div>

          {/* Table / view */}
          <div style={scrollContainerStyle}>
            {/* Configuration Matrix (Per SKU) */}
            {viewMode === "config" && (
              <table style={tableStyle}>
                <thead style={theadStyle}>
                  <tr>
                    <th style={stickyCheckCol}>
                      <input
                        type="checkbox"
                        onChange={(e) => toggleSelectAll(e.target.checked)}
                        checked={
                          selectedRows.length === filteredRows.length &&
                          filteredRows.length > 0
                        }
                      />
                    </th>
                    {["SKU Description", ...allAttributes].map((attr, idx) => (
                      <th
                        key={idx}
                        style={{
                          ...thStyle,
                          ...(attr === "SKU Description" ? stickySKUCol : {}),
                        }}
                      >
                        {attr}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row, idx) => (
                    <tr key={idx}>
                      <td style={stickyCheckCol}>
                        <input
                          type="checkbox"
                          checked={selectedRows.includes(row.SKU)}
                          onChange={(e) => {
                            const updated = e.target.checked
                              ? [...selectedRows, row.SKU]
                              : selectedRows.filter((r) => r !== row.SKU);
                            setSelectedRows(updated);
                          }}
                        />
                      </td>
                      {["SKU", ...allAttributes].map((a, j) => (
                        <td
                          key={j}
                          style={{
                            ...tdStyle,
                            ...(a === "SKU" ? stickySKUCol : {}),
                          }}
                        >
                          {row[a]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {/* Aggregated Attributes */}
            {viewMode === "aggregated" && (
              <table style={tableStyle}>
                <thead style={theadStyle}>
                  <tr>
                    <th style={{ ...thStyle, width: "50px", textAlign: "center" }}>✔</th>
                    {(localAggregated.columns || []).map((col, idx) => (
                      <th key={idx} style={thStyle}>
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(localAggregated.rows || []).map((row, i) => (
                    <tr key={i}>
                      <td style={{ textAlign: "center", width: "50px" }}>
                        <input
                          type="checkbox"
                          checked={selectedAttributes.includes(row[0])}
                          onChange={() => {
                            setSelectedAttributes((prev) =>
                              prev.includes(row[0]) ? prev.filter((a) => a !== row[0]) : [...prev, row[0]]
                            );
                          }}
                        />
                      </td>
                      {row.map((cell, j) => (
                        <td key={j} style={tdStyle}>
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {/* Configuration Rules */}
            {viewMode === "config_rules" && (
              <div style={{ padding: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <div>
                    <strong style={{ fontSize: 16 }}>Configuration Rules</strong>
                    <div style={{ fontSize: 12, color: "#475569" }}>
                      Generated at: {configRules?.generated_at || "N/A"}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button style={secondaryButtonStyle} onClick={recomputeRulesFromFrontend}>
                      Recompute Rules
                    </button>
                    <button style={secondaryButtonStyle} onClick={() => fetchConfigRules()}>
                      Refresh
                    </button>
                  </div>
                </div>

                {renderConfigRulesTable()}
              </div>
            )}

            {/* Pricing Rules */}
            {viewMode === "pricing_rules" && (
              <div style={{ padding: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <div>
                    <strong style={{ fontSize: 16 }}>Pricing Rules</strong>
                    <div style={{ fontSize: 12, color: "#475569" }}>{pricingRules?.message || ""}</div>
                  </div>
                  <div>
                    <button style={secondaryButtonStyle} onClick={() => fetchPricingRules()}>
                      Refresh
                    </button>
                  </div>
                </div>

                {renderPricingRulesTable()}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

/* -------------------------
   Inline Styles
   ------------------------- */
const pageStyle = {
  background: "linear-gradient(135deg, #f8fafc, #e0e7ff)",
  minHeight: "100vh",
  fontFamily: "'Inter', sans-serif",
  color: "#1e293b",
};

const headerStyle = {
  display: "flex",
  alignItems: "center",
  padding: "24px",
  background: "rgba(255,255,255,0.6)",
  backdropFilter: "blur(10px)",
  boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
  zIndex: 20,
  position: "sticky",
  top: 0,
};

const headerUserStyle = {
  marginLeft: "auto",
  background: "rgba(255,255,255,0.5)",
  padding: "8px 16px",
  borderRadius: 12,
  fontWeight: 600,
};

const mainStyle = {
  padding: 30,
  display: "flex",
  justifyContent: "center",
};

const cardStyle = {
  width: "95%",
  background: "rgba(255,255,255,0.9)",
  borderRadius: 20,
  boxShadow: "0 10px 40px rgba(0,0,0,0.1)",
  padding: 24,
};

const controlsRowStyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 10,
  marginBottom: 20,
};

const searchInputStyle = {
  flex: 1,
  padding: "12px 14px",
  borderRadius: "10px",
  border: "1px solid #cbd5e1",
  outline: "none",
  background: "rgba(255,255,255,0.8)",
  fontSize: "15px",
};

const buttonStyle = {
  padding: "10px 20px",
  borderRadius: 8,
  border: "none",
  background: "linear-gradient(135deg,#6366f1,#3b82f6)",
  color: "white",
  fontWeight: 600,
  cursor: "pointer",
};

const secondaryButtonStyle = {
  padding: "8px 12px",
  borderRadius: 8,
  border: "1px solid #e2e8f0",
  background: "white",
  color: "#1f2937",
  fontWeight: 600,
  cursor: "pointer",
};

const tableStyle = { width: "100%", borderCollapse: "collapse", fontSize: "14px" };
const theadStyle = { background: "linear-gradient(135deg,#eef2ff,#dbeafe)" };
const thStyle = {
  padding: "12px",
  borderBottom: "2px solid #cbd5e1",
  fontWeight: 700,
  color: "#1e293b",
  textAlign: "left",
};
const tdStyle = {
  padding: "10px",
  borderBottom: "1px solid #e2e8f0",
  color: "#334155",
};

const stickyCheckCol = {
  position: "sticky",
  left: 0,
  background: "rgba(240,242,255,0.98)",
  zIndex: 8,
  textAlign: "center",
  width: "60px",
};
const stickySKUCol = {
  position: "sticky",
  left: "22.4px",
  background: "rgba(240,242,255,0.98)",
  zIndex: 7,
  fontWeight: 600,
  minWidth: "220px",
};
const scrollContainerStyle = {
  maxHeight: "85vh",
  overflow: "auto",
  borderRadius: 12,
};
const exportMenuStyle = {
  position: "absolute",
  right: 0,
  top: "110%",
  background: "white",
  boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
  borderRadius: 8,
};
const exportMenuItemStyle = {
  padding: "10px 16px",
  cursor: "pointer",
  borderBottom: "1px solid #eee",
};

const stickyCheckHeader = {
  position: "sticky",
  left: 0,
  zIndex: 10,
  background: "white",
};

const smallButtonStyle = {
  padding: "6px 8px",
  borderRadius: 6,
  border: "1px solid #e2e8f0",
  background: "white",
  cursor: "pointer",
  fontWeight: 600,
};
