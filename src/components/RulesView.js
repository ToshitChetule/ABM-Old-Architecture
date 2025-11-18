// frontend/src/components/RulesView.js
import React, { useEffect, useState } from "react";

export default function RulesView() {
  const [rulesData, setRulesData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedRule, setSelectedRule] = useState(null);

  useEffect(() => {
    fetchRules();
  }, []);

  const fetchRules = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:5000/rules/configuration");
      const data = await res.json();
      setRulesData(data);
    } catch (err) {
      console.error("Error loading rules", err);
    } finally {
      setLoading(false);
    }
  };

  const previewExamples = (rule) => {
    setSelectedRule(rule);
  };

  if (loading) return <div>Loading rules...</div>;
  if (!rulesData) return <div>No rules generated yet.</div>;

  return (
    <div style={{ padding: 20, fontFamily: "Inter, sans-serif" }}>
      <h2>Configuration Rules</h2>
      <p>
        Generated at: {rulesData.generated_at || "n/a"} — Total SKUs:{" "}
        {rulesData.total_rows}
      </p>

      <div style={{ marginTop: 12 }}>
        {rulesData.rules.length === 0 && <div>No strong rules found.</div>}

        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ textAlign: "left" }}>
              <th style={{ padding: 8 }}>IF</th>
              <th style={{ padding: 8 }}>THEN</th>
              <th style={{ padding: 8 }}>Support</th>
              <th style={{ padding: 8 }}>Confidence</th>
              <th style={{ padding: 8 }}>Examples</th>
            </tr>
          </thead>
          <tbody>
            {rulesData.rules.map((r) => {
              const ifText = Object.entries(r.if).map(([k, v]) => `${k} = ${v}`).join(" AND ");
              const thenText = Object.entries(r.then).map(([k, v]) => `${k} = ${v}`).join(" , ");
              return (
                <tr key={r.id} style={{ borderTop: "1px solid #eee" }}>
                  <td style={{ padding: 8 }}>{ifText}</td>
                  <td style={{ padding: 8 }}>{thenText}</td>
                  <td style={{ padding: 8 }}>{r.support} ({Math.round(r.support_pct*100)}%)</td>
                  <td style={{ padding: 8 }}>{Math.round(r.confidence*100)}%</td>
                  <td style={{ padding: 8 }}>
                    <button onClick={() => previewExamples(r)}>Preview</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {selectedRule && (
          <div style={{ marginTop: 16, padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
            <h4>Examples for rule</h4>
            <div style={{ marginBottom: 8 }}>
              <strong>IF:</strong>{" "}
              {Object.entries(selectedRule.if).map(([k, v]) => `${k} = ${v}`).join(" AND ")}
            </div>
            <div style={{ marginBottom: 8 }}>
              <strong>THEN:</strong>{" "}
              {Object.entries(selectedRule.then).map(([k, v]) => `${k} = ${v}`).join(" , ")}
            </div>
            <div>
              <strong>Example SKUs:</strong>
              <ul>
                {(selectedRule.examples || []).map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>

            <div style={{ marginTop: 8 }}>
              <button onClick={() => setSelectedRule(null)}>Close</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
