// import React, { useMemo, useState, useEffect } from "react";
// import { useNavigate, useLocation } from "react-router-dom";
// import * as XLSX from "xlsx";
// import { jsPDF } from "jspdf";
// import "jspdf-autotable";
// import Swal from "sweetalert2";
// import RefineWizard from "./components/RefineWizard";
// import { Button } from "@mui/material";

// export default function ExtractionReviewPage() {
//   const navigate = useNavigate();
//   const location = useLocation();
//   const { uploadedFilename, sku_matrix = [], aggregated_matrix = {} } =
//     location.state || {};

//   const [viewMode, setViewMode] = useState("config");
//   const [attributeSearch, setAttributeSearch] = useState("");
//   const [valueSearch, setValueSearch] = useState("");
//   const [prompt, setPrompt] = useState("");
//   const [selectedRows, setSelectedRows] = useState([]);
//   const [exportMenuOpen, setExportMenuOpen] = useState(false);
//   const [loading, setLoading] = useState(false);
//   const [localAggregated, setLocalAggregated] = useState(aggregated_matrix);
//   const [selectedAttributes, setSelectedAttributes] = useState([]); // ✅ multiple selection support

//   const storedUser = JSON.parse(localStorage.getItem("user"));
//   const username = storedUser?.username || "User";

//   // ✅ Extract unique attributes
//   const allAttributes = useMemo(() => {
//     const attrs = new Set();
//     sku_matrix.forEach((s) => s.attributes.forEach(([a]) => attrs.add(a)));
//     return Array.from(attrs);
//   }, [sku_matrix]);

//   useEffect(() => {
//     if (viewMode === "aggregated" || !localAggregated?.rows?.length) {
//       console.log("🔄 Fetching latest aggregated data from Neo4j...");
//       fetch("http://localhost:5000/graph/aggregated")
//         .then((res) => res.json())
//         .then((data) => {
//           if (data && data.columns && data.rows) {
//             setLocalAggregated(data);
//           }
//         })
//         .catch((err) => console.error("❌ Error loading from graph:", err));
//     }
//   }, [viewMode]);

//   // ✅ Build configuration matrix
//   const matrixRows = useMemo(() => {
//     return sku_matrix.map((skuObj) => {
//       const row = { SKU: skuObj.sku };
//       allAttributes.forEach((a) => {
//         const found = skuObj.attributes.find(([attr]) => attr === a);
//         row[a] = found ? found[1] : "";
//       });
//       return row;
//     });
//   }, [sku_matrix, allAttributes]);

//   // ✅ Filter logic
//   const filteredRows = useMemo(() => {
//     if (viewMode === "config") {
//       return matrixRows.filter((row) => {
//         const attrMatch = Object.keys(row)
//           .join(" ")
//           .toLowerCase()
//           .includes(attributeSearch.toLowerCase());
//         const valMatch = Object.values(row)
//           .join(" ")
//           .toLowerCase()
//           .includes(valueSearch.toLowerCase());
//         return attrMatch && valMatch;
//       });
//     } else {
//       const columns = localAggregated.columns || [];
//       const rows = localAggregated.rows || [];

//       return rows.filter((row) => {
//         const headerText = columns.join(" ").toLowerCase();
//         const rowText = row.join(" ").toLowerCase();
//         const attrMatch =
//           !attributeSearch ||
//           headerText.includes(attributeSearch.toLowerCase());
//         const valMatch =
//           !valueSearch || rowText.includes(valueSearch.toLowerCase());
//         return attrMatch && valMatch;
//       });
//     }
//   }, [matrixRows, localAggregated, attributeSearch, valueSearch, viewMode]);

//   // ✅ Select all toggle
//   const toggleSelectAll = (checked) => {
//     setSelectedRows(checked ? filteredRows.map((r) => r.SKU) : []);
//   };

//   // ✅ Export logic
//   const handleExport = (format) => {
//     try {
//       let data, filename, columns, rows;

//       if (viewMode === "config") {
//         data = matrixRows;
//         filename = uploadedFilename
//           ? `${uploadedFilename.replace(/\.[^/.]+$/, "")}_Configuration_Matrix`
//           : "Configuration_Matrix";
//       } else {
//         columns = localAggregated.columns || [];
//         rows = localAggregated.rows || [];

//         if (!columns.length || !rows.length) {
//           Swal.fire("No data available to export.", "", "info");
//           return;
//         }

//         data = rows.map((r) =>
//           Object.fromEntries(columns.map((c, i) => [c, r[i]]))
//         );
//         filename = uploadedFilename
//           ? `${uploadedFilename.replace(/\.[^/.]+$/, "")}_Aggregated_Attributes`
//           : "Aggregated_Attributes";
//       }

//       if (format === "xlsx") {
//         const ws = XLSX.utils.json_to_sheet(data);
//         const wb = XLSX.utils.book_new();
//         XLSX.utils.book_append_sheet(
//           wb,
//           ws,
//           viewMode === "config" ? "Configuration" : "Aggregated"
//         );
//         XLSX.writeFile(wb, `${filename}.xlsx`);
//         Swal.fire({
//           icon: "success",
//           title: "Excel Exported Successfully!",
//           timer: 1500,
//           showConfirmButton: false,
//         });
//       } else if (format === "pdf") {
//         const doc = new jsPDF({
//           orientation: "landscape",
//           unit: "pt",
//           format: "a4",
//         });

//         doc.setFontSize(14);
//         doc.text(filename, 40, 40);

//         if (viewMode === "config") {
//           const headers = Object.keys(matrixRows[0] || {});
//           const body = filteredRows.map((r) => Object.values(r));
//           doc.autoTable({
//             head: [headers],
//             body,
//             startY: 60,
//             styles: { fontSize: 8, overflow: "linebreak" },
//             headStyles: { fillColor: [99, 102, 241] },
//           });
//         } else {
//           doc.autoTable({
//             head: [columns],
//             body: rows,
//             startY: 60,
//             styles: { fontSize: 8, overflow: "linebreak" },
//             headStyles: { fillColor: [59, 130, 246] },
//           });
//         }

//         doc.save(`${filename}.pdf`);
//         Swal.fire({
//           icon: "success",
//           title: "PDF Exported Successfully!",
//           timer: 1500,
//           showConfirmButton: false,
//         });
//       }

//       setExportMenuOpen(false);
//     } catch (err) {
//       console.error("Export Error:", err);
//       Swal.fire("Error", err.message, "error");
//     }
//   };

//   // ✅ Graph-based Refinement (multi-attribute)
//   const handleRefineAttributes = async () => {
//     if (!prompt.trim()) {
//       Swal.fire("Please enter a refinement prompt.", "", "warning");
//       return;
//     }
//     if (viewMode !== "aggregated") {
//       Swal.fire(
//         "Switch to Aggregated View",
//         "Refinement is available only in Aggregated Attributes view.",
//         "info"
//       );
//       return;
//     }
//     if (selectedAttributes.length === 0) {
//       Swal.fire("Please select at least one Attribute to refine.", "", "info");
//       return;
//     }

//     setLoading(true);
//     try {
//       const response = await fetch("http://localhost:5000/refine_graph", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           attributes: selectedAttributes,
//           prompt: prompt,
//         }),
//       });

//       const result = await response.json();

//       if (!response.ok) {
//         Swal.fire("Refinement Failed", result.error || "Unknown error", "error");
//         return;
//       }

//       // Refresh UI after refinement
//       fetch("http://localhost:5000/graph/aggregated")
//         .then((res) => res.json())
//         .then((data) => setLocalAggregated(data))
//         .catch((err) => console.error("Error refreshing data:", err));

//       Swal.fire({
//         icon: "success",
//         title: "Refined Successfully!",
//         html: `<div style="text-align:left">
//           <p><b>Processed Attributes:</b> ${selectedAttributes.join(", ")}</p>
//           <p><b>Actions:</b><br>${
//             (result.actions || []).join("<br>") || "No changes detected"
//           }</p>
//         </div>`,
//         confirmButtonText: "OK",
//       });

//       setSelectedAttributes([]);
//       setPrompt("");
//     } catch (err) {
//       Swal.fire("Error", err.message, "error");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div
//       style={{
//         background: "linear-gradient(135deg, #f8fafc, #e0e7ff)",
//         minHeight: "100vh",
//         fontFamily: "'Inter', sans-serif",
//         color: "#1e293b",
//       }}
//     >
//       <header
//         style={{
//           display: "flex",
//           alignItems: "center",
//           padding: "24px",
//           background: "rgba(255,255,255,0.6)",
//           backdropFilter: "blur(10px)",
//           boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
//         }}
//       >
//         <img
//           src="https://img.icons8.com/fluency/96/000000/bot.png"
//           alt="bot"
//           style={{ width: 48, height: 48, marginRight: 16 }}
//         />
//         <h2 style={{ margin: 0, fontWeight: 700 }}>
//           {viewMode === "config"
//             ? "Configuration Matrix (Per SKU)"
//             : "Aggregated Attributes"}
//         </h2>
//         <span
//           style={{
//             marginLeft: "auto",
//             background: "rgba(255,255,255,0.5)",
//             padding: "8px 16px",
//             borderRadius: 12,
//             fontWeight: 600,
//           }}
//         >
//           Hi {username} 👋
//         </span>
//       </header>

//       <main
//         style={{
//           padding: 30,
//           display: "flex",
//           justifyContent: "center",
//         }}
//       >
//         <div
//           style={{
//             width: "95%",
//             background: "rgba(255,255,255,0.9)",
//             borderRadius: 20,
//             boxShadow: "0 10px 40px rgba(0,0,0,0.1)",
//             padding: 24,
//             backdropFilter: "blur(15px)",
//           }}
//         >
//           {/* Controls */}
//           <div
//             style={{
//               display: "flex",
//               justifyContent: "space-between",
//               alignItems: "center",
//               gap: 10,
//               marginBottom: 20,
//             }}
//           >
//             <div style={{ display: "flex", gap: 10, flex: 1 }}>
//               <input
//                 placeholder="Search by Attribute..."
//                 value={attributeSearch}
//                 onChange={(e) => setAttributeSearch(e.target.value)}
//                 style={searchInputStyle}
//               />
//               <input
//                 placeholder="Search by Value..."
//                 value={valueSearch}
//                 onChange={(e) => setValueSearch(e.target.value)}
//                 style={searchInputStyle}
//               />
//             </div>

//             <div style={{ position: "relative" }}>
//               <button
//                 style={buttonStyle}
//                 onClick={() => setExportMenuOpen(!exportMenuOpen)}
//               >
//                 Export ▼
//               </button>
//               {exportMenuOpen && (
//                 <div style={exportMenuStyle}>
//                   {["xlsx", "pdf"].map((fmt) => (
//                     <div
//                       key={fmt}
//                       onClick={() => handleExport(fmt)}
//                       style={exportMenuItemStyle}
//                     >
//                       Export as {fmt.toUpperCase()}
//                     </div>
//                   ))}
//                 </div>
//               )}
//             </div>

//             <div style={{ display: "flex", gap: 10 }}>
//               <button
//                 style={{
//                   ...buttonStyle,
//                   background: "linear-gradient(135deg, #16a34a, #22c55e)",
//                 }}
//                 onClick={() =>
//                   setViewMode(viewMode === "config" ? "aggregated" : "config")
//                 }
//               >
//                 {viewMode === "config"
//                   ? "View Aggregated Attributes"
//                   : "← Back to Configuration Matrix"}
//               </button>

//               {viewMode === "aggregated" && (
//                 <button
//                   style={{
//                     ...buttonStyle,
//                     background: "linear-gradient(135deg, #3b82f6, #2563eb)",
//                   }}
//                   onClick={() => navigate("/variant")}
//                 >
//                   Variant Attribute Analysis
//                 </button>
//               )}
//             </div>
//           </div>

//           {/* Prompt */}
//           <div style={{ display: "flex", gap: 10, marginBottom: 24 }}>
//             <textarea
//               placeholder="Write a prompt to refine multiple attributes (e.g., rename attribute Edition to Level and rename value Orchestrator to Org)"
//               value={prompt}
//               onChange={(e) => setPrompt(e.target.value)}
//               style={textareaStyle}
//             />
//             <button onClick={handleRefineAttributes} style={buttonStyle}>
//               {loading ? "Refining..." : "Refine Attributes"}
//             </button>
//           </div>

//           {/* Table */}
//           <div style={scrollContainerStyle}>
//             <table style={tableStyle}>
//               <thead style={theadStyle}>
//                 {viewMode === "config" ? (
//                   <tr>
//                     <th style={stickyCheckCol}>
//                       <input
//                         type="checkbox"
//                         onChange={(e) => toggleSelectAll(e.target.checked)}
//                         checked={
//                           selectedRows.length === filteredRows.length &&
//                           filteredRows.length > 0
//                         }
//                       />
//                     </th>
//                     {["SKU Description", ...allAttributes].map((attr, idx) => (
//                       <th
//                         key={idx}
//                         style={{
//                           ...thStyle,
//                           ...(attr === "SKU Description" ? stickySKUCol : {}),
//                         }}
//                       >
//                         {attr}
//                       </th>
//                     ))}
//                   </tr>
//                 ) : (
//                   <tr>
//                     <th style={{ ...thStyle, width: "50px", textAlign: "center" }}>
//                       ✔
//                     </th>
//                     {(localAggregated.columns || []).map((col, idx) => (
//                       <th key={idx} style={thStyle}>
//                         {col}
//                       </th>
//                     ))}
//                   </tr>
//                 )}
//               </thead>

//               <tbody>
//                 {viewMode === "config"
//                   ? filteredRows.map((row, idx) => (
//                       <tr
//                         key={idx}
//                         style={{
//                           background:
//                             idx % 2 === 0
//                               ? "rgba(255,255,255,0.9)"
//                               : "rgba(249,250,251,0.9)",
//                         }}
//                       >
//                         <td style={stickyCheckCol}>
//                           <input
//                             type="checkbox"
//                             checked={selectedRows.includes(row.SKU)}
//                             onChange={(e) => {
//                               const updated = e.target.checked
//                                 ? [...selectedRows, row.SKU]
//                                 : selectedRows.filter((r) => r !== row.SKU);
//                               setSelectedRows(updated);
//                             }}
//                           />
//                         </td>
//                         {["SKU", ...allAttributes].map((a, j) => (
//                           <td
//                             key={j}
//                             style={{
//                               ...tdStyle,
//                               ...(a === "SKU" ? stickySKUCol : {}),
//                             }}
//                           >
//                             {row[a]}
//                           </td>
//                         ))}
//                       </tr>
//                     ))
//                   : (localAggregated.rows || []).map((row, i) => (
//                       <tr
//                         key={i}
//                         style={{
//                           background: selectedAttributes.includes(row[0])
//                             ? "rgba(191,219,254,0.6)"
//                             : i % 2 === 0
//                             ? "rgba(255,255,255,0.9)"
//                             : "rgba(249,250,251,0.9)",
//                         }}
//                       >
//                         <td style={{ textAlign: "center", width: "50px" }}>
//                           <input
//                             type="checkbox"
//                             checked={selectedAttributes.includes(row[0])}
//                             onChange={() => {
//                               setSelectedAttributes((prev) =>
//                                 prev.includes(row[0])
//                                   ? prev.filter((a) => a !== row[0])
//                                   : [...prev, row[0]]
//                               );
//                             }}
//                           />
//                         </td>
//                         {row.map((cell, j) => (
//                           <td key={j} style={tdStyle}>
//                             {cell}
//                           </td>
//                         ))}
//                       </tr>
//                     ))}
//               </tbody>
//             </table>
//           </div>
//         </div>
//       </main>
//     </div>
//   );
// }

// /* ✨ Styles — unchanged */
// const searchInputStyle = {
//   flex: 1,
//   padding: "12px 14px",
//   borderRadius: "10px",
//   border: "1px solid #cbd5e1",
//   outline: "none",
//   background: "rgba(255,255,255,0.8)",
//   fontSize: "15px",
//   transition: "0.2s",
// };

// const buttonStyle = {
//   padding: "10px 20px",
//   borderRadius: 8,
//   border: "none",
//   background: "linear-gradient(135deg,#6366f1,#3b82f6)",
//   color: "white",
//   fontWeight: 600,
//   cursor: "pointer",
//   boxShadow: "0 4px 10px rgba(59,130,246,0.3)",
//   transition: "all 0.25s ease",
// };

// const tableStyle = { width: "100%", borderCollapse: "collapse", fontSize: "14px" };
// const theadStyle = {
//   background: "linear-gradient(135deg,#eef2ff,#dbeafe)",
//   position: "sticky",
//   top: 0,
//   zIndex: 10,
// };
// const thStyle = {
//   padding: "12px",
//   borderBottom: "2px solid #cbd5e1",
//   borderRight: "1px solid #e5e7eb",
//   fontWeight: 700,
//   color: "#1e293b",
//   textAlign: "left",
//   whiteSpace: "nowrap",
// };
// const tdStyle = {
//   padding: "10px",
//   borderBottom: "1px solid #e2e8f0",
//   borderRight: "1px solid #e5e7eb",
//   color: "#334155",
//   background: "rgba(255,255,255,0.95)",
//   whiteSpace: "nowrap",
// };
// const stickyCheckCol = {
//   position: "sticky",
//   left: 0,
//   background: "rgba(240,242,255,0.98)",
//   boxShadow: "2px 0 6px rgba(0,0,0,0.05)",
//   zIndex: 8,
//   textAlign: "center",
//   width: "60px",
//   borderRight: "1px solid #e5e7eb",
// };
// const stickySKUCol = {
//   position: "sticky",
//   left: "22.4px",
//   background: "rgba(240,242,255,0.98)",
//   boxShadow: "2px 0 6px rgba(0,0,0,0.05)",
//   zIndex: 7,
//   fontWeight: 600,
//   minWidth: "220px",
//   borderRight: "1px solid #e5e7eb",
// };
// const scrollContainerStyle = {
//   maxHeight: "85vh",
//   overflow: "auto",
//   borderRadius: 12,
//   boxShadow: "inset 0 0 10px rgba(0,0,0,0.05)",
// };
// const textareaStyle = {
//   flex: 1,
//   minHeight: 80,
//   padding: "12px 14px",
//   borderRadius: 10,
//   border: "1px solid #cbd5e1",
//   background: "rgba(255,255,255,0.9)",
//   resize: "vertical",
// };
// const exportMenuStyle = {
//   position: "absolute",
//   right: 0,
//   top: "110%",
//   background: "white",
//   boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
//   borderRadius: 8,
//   overflow: "hidden",
//   zIndex: 100,
// };
// const exportMenuItemStyle = {
//   padding: "10px 16px",
//   cursor: "pointer",
//   borderBottom: "1px solid #eee",
//   transition: "background 0.2s",
// };




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
  const [attributeSearch, setAttributeSearch] = useState("");
  const [valueSearch, setValueSearch] = useState("");
  const [prompt, setPrompt] = useState("");
  const [selectedRows, setSelectedRows] = useState([]);
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [localAggregated, setLocalAggregated] = useState(aggregated_matrix);
  const [selectedAttributes, setSelectedAttributes] = useState([]);
  const [openWizard, setOpenWizard] = useState(false); // retained
  const storedUser = JSON.parse(localStorage.getItem("user"));
  const username = storedUser?.username || "User";

  const allAttributes = useMemo(() => {
    const attrs = new Set();
    sku_matrix.forEach((s) => s.attributes.forEach(([a]) => attrs.add(a)));
    return Array.from(attrs);
  }, [sku_matrix]);

  useEffect(() => {
    if (viewMode === "aggregated" || !localAggregated?.rows?.length) {
      fetch("http://localhost:5000/graph/aggregated")
        .then((res) => res.json())
        .then((data) => {
          if (data && data.columns && data.rows) setLocalAggregated(data);
        })
        .catch((err) => console.error("❌ Error loading from graph:", err));
    }
  }, [viewMode]);

  const refreshGraph = () => {
    fetch("http://localhost:5000/graph/aggregated")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.columns && data.rows) setLocalAggregated(data);
      })
      .catch((err) => console.error("Error refreshing data:", err));
  };

  const matrixRows = useMemo(() => {
    return sku_matrix.map((skuObj) => {
      const row = { SKU: skuObj.sku };
      allAttributes.forEach((a) => {
        const found = skuObj.attributes.find(([attr]) => attr === a);
        row[a] = found ? found[1] : "";
      });
      return row;
    });
  }, [sku_matrix, allAttributes]);

  const filteredRows = useMemo(() => {
    if (viewMode === "config") {
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
      const columns = localAggregated.columns || [];
      const rows = localAggregated.rows || [];
      return rows.filter((row) => {
        const headerText = columns.join(" ").toLowerCase();
        const rowText = row.join(" ").toLowerCase();
        const attrMatch =
          !attributeSearch ||
          headerText.includes(attributeSearch.toLowerCase());
        const valMatch =
          !valueSearch || rowText.includes(valueSearch.toLowerCase());
        return attrMatch && valMatch;
      });
    }
  }, [matrixRows, localAggregated, attributeSearch, valueSearch, viewMode]);

  const toggleSelectAll = (checked) => {
    setSelectedRows(checked ? filteredRows.map((r) => r.SKU) : []);
  };

  const handleExport = (format) => {
    try {
      let data, filename, columns, rows;

      if (viewMode === "config") {
        data = matrixRows;
        filename = uploadedFilename
          ? `${uploadedFilename.replace(/\.[^/.]+$/, "")}_Configuration_Matrix`
          : "Configuration_Matrix";
      } else {
        columns = localAggregated.columns || [];
        rows = localAggregated.rows || [];

        if (!columns.length || !rows.length) {
          Swal.fire("No data available to export.", "", "info");
          return;
        }

        data = rows.map((r) =>
          Object.fromEntries(columns.map((c, i) => [c, r[i]]))
        );
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
          viewMode === "config" ? "Configuration" : "Aggregated"
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
        if (viewMode === "config") {
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
            head: [columns],
            body: rows,
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

  return (
    <div
      style={{
        background: "linear-gradient(135deg, #f8fafc, #e0e7ff)",
        minHeight: "100vh",
        fontFamily: "'Inter', sans-serif",
        color: "#1e293b",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          padding: "24px",
          background: "rgba(255,255,255,0.6)",
          backdropFilter: "blur(10px)",
          boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
          zIndex: 20,
          position: "sticky",
          top: 0,
        }}
      >
        <img
          src="https://img.icons8.com/fluency/96/000000/bot.png"
          alt="bot"
          style={{ width: 48, height: 48, marginRight: 16 }}
        />
        <h2 style={{ margin: 0, fontWeight: 700 }}>
          {viewMode === "config"
            ? "Configuration Matrix (Per SKU)"
            : "Aggregated Attributes"}
        </h2>
        <span
          style={{
            marginLeft: "auto",
            background: "rgba(255,255,255,0.5)",
            padding: "8px 16px",
            borderRadius: 12,
            fontWeight: 600,
          }}
        >
          Hi {username} 👋
        </span>
      </header>

      {/* 🧠 Fixed Refine Wizard (only for Aggregated View) */}
      {viewMode === "aggregated" && (
        <div style={{ position: "sticky", top: "85px", zIndex: 15 }}>
          <RefineWizard onRefresh={refreshGraph} />
        </div>
      )}

      <main
        style={{
          padding: 30,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: "95%",
            background: "rgba(255,255,255,0.9)",
            borderRadius: 20,
            boxShadow: "0 10px 40px rgba(0,0,0,0.1)",
            padding: 24,
          }}
        >
          {/* Controls */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 10,
              marginBottom: 20,
            }}
          >
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
                onClick={() => setExportMenuOpen(!exportMenuOpen)}
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

            <div style={{ display: "flex", gap: 10 }}>
              <button
                style={{
                  ...buttonStyle,
                  background: "linear-gradient(135deg, #16a34a, #22c55e)",
                }}
                onClick={() =>
                  setViewMode(viewMode === "config" ? "aggregated" : "config")
                }
              >
                {viewMode === "config"
                  ? "View Aggregated Attributes"
                  : "← Back to Configuration Matrix"}
              </button>

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

          {/* Table Section (unchanged) */}
          <div style={scrollContainerStyle}>
            <table style={tableStyle}>
              <thead style={theadStyle}>
                {viewMode === "config" ? (
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
                ) : (
                  <tr>
                    <th style={{ ...thStyle, width: "50px", textAlign: "center" }}>
                      ✔
                    </th>
                    {(localAggregated.columns || []).map((col, idx) => (
                      <th key={idx} style={thStyle}>
                        {col}
                      </th>
                    ))}
                  </tr>
                )}
              </thead>

              <tbody>
                {viewMode === "config"
                  ? filteredRows.map((row, idx) => (
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
                    ))
                  : (localAggregated.rows || []).map((row, i) => (
                      <tr key={i}>
                        <td style={{ textAlign: "center", width: "50px" }}>
                          <input
                            type="checkbox"
                            checked={selectedAttributes.includes(row[0])}
                            onChange={() => {
                              setSelectedAttributes((prev) =>
                                prev.includes(row[0])
                                  ? prev.filter((a) => a !== row[0])
                                  : [...prev, row[0]]
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
          </div>
        </div>
      </main>
    </div>
  );
}

/* 🎨 Styles unchanged */
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
