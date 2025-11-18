import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Select from "react-select";

import excelIcon from "../icons/excel.png";
import pdfIcon from "../icons/pdf.png";
import wordIcon from "../icons/doc.png";
import textIcon from "../icons/txt-file.png";
// import defaultIcon from "../icons/default.png";

export default function DocReviewPage() {
  const location = useLocation();
  const navigate = useNavigate();

  const { uploadedFilename, fileSize, fileType, uploadedDate, fileObject } =
    location.state || {};

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [numRows, setNumRows] = useState(null);
  const [industry, setIndustry] = useState(null);
  const [productType, setProductType] = useState(null);

  const industryOptions = [
    { value: "automotive", label: "Automotive" },
    { value: "aerospace", label: "Aerospace" },
    { value: "banking", label: "Banking" },
    { value: "chemicals", label: "Chemicals" },
    { value: "construction", label: "Construction" },
    { value: "consumer_goods", label: "Consumer Goods" },
    { value: "education", label: "Education" },
    { value: "electronics", label: "Electronics" },
    { value: "energy", label: "Energy & Utilities" },
    { value: "entertainment", label: "Entertainment" },
    { value: "food_beverages", label: "Food & Beverages" },
    { value: "healthcare", label: "Healthcare" },
    { value: "hospitality", label: "Hospitality" },
    { value: "insurance", label: "Insurance" },
    { value: "it_services", label: "IT & Software Services" },
    { value: "legal", label: "Legal Services" },
    { value: "logistics", label: "Logistics" },
    { value: "manufacturing", label: "Manufacturing" },
    { value: "media", label: "Media" },
    { value: "mining", label: "Mining" },
    { value: "pharmaceuticals", label: "Pharmaceuticals" },
    { value: "public_sector", label: "Public Sector" },
    { value: "real_estate", label: "Real Estate" },
    { value: "retail", label: "Retail" },
    { value: "telecom", label: "Telecommunications" },
    { value: "textile", label: "Textile" },
    { value: "transport", label: "Transport" },
    { value: "utilities", label: "Utilities" },
  ];

  const productTypeOptions = [
    { value: "raw_material", label: "Raw Material" },
    { value: "semi_finished", label: "Semi-Finished Product" },
    { value: "finished_good", label: "Finished Good" },
    { value: "service", label: "Service" },
    { value: "equipment", label: "Equipment" },
    { value: "software", label: "Software" },
    { value: "chemical", label: "Chemical Compound" },
    { value: "machinery", label: "Machinery" },
    { value: "packaging", label: "Packaging Material" },
    { value: "electronics", label: "Electronic Device" },
    { value: "automotive_part", label: "Automotive Part" },
    { value: "food_item", label: "Food Item" },
    { value: "pharma_drug", label: "Pharmaceutical Drug" },
    { value: "textile_fabric", label: "Textile/Fabric" },
  ];

  useEffect(() => {
    async function getRowCount() {
      if (fileObject && fileType?.includes("sheet")) {
        const ExcelJS = await import("exceljs");
        const workbook = new ExcelJS.Workbook();
        const buffer = await fileObject.arrayBuffer();
        await workbook.xlsx.load(buffer);
        const worksheet = workbook.worksheets[0];
        setNumRows(worksheet.rowCount);
      }
    }
    getRowCount();
  }, [fileObject, fileType]);

  function prettyFileSize(size) {
    if (!size) return "";
    if (size > 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
    if (size > 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${size} bytes`;
  }

  // 🧠 Detect file type to show icon
  function getFileIcon() {
    // if (!uploadedFilename) return defaultIcon;
    const name = uploadedFilename.toLowerCase();
    if (name.endsWith(".xlsx") || name.endsWith(".xls")) return excelIcon;
    if (name.endsWith(".pdf")) return pdfIcon;
    if (name.endsWith(".doc") || name.endsWith(".docx")) return wordIcon;
    if (name.endsWith(".txt")) return textIcon;
    // if (name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg")) return imageIcon;
    // return defaultIcon;
  }

  async function handleProcessClick() {
    if (!fileObject) {
      setError("⚠️ No file uploaded. Please upload a file first.");
      return;
    }

    if (!industry || !productType) {
      setError("⚠️ Please select both Industry and Product Type.");
      return;
    }

    setLoading(true);
    setError("");
    setProgress(0);

    const progressInterval = setInterval(() => {
      setProgress((prev) => (prev < 90 ? prev + Math.random() * 5 : prev));
    }, 400);

    try {
      const formData = new FormData();
      formData.append("file", fileObject);
      formData.append("industry", industry.value);
      formData.append("productType", productType.value);

      const res = await fetch("http://localhost:5000/process", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      clearInterval(progressInterval);
      setProgress(100);

      setTimeout(() => {
        // 🧭 Check model type — if it's Mistral (PDF), go directly to Aggregated View
        if (data.model_used === "mistral") {
          navigate("/extraction_review", {
            state: {
              uploadedFilename,
              sku_matrix: [], // hide SKU-level data
              aggregated_matrix: data.aggregated_matrix || { columns: [], rows: [] },
              modelUsed: data.model_used || "Mistral",
              defaultView: "aggregated", // 🧠 tell review page to show aggregated view
            },
          });
        } else {
          // Default behavior for Excel (LLaMA)
          navigate("/extraction_review", {
            state: {
              uploadedFilename,
              sku_matrix: data.sku_matrix || [],
              aggregated_matrix: data.aggregated_matrix || { columns: [], rows: [] },
              modelUsed: data.model_used || "Unknown Model",
              defaultView: "sku",
            },
          });
        }
      }, 800);

    } catch (e) {
      clearInterval(progressInterval);
      setError("Failed to process file: " + e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        background: "linear-gradient(135deg, #667eea, #764ba2)",
        minHeight: "100vh",
        paddingBottom: "60px",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          padding: "24px 24px",
        }}
      >
        <img
          src="https://img.icons8.com/fluency/96/000000/bot.png"
          alt="bot"
          style={{ width: 48, height: 48, marginRight: 16 }}
        />
        <h2 style={{ margin: 0, fontWeight: 700, color: "#1e1e1e" }}>
          AI Extraction Automation - Document Review Page
        </h2>
      </header>

      <main
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
          marginTop: 40,
          gap: "40px",
        }}
      >
        {/* LEFT COLUMN - FILE ICON */}
        <div
          style={{
            background: "rgba(255,255,255,0.75)",
            borderRadius: 20,
            padding: "50px 60px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
            backdropFilter: "blur(8px)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            flexDirection: "column",
            minWidth: "200px",
          }}
        >
          <img
            src={getFileIcon()}
            alt="file icon"
            style={{
              width: 120,
              height: 120,
              objectFit: "contain",
              filter: "drop-shadow(0px 4px 8px rgba(0,0,0,0.3))",
            }}
          />
        </div>

        {/* RIGHT COLUMN - DETAILS */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            background: "rgba(255,255,255,0.75)",
            borderRadius: 20,
            padding: "40px 60px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
            backdropFilter: "blur(8px)",
            width: "600px",
          }}
        >
          <b style={{ fontSize: "20px", marginBottom: 8 }}>
            {uploadedFilename || "No file uploaded"}
          </b>
          <div style={{ color: "#555" }}>
            {fileSize ? `File size: ${prettyFileSize(fileSize)}` : null}
          </div>
          <div style={{ color: "#555", marginBottom: 10 }}>
            {uploadedDate ? `Uploaded: ${uploadedDate}` : null}
          </div>

          {numRows && (
            <div
              style={{
                color: "#2563eb",
                fontWeight: 600,
                marginBottom: 14,
              }}
            >
              Rows detected in Excel: {numRows}
            </div>
          )}

          {/* Dropdowns */}
          <div style={{ width: "100%", marginTop: 20 }}>
            <label
              style={{ fontWeight: 600, marginBottom: 4, display: "block" }}
            >
              Industry
            </label>
            <Select
              options={industryOptions}
              value={industry}
              onChange={setIndustry}
              isSearchable
              placeholder="Select Industry..."
            />
          </div>

          <div style={{ width: "100%", marginTop: 20 }}>
            <label
              style={{ fontWeight: 600, marginBottom: 4, display: "block" }}
            >
              Product Type
            </label>
            <Select
              options={productTypeOptions}
              value={productType}
              onChange={setProductType}
              isSearchable
              placeholder="Select Product Type..."
            />
          </div>

          <button
            style={{
              marginTop: 30,
              padding: "14px 36px",
              fontSize: "18px",
              color: "#fff",
              background:
                !industry || !productType
                  ? "#94a3b8"
                  : "linear-gradient(135deg,#6366f1,#3b82f6)",
              border: "none",
              borderRadius: "10px",
              cursor:
                !industry || !productType ? "not-allowed" : "pointer",
              transition: "0.3s",
              boxShadow:
                !industry || !productType
                  ? "none"
                  : "0 4px 16px rgba(59,130,246,0.3)",
            }}
            onClick={handleProcessClick}
            disabled={!industry || !productType || loading}
          >
            {loading ? "Processing..." : "Process with AI Model"}
          </button>

          {loading && (
            <>
              <div
                style={{
                  width: "100%",
                  height: "14px",
                  background: "rgba(0,0,0,0.1)",
                  borderRadius: "8px",
                  overflow: "hidden",
                  marginTop: "16px",
                  boxShadow: "inset 0 0 6px rgba(0,0,0,0.1)",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${progress}%`,
                    background:
                      "linear-gradient(90deg, #42a5f5, #1976d2)",
                    borderRadius: "8px",
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
              <div
                style={{
                  fontSize: "16px",
                  fontWeight: "500",
                  color: "#1976d2",
                  marginTop: "8px",
                }}
              >
                {Math.floor(progress)}% completed
              </div>
            </>
          )}

          {error && (
            <div style={{ color: "red", marginTop: "12px" }}>{error}</div>
          )}
        </div>
      </main>
    </div>
  );
}
