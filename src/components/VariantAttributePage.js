import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Swal from "sweetalert2";
import FileUploader from "./FileUploader";

export default function VariantAttributePage() {
  const [file, setFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState("");
  const [processing, setProcessing] = useState(false);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [outputFile, setOutputFile] = useState("");
  const navigate = useNavigate();

  // ✅ Keep user logged in
  useEffect(() => {
    const user = localStorage.getItem("user");
    if (!user) navigate("/");
  }, [navigate]);

  const handleFileSelected = (selectedFile) => {
    setFile(selectedFile);
    setUploadedFilename(selectedFile.name);
    setAnalysisReady(false); // Reset state if user uploads a new file
  };

  const handleProcess = async () => {
    if (!file) return;

    setProcessing(true);
    setAnalysisReady(false);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:5000/process_variant", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || "Processing failed");
      }

      setOutputFile(data.filename);
      setAnalysisReady(true);

      Swal.fire({
        icon: "success",
        title: "Analysis Completed!",
        text: "Variant attribute analysis finished successfully.",
        showConfirmButton: false,
        timer: 1800,
      });
    } catch (err) {
      Swal.fire("Error", err.message, "error");
    } finally {
      setProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!outputFile) return;
    window.open(`http://localhost:5000/download/${outputFile}`, "_blank");
  };

  const handleBack = () => {
    navigate("/extraction_review");
  };

  // 🚪 Logout handler
  const handleLogout = () => {
    Swal.fire({
      title: "Logout?",
      text: "Are you sure you want to log out?",
      icon: "question",
      showCancelButton: true,
      confirmButtonText: "Yes, Logout",
      cancelButtonText: "Cancel",
      confirmButtonColor: "#3b82f6",
      cancelButtonColor: "#94a3b8",
    }).then((result) => {
      if (result.isConfirmed) {
        localStorage.removeItem("user");
        Swal.fire({
          icon: "success",
          title: "Logged out successfully!",
          timer: 1200,
          showConfirmButton: false,
        });
        setTimeout(() => navigate("/"), 1200);
      }
    });
  };

  // ✨ Shared button style
  const buttonBase = {
    padding: "12px 32px",
    border: "none",
    borderRadius: "10px",
    fontSize: 16,
    fontWeight: 600,
    color: "white",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
    transition:
      "all 0.3s ease, transform 0.2s ease-in-out, box-shadow 0.3s ease",
  };

  return (
    <div
      style={{
        background: "linear-gradient(135deg, #667eea, #764ba2)",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        fontFamily: "'Inter', sans-serif",
      }}
    >
      {/* HEADER */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          width: "100%",
          padding: "20px 40px",
          background: "linear-gradient(135deg, #667eea, #764ba2)",
          boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
          backdropFilter: "blur(10px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center" }}>
          <img
            src="https://img.icons8.com/fluency/96/000000/bot.png"
            alt="bot"
            style={{ width: 48, height: 48, marginRight: 16 }}
          />
          <h2 style={{ margin: 0, fontWeight: 700 }}>
            Variant Attribute Analysis
          </h2>
        </div>

        <div style={{ display: "flex", gap: "12px" }}>
          {/* ← Back Button */}
          <button
            onClick={handleBack}
            style={{
              ...buttonBase,
              background: "linear-gradient(135deg,#6366f1,#3b82f6)",
            }}
            onMouseOver={(e) => {
              e.target.style.transform = "translateY(-3px)";
              e.target.style.boxShadow = "0 8px 25px rgba(59,130,246,0.4)";
            }}
            onMouseOut={(e) => {
              e.target.style.transform = "translateY(0)";
              e.target.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
            }}
          >
            ← Back
          </button>

          {/* 🚪 Logout Button */}
          <button
            onClick={handleLogout}
            style={{
              ...buttonBase,
              background: "linear-gradient(135deg,#ef4444,#dc2626)",
            }}
            onMouseOver={(e) => {
              e.target.style.transform = "translateY(-3px)";
              e.target.style.boxShadow = "0 8px 25px rgba(239,68,68,0.4)";
            }}
            onMouseOut={(e) => {
              e.target.style.transform = "translateY(0)";
              e.target.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
            }}
          >
            Logout
          </button>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main style={{ marginTop: 40, width: "90%", maxWidth: 800 }}>
        <FileUploader
          onFileSelected={handleFileSelected}
          uploadedFilename={uploadedFilename}
        />

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 20,
            marginTop: 30,
          }}
        >
          {/* Process Button */}
          <button
            onClick={handleProcess}
            disabled={!file || processing}
            style={{
              ...buttonBase,
              background: file
                ? "linear-gradient(135deg, #16a34a, #22c55e)"
                : "#94a3b8",
              cursor: file ? "pointer" : "not-allowed",
              opacity: processing ? 0.8 : 1,
            }}
            onMouseOver={(e) => {
              if (file) {
                e.target.style.transform = "translateY(-3px)";
                e.target.style.boxShadow = "0 8px 25px rgba(34,197,94,0.4)";
              }
            }}
            onMouseOut={(e) => {
              if (file) {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
              }
            }}
          >
            {processing ? "Processing..." : "Process"}
          </button>

          {/* Open Analysis Button (only after success) */}
          {analysisReady && (
            <button
              onClick={handleDownload}
              style={{
                ...buttonBase,
                background: "linear-gradient(135deg, #3b82f6, #2563eb)",
              }}
              onMouseOver={(e) => {
                e.target.style.transform = "translateY(-3px)";
                e.target.style.boxShadow = "0 8px 25px rgba(59,130,246,0.4)";
              }}
              onMouseOut={(e) => {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
              }}
            >
              Open Analysis
            </button>
          )}
        </div>
      </main>
    </div>
  );
}
