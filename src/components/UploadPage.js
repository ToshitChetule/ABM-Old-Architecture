import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import FileUploader from "./FileUploader";
import "./UploadPage.css";
import logo from "../logo.svg";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState("");
  const navigate = useNavigate();

  // Block unauthenticated access: redirect to login if not logged in
  useEffect(() => {
    const user = localStorage.getItem("user");
    if (!user) {
      navigate("/");
    }
  }, [navigate]);

  // Called when file is chosen in FileUploader
  async function handleFileSelected(selectedFile) {
    // store file reference and filename for UI
    setFile(selectedFile);
    setUploadedFilename(selectedFile.name);
    // If later you want to upload to backend, do it here (fetch / axios)
  }

  // function handleContinue() {
  //   if (!uploadedFilename) return;
  //   // Save the uploaded file metadata into localStorage or context if needed
  //   localStorage.setItem("uploadedFileName", uploadedFilename);
  //   // Navigate to the next step (Doc review)
  //   navigate("/review", { state: { uploadedFile: file, uploadedFilename } });
  // }

  function handleContinue() {
    if (!file) return;

    const fileSize = file.size;
    const fileType = file.type || "Unknown type";
    const uploadedDate = new Date().toLocaleString();

    // Save if needed
    localStorage.setItem("uploadedFileName", file.name);

    // Pass all metadata to review page
    navigate("/review", {
      state: {
        uploadedFilename: file.name,
        fileSize: file.size,
        fileType: file.type,
        uploadedDate: new Date().toLocaleString(),
        fileObject: file, // ✅ important
      },
    });
  }


  function handleLogout() {
    localStorage.removeItem("user");
    navigate("/");
  }

  return (
    <div className="upload-root">
      <header className="upload-header" style={{ display: "flex", alignItems: "center", gap: 16, padding: 20 }}>
        <img src="https://img.icons8.com/fluency/96/000000/bot.png" alt="bot"
          style={{ width: 48, height: 48, marginRight: 16 }} />
        <h2>AI Extraction Automation - Upload</h2>
        <div
          style={{
            marginLeft: "auto",
            display: "flex",
            gap: 12,
            alignItems: "center",
          }}
        >
          <button
            onClick={handleLogout}
            style={{
              padding: "10px 18px",
              cursor: "pointer",
              background: "linear-gradient(135deg, #667eea, #764ba2)",
              color: "white",
              border: "none",
              borderRadius: "10px",
              fontSize: "15px",
              fontWeight: 500,
              letterSpacing: "0.3px",
              boxShadow: "0 4px 15px rgba(118, 75, 162, 0.4)",
              transition: "all 0.3s ease",
              backdropFilter: "blur(8px)",
            }}
            onMouseOver={(e) => {
              e.target.style.transform = "translateY(-3px)";
              e.target.style.boxShadow = "0 8px 25px rgba(118, 75, 162, 0.6)";
            }}
            onMouseOut={(e) => {
              e.target.style.transform = "translateY(0)";
              e.target.style.boxShadow = "0 4px 15px rgba(118, 75, 162, 0.4)";
            }}
          >
            Logout
          </button>
        </div>

      </header>

      <main style={{ padding: 28 }}>
        <div style={{ maxWidth: 980, margin: "0 auto", display: "flex", flexDirection: "column", alignItems: "center" }}>
          <FileUploader
            onFileSelected={handleFileSelected}
            uploadedFilename={uploadedFilename}
          />


          <div style={{ marginTop: 24 }}>
            <button
              style={{
                marginTop: "12px",
                padding: "12px 36px",
                fontSize: 16,
                borderRadius: 12,
                cursor: uploadedFilename ? "pointer" : "not-allowed",
                background: uploadedFilename
                  ? "linear-gradient(135deg, #3a8dff, #6b73ff)"
                  : "#cbd5e1",
                color: uploadedFilename ? "#fff" : "#666",
                border: "none",
                boxShadow: uploadedFilename
                  ? "0 4px 15px rgba(58, 141, 255, 0.3)"
                  : "none",
                transition:
                  "all 0.3s ease, transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out",
              }}
              disabled={!uploadedFilename}
              onMouseEnter={(e) => {
                if (uploadedFilename) {
                  e.target.style.transform = "translateY(-3px)";
                  e.target.style.boxShadow = "0 6px 25px rgba(58, 141, 255, 0.45)";
                }
              }}
              onMouseLeave={(e) => {
                if (uploadedFilename) {
                  e.target.style.transform = "translateY(0)";
                  e.target.style.boxShadow = "0 4px 15px rgba(58, 141, 255, 0.3)";
                }
              }}
              onClick={handleContinue}
            >
              Continue
            </button>
          </div>

        </div>
      </main>
    </div>
  );
}
