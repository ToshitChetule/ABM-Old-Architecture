import React, { useRef } from "react";
import "./FileUploader.css";
import excel from "../icons/excel.png";
import doc from "../icons/doc.png";
import pdf from "../icons/pdf.png";

import txt from "../icons/txt-file.png";


export default function FileUploader({ onFileSelected, uploadedFilename }) {
  const fileInputRef = useRef(null);

  

  function handleButtonClick() {
    fileInputRef.current?.click();
  }

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (file) {
      onFileSelected(file);
    }
  }

  return (
    <div className="upload-widget">
      <div className="upload-icon">📄</div>
      <div className="upload-title"><b>Upload Document</b></div>
      <div className="upload-desc">Drag and drop your file, or click to browse</div>

      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        accept=".xlsx,.pdf,.docx,.txt"
        onChange={handleFileChange}
      />

      <button type="button" onClick={handleButtonClick}>Choose File</button>

      {/* ✅ show uploaded filename clearly */}
      {uploadedFilename ? (
        <p style={{ marginTop: "16px", fontWeight: "500", color: "black" }}>
          ✅ {uploadedFilename} uploaded successfully
        </p>
      ) : (
        <p style={{ marginTop: "16px", color: "rgba(255,255,255,0.6)" }}>
          <br/>
          No file uploaded yet
        </p>
      )}
  <br/>
      <div style={{ marginTop: 16 }}>
        <img src={excel} alt="logo" className="upload-logo" style={{ width: 42, height: 42,marginLeft:"15px" }} />
        <img src={doc} alt="logo" className="upload-logo" style={{ width: 42, height: 42,marginLeft:"15px" }} />
        <img src={pdf} alt="logo" className="upload-logo" style={{ width: 42, height: 42,marginLeft:"15px"  }} />
        <img src={txt} alt="logo" className="upload-logo" style={{ width: 42, height: 42,marginLeft:"15px"  }} />
      </div>
    </div>
  );
}
