import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../logo.svg";
import "./LoginPage.css";
import Snackbar from "@mui/material/Snackbar";
import Alert from "@mui/material/Alert";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [toast, setToast] = useState({ open: false, message: "", severity: "success" });
  const navigate = useNavigate();

  function showToast(message, severity = "success") {
    setToast({ open: true, message, severity });
    setTimeout(() => setToast({ open: false, message: "", severity }), 3000);
  }

  // Handle Sign Up or Sign In
  function handleSubmit(e) {
    e.preventDefault();

    if (isRegistering) {
      // Registration flow
      if (!username || !password) {
        setError("Please fill in both fields.");
        return;
      }
      const newUser = { username, password };
      localStorage.setItem("registeredUser", JSON.stringify(newUser));
      setUsername("");
      setPassword("");
      setIsRegistering(false);
      setError("");
      showToast("✅ Registered successfully! Please log in now.", "success");
    } else {
      // Login flow
      const stored = localStorage.getItem("registeredUser");
      if (!stored) {
        setError("No registered user found. Please sign up first.");
        return;
      }
      const { username: storedUsername, password: storedPassword } = JSON.parse(stored);

      if (username === storedUsername && password === storedPassword) {
        localStorage.setItem("user", JSON.stringify({ username }));
        setError("");
        showToast("✅ Login successful!", "success");
        setTimeout(() => navigate("/upload"), 1500);
      } else {
        setError("Invalid credentials. Please try again.");
        showToast("❌ Invalid credentials.", "error");
      }
    }
  }

  return (
    <div className="login-root">
      <header className="login-header">
        <img src="https://img.icons8.com/fluency/96/000000/bot.png" alt="bot"
          style={{ width: 48, height: 48, marginRight: 16 }} />
        <h2>AI Extraction Automation</h2>
      </header>

      <div className="login-body">
        <section className="login-features">
          <h1>Extract & Validate Data with AI Precision</h1>
          <ul>
            <li>
              <strong>AI-Powered Extraction</strong>: Automatically extract key
              fields from any document.
            </li>
            <li>
              <strong>Confidence Scoring</strong>: See how sure the AI is, at a
              glance.
            </li>
            <li>
              <strong>Validate & Export</strong>: Review, edit, and export to
              your preferred format.
            </li>
          </ul>
        </section>

        <section className="login-formbox">
          <h3>{isRegistering ? "Create Account ✨" : "Welcome Back 👋"}</h3>
          <form onSubmit={handleSubmit}>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Username"
              autoFocus
            />
            <input
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              type="password"
            />
            <button type="submit">
              {isRegistering ? "Register" : "Sign In"}
            </button>
          </form>

          {error && <div style={{ color: "red", marginTop: "8px" }}>{error}</div>}

          <div style={{ marginTop: "15px", textAlign: "center" }}>
            {isRegistering ? (
              <p style={{ color: "#555" }}>
                Already have an account?{" "}
                <span
                  style={{
                    color: "#0077c6",
                    cursor: "pointer",
                    textDecoration: "underline",
                  }}
                  onClick={() => {
                    setIsRegistering(false);
                    setError("");
                  }}
                >
                  Sign In
                </span>
              </p>
            ) : (
              <p style={{ color: "#555" }}>
                Don’t have an account?{" "}
                <span
                  style={{
                    color: "#0077c6",
                    cursor: "pointer",
                    textDecoration: "underline",
                  }}
                  onClick={() => {
                    setIsRegistering(true);
                    setError("");
                  }}
                >
                  Register here
                </span>
              </p>
            )}
          </div>
        </section>
      </div>

      {/* Snackbar Toast */}
      <Snackbar
        open={toast.open}
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
      >
        <Alert severity={toast.severity} sx={{ width: "100%" }}>
          {toast.message}
        </Alert>
      </Snackbar>
    </div>
  );
}
