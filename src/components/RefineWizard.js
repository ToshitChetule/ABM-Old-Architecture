import React, { useEffect, useState } from "react";
import {
    Select,
    MenuItem,
    TextField,
    Button,
    Typography,
    CircularProgress,
    IconButton,
    Tooltip,
} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import axios from "axios";
import Swal from "sweetalert2";

export default function RefineWizard({ onRefresh }) {
    const [actionType, setActionType] = useState("");
    const [targetType, setTargetType] = useState("");
    const [selectedAttr, setSelectedAttr] = useState("");
    const [oldValue, setOldValue] = useState("");
    const [newValue, setNewValue] = useState("");
    const [attributes, setAttributes] = useState([]);
    const [queue, setQueue] = useState([]);
    const [loading, setLoading] = useState(false);

    // ✅ Load all available attributes from Neo4j
    useEffect(() => {
        axios
            .get("http://localhost:5000/graph/aggregated")
            .then((res) => {
                if (res.data?.rows) {
                    setAttributes(res.data.rows.map((r) => r[0]));
                }
            })
            .catch((err) => console.error("⚠️ Failed to fetch attributes:", err));
    }, []);

    // ✅ Add to queue
    const addToQueue = () => {
        if (!actionType || !targetType) {
            Swal.fire("Please select both Action and Target Type", "", "info");
            return;
        }

        const entry = {
            id: Date.now(),
            actionType,
            targetType,
            attribute: selectedAttr,
            oldValue,
            newValue,
        };

        setQueue((prev) => [...prev, entry]);
        setActionType("");
        setTargetType("");
        setSelectedAttr("");
        setOldValue("");
        setNewValue("");
    };

    // ✅ Remove individual queue item
    const removeFromQueue = (id) => {
        setQueue((prev) => prev.filter((item) => item.id !== id));
    };

    // ✅ Clear queue
    const clearQueue = () => {
        if (queue.length === 0) return;
        Swal.fire({
            title: "Clear All?",
            text: "This will remove all queued refinements.",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#3085d6",
            cancelButtonColor: "#d33",
            confirmButtonText: "Yes, clear all",
        }).then((res) => {
            if (res.isConfirmed) setQueue([]);
        });
    };

    // ✅ Apply all queued refinements
    const applyAll = async () => {
        if (queue.length === 0) {
            Swal.fire("No refinements to apply", "", "info");
            return;
        }

        setLoading(true);

        const combinedPrompt = queue
            .map((q) => {
                if (q.actionType === "Rename" && q.targetType === "Attribute")
                    return `rename attribute ${q.attribute} to ${q.newValue}`;
                if (q.actionType === "Rename" && q.targetType === "Value")
                    return `rename value ${q.oldValue} to ${q.newValue}`;
                if (q.actionType === "Add")
                    return `add value ${q.newValue} under ${q.attribute}`;
                if (q.actionType === "Remove")
                    return `remove value ${q.oldValue} under ${q.attribute}`;
                if (q.actionType === "Delete")
                    return `delete attribute ${q.attribute}`;
                return "";
            })
            .filter(Boolean)
            .join(" and ");

        try {
            const res = await axios.post("http://localhost:5000/refine_graph", {
                prompt: combinedPrompt,
                attributes: attributes,
            });

            if (res.data.status === "success") {
                onRefresh();
                setQueue([]);
                Swal.fire({
                    icon: "success",
                    title: "Refinements Applied!",
                    html: `<p style="text-align:left"><b>Actions:</b><br>${res.data.actions
                        .map((a) => `• ${a}`)
                        .join("<br>")}</p>`,
                    confirmButtonText: "OK",
                });
            } else {
                Swal.fire("Refinement Failed", res.data.error || "Unknown error", "error");
            }
        } catch (err) {
            console.error(err);
            Swal.fire("Error", err.message, "error");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div
            style={{
                background: "rgba(255,255,255,0.95)",
                backdropFilter: "blur(8px)",
                boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
                padding: "16px 24px",
                borderBottom: "1px solid #e2e8f0",
            }}
        >
            <Typography
                variant="h6"
                fontWeight={700}
                sx={{ mb: 1, color: "#1e293b", display: "flex", alignItems: "center" }}
            >
                🧠 Refine Wizard
            </Typography>

            {/* Input Controls */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center" }}>
                <Select
                    value={actionType}
                    onChange={(e) => setActionType(e.target.value)}
                    displayEmpty
                    size="small"
                    style={{ minWidth: 130 }}
                >
                    <MenuItem value="">Action</MenuItem>
                    <MenuItem value="Rename">Rename</MenuItem>
                    <MenuItem value="Add">Add</MenuItem>
                    <MenuItem value="Remove">Remove</MenuItem>
                    <MenuItem value="Delete">Delete</MenuItem>
                </Select>

                {/* Target Type Dropdown (auto-locks for Add/Remove) */}
                <Select
                    value={targetType}
                    onChange={(e) => setTargetType(e.target.value)}
                    displayEmpty
                    size="small"
                    style={{ minWidth: 130 }}
                    disabled={["Add", "Remove","Delete"].includes(actionType)} // disable for Add or Remove
                >
                    <MenuItem value="">Target</MenuItem>
                    <MenuItem value="Attribute">Attribute</MenuItem>
                    <MenuItem value="Value">Value</MenuItem>
                </Select>

                {/* Automatically fix target when Add or Remove is chosen */}
                {["Add", "Remove"].includes(actionType) && targetType !== "Value" && setTargetType("Value")}
                {actionType === "Delete" && targetType !== "Attribute" && setTargetType("Attribute")}


                <Select
                    value={selectedAttr}
                    onChange={(e) => setSelectedAttr(e.target.value)}
                    displayEmpty
                    size="small"
                    style={{ minWidth: 180 }}
                >
                    <MenuItem value="">Select Attribute</MenuItem>
                    {attributes.map((a, i) => (
                        <MenuItem key={i} value={a}>
                            {a}
                        </MenuItem>
                    ))}
                </Select>

                {/* Old Value (for rename/remove) */}
                {(actionType === "Rename" || actionType === "Remove") &&
                    targetType === "Value" && (
                        <TextField
                            label="Old Value"
                            size="small"
                            value={oldValue}
                            onChange={(e) => setOldValue(e.target.value)}
                            style={{ minWidth: 140 }}
                        />
                    )}

                {/* New Value (for rename/add) */}
                {(actionType === "Rename" || actionType === "Add") && (
                    <TextField
                        label="New Value"
                        size="small"
                        value={newValue}
                        onChange={(e) => setNewValue(e.target.value)}
                        style={{ minWidth: 140 }}
                    />
                )}

                <Button
                    variant="contained"
                    sx={{
                        background: "linear-gradient(135deg,#6366f1,#3b82f6)",
                        textTransform: "none",
                    }}
                    onClick={addToQueue}
                >
                    + Add
                </Button>

                <Button
                    variant="contained"
                    color="success"
                    sx={{ textTransform: "none" }}
                    onClick={applyAll}
                    disabled={loading}
                >
                    {loading ? <CircularProgress size={20} color="inherit" /> : "Apply All"}
                </Button>

                <Button
                    variant="outlined"
                    color="error"
                    sx={{ textTransform: "none" }}
                    onClick={clearQueue}
                >
                    Clear All
                </Button>
            </div>

            {/* Queue Table */}
            {queue.length > 0 && (
                <div
                    style={{
                        marginTop: 12,
                        background: "rgba(241,245,249,0.7)",
                        borderRadius: 8,
                        padding: "10px",
                        maxHeight: "180px",
                        overflowY: "auto",
                    }}
                >
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                        🧾 Pending Refinements ({queue.length})
                    </Typography>
                    <table
                        style={{
                            width: "100%",
                            borderCollapse: "collapse",
                            fontSize: "14px",
                            textAlign: "left",
                        }}
                    >
                        <thead>
                            <tr style={{ background: "#e0e7ff" }}>
                                <th style={{ padding: "6px 8px" }}>#</th>
                                <th style={{ padding: "6px 8px" }}>Action</th>
                                <th style={{ padding: "6px 8px" }}>Target</th>
                                <th style={{ padding: "6px 8px" }}>Attribute</th>
                                <th style={{ padding: "6px 8px" }}>Old</th>
                                <th style={{ padding: "6px 8px" }}>New</th>
                                <th style={{ padding: "6px 8px" }}>Remove</th>
                            </tr>
                        </thead>
                        <tbody>
                            {queue.map((q, i) => (
                                <tr key={q.id}>
                                    <td style={{ padding: "6px 8px" }}>{i + 1}</td>
                                    <td style={{ padding: "6px 8px" }}>{q.actionType}</td>
                                    <td style={{ padding: "6px 8px" }}>{q.targetType}</td>
                                    <td style={{ padding: "6px 8px" }}>{q.attribute}</td>
                                    <td style={{ padding: "6px 8px" }}>{q.oldValue}</td>
                                    <td style={{ padding: "6px 8px" }}>{q.newValue}</td>
                                    <td style={{ padding: "6px 8px" }}>
                                        <Tooltip title="Remove from Queue">
                                            <IconButton
                                                size="small"
                                                color="error"
                                                onClick={() => removeFromQueue(q.id)}
                                            >
                                                <DeleteIcon fontSize="small" />
                                            </IconButton>
                                        </Tooltip>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
