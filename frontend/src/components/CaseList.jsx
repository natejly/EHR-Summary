import React, { useState, useEffect } from "react";

const T = {
  text: "#1d1d1f",
  textSecondary: "#6e6e73",
  textTertiary: "#aeaeb2",
  accent: "#0071e3",
  accentBg: "rgba(0,113,227,0.08)",
  border: "rgba(0,0,0,0.08)",
  surface: "#ffffff",
  green: "#34c759",
  orange: "#ff9f0a",
};

const STAGE_SHORT = {
  stage1_evidence: "S1", stage2_extract: "S2", stage3_verify: "S3",
  stage4_context: "S4", stage5_fact_sheet: "S5", stage6_summarize: "S6",
  stage7_check: "S7", stage8_review: "S8", stage9_patient_summary: "S9",
};
const ALL_STAGES = Object.keys(STAGE_SHORT);

function Spinner() {
  return (
    <span style={{
      display: "inline-block", width: 11, height: 11,
      border: `2px solid ${T.orange}`, borderTopColor: "transparent",
      borderRadius: "50%", animation: "spin 0.7s linear infinite",
    }} />
  );
}

export default function CaseList({ cases, selectedCase, runningCase, onSelect, onRun, onRefresh }) {
  const [benchCases, setBenchCases] = useState([]);
  const [showBench, setShowBench] = useState(false);
  const [hovered, setHovered] = useState(null);

  useEffect(() => {
    fetch("/api/bench-cases").then(r => r.json()).then(setBenchCases).catch(() => {});
  }, []);

  const pendingBenchCases = benchCases.filter(b => !cases.some(c => c.case_id === b.case_id));

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 0 4px" }}>

        {/* Section label */}
        <div style={{
          fontSize: 10, fontWeight: 700, color: T.textTertiary,
          letterSpacing: "0.07em", textTransform: "uppercase",
          padding: "8px 16px 4px",
        }}>
          Cases
        </div>

        {cases.length === 0 && (
          <div style={{ padding: "8px 16px", fontSize: 12, color: T.textTertiary }}>
            No cases found in outputs/
          </div>
        )}

        {cases.map(c => {
          const isSel = c.case_id === selectedCase;
          const isRun = c.case_id === runningCase;
          const isHov = hovered === c.case_id;

          return (
            <div
              key={c.case_id}
              onClick={() => !isRun && onSelect(c.case_id)}
              onMouseEnter={() => setHovered(c.case_id)}
              onMouseLeave={() => setHovered(null)}
              style={{
                padding: "9px 14px",
                margin: "1px 8px",
                borderRadius: 8,
                cursor: isRun ? "default" : "pointer",
                background: isSel
                  ? T.accentBg
                  : isHov
                  ? "rgba(0,0,0,0.04)"
                  : "transparent",
                transition: "background 0.1s",
                display: "flex",
                flexDirection: "column",
                gap: 5,
                border: isSel ? `1px solid rgba(0,113,227,0.2)` : "1px solid transparent",
              }}
            >
              {/* Case name */}
              <div style={{
                fontSize: 12, fontWeight: 600,
                color: isSel ? T.accent : T.text,
                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                letterSpacing: "-0.01em",
              }} title={c.case_id}>
                {c.case_id.replace(/^mimic-/, "")}
              </div>

              {/* Stage pips */}
              <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
                {ALL_STAGES.map(stage => {
                  const done = c.stages_present.includes(stage);
                  return (
                    <span key={stage} title={stage} style={{
                      fontSize: 9, fontWeight: 700,
                      padding: "1px 4px", borderRadius: 4,
                      background: done ? (isSel ? "rgba(0,113,227,0.15)" : "rgba(52,199,89,0.12)") : "rgba(0,0,0,0.04)",
                      color: done ? (isSel ? T.accent : T.green) : T.textTertiary,
                      border: `1px solid ${done ? (isSel ? "rgba(0,113,227,0.25)" : "rgba(52,199,89,0.25)") : "rgba(0,0,0,0.06)"}`,
                    }}>
                      {STAGE_SHORT[stage]}
                    </span>
                  );
                })}
              </div>

              {/* Actions */}
              <div style={{ display: "flex", gap: 5 }}>
                {isRun ? (
                  <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: T.orange, fontWeight: 600 }}>
                    <Spinner /> Running…
                  </span>
                ) : (
                  <>
                    <button
                      onClick={e => { e.stopPropagation(); onSelect(c.case_id); }}
                      style={btnStyle("ghost", isSel)}
                    >
                      View
                    </button>
                    <button
                      disabled={!!runningCase}
                      onClick={e => { e.stopPropagation(); onRun(c.case_id); }}
                      style={btnStyle("primary", false, !!runningCase)}
                      title="Re-run pipeline"
                    >
                      ↺ Re-run
                    </button>
                  </>
                )}
              </div>
            </div>
          );
        })}

        {/* Available to run */}
        {pendingBenchCases.length > 0 && (
          <>
            <div style={{ height: 1, background: T.border, margin: "8px 16px" }} />
            <div
              onClick={() => setShowBench(v => !v)}
              style={{
                fontSize: 10, fontWeight: 700, color: T.textTertiary,
                letterSpacing: "0.07em", textTransform: "uppercase",
                padding: "8px 16px 4px", cursor: "pointer", userSelect: "none",
                display: "flex", alignItems: "center", gap: 6,
              }}
            >
              Available to Run
              <span style={{ fontSize: 8 }}>{showBench ? "▼" : "▶"}</span>
            </div>
            {showBench && pendingBenchCases.map(b => (
              <div key={b.case_id} style={{
                padding: "9px 14px", margin: "1px 8px", borderRadius: 8,
                display: "flex", flexDirection: "column", gap: 5,
              }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: T.textSecondary }}>
                  {b.case_id.replace(/^mimic-/, "")}
                </div>
                <button
                  disabled={!!runningCase}
                  onClick={() => onRun(b.case_id)}
                  style={btnStyle("primary", false, !!runningCase)}
                >
                  Run Pipeline
                </button>
              </div>
            ))}
          </>
        )}
      </div>

      {/* Refresh footer */}
      <div style={{
        padding: "10px 16px",
        borderTop: `1px solid ${T.border}`,
        display: "flex",
        justifyContent: "flex-end",
      }}>
        <button
          onClick={onRefresh}
          style={{
            fontSize: 11, color: T.textTertiary, background: "none",
            border: "none", cursor: "pointer", padding: "3px 6px",
            borderRadius: 6, transition: "color 0.1s",
          }}
          onMouseEnter={e => e.target.style.color = T.textSecondary}
          onMouseLeave={e => e.target.style.color = T.textTertiary}
        >
          ↻ Refresh
        </button>
      </div>
    </div>
  );
}

function btnStyle(variant, active, disabled) {
  const base = {
    fontSize: 10, fontWeight: 600, padding: "3px 9px", borderRadius: 6,
    border: "none", cursor: disabled ? "not-allowed" : "pointer",
    transition: "all 0.1s", opacity: disabled ? 0.4 : 1,
  };
  if (variant === "primary") {
    return { ...base, background: "rgba(0,113,227,0.1)", color: "#0071e3", border: "1px solid rgba(0,113,227,0.2)" };
  }
  return { ...base, background: active ? "rgba(0,113,227,0.06)" : "rgba(0,0,0,0.04)", color: active ? "#0071e3" : "#6e6e73" };
}
