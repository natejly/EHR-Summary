import React from "react";

const T = {
  accent: "#0071e3",
  green: "#34c759",
  orange: "#ff9f0a",
  textTertiary: "#aeaeb2",
  border: "rgba(0,0,0,0.1)",
};

const STAGES = [
  { key: "stage1_evidence",        short: "S1", label: "Evidence" },
  { key: "stage2_extract",         short: "S2", label: "Extract" },
  { key: "stage3_verify",          short: "S3", label: "Verify" },
  { key: "stage4_context",         short: "S4", label: "Context" },
  { key: "stage5_fact_sheet",      short: "S5", label: "Facts" },
  { key: "stage6_summarize",       short: "S6", label: "Summary" },
  { key: "stage7_check",           short: "S7", label: "Check" },
  { key: "stage8_review",          short: "S8", label: "Review" },
  { key: "stage9_patient_summary", short: "S9", label: "Patient" },
];

function getStatus(key, stagesPresent, isRunning) {
  if (stagesPresent.includes(key)) return "done";
  if (!isRunning) return "pending";
  const idx = STAGES.findIndex(s => s.key === key);
  const prevDone = STAGES.slice(0, idx).every(s => stagesPresent.includes(s.key));
  return prevDone ? "running" : "pending";
}

export default function PipelineStepper({ stagesPresent = [], isRunning = false }) {
  return (
    <>
      <style>{`
        @keyframes stepPulse {
          0%,100% { box-shadow: 0 0 0 0 rgba(255,159,10,0.4); }
          50%      { box-shadow: 0 0 0 4px rgba(255,159,10,0); }
        }
      `}</style>
      <div style={{ display: "flex", alignItems: "center", gap: 0, flexShrink: 0 }}>
        {STAGES.map((stage, idx) => {
          const status = getStatus(stage.key, stagesPresent, isRunning);
          const prevDone = idx === 0 || stagesPresent.includes(STAGES[idx - 1].key);

          const circleStyle = {
            width: 24, height: 24, borderRadius: "50%",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 9, fontWeight: 700, flexShrink: 0,
            transition: "all 0.3s",
            ...(status === "done" ? {
              background: "rgba(52,199,89,0.12)",
              color: T.green,
              border: `1.5px solid rgba(52,199,89,0.4)`,
            } : status === "running" ? {
              background: "rgba(255,159,10,0.12)",
              color: T.orange,
              border: `1.5px solid rgba(255,159,10,0.5)`,
              animation: "stepPulse 1.2s ease-in-out infinite",
            } : {
              background: "rgba(0,0,0,0.04)",
              color: T.textTertiary,
              border: `1.5px solid rgba(0,0,0,0.1)`,
            }),
          };

          return (
            <div key={stage.key} style={{ display: "flex", alignItems: "center" }}>
              {idx > 0 && (
                <div style={{
                  width: 12, height: 1.5, flexShrink: 0, marginBottom: 10,
                  background: prevDone ? "rgba(52,199,89,0.35)" : "rgba(0,0,0,0.08)",
                  transition: "background 0.3s",
                }} />
              )}
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}
                   title={`${stage.key}: ${status}`}>
                <div style={circleStyle}>
                  {status === "done" ? "✓" : stage.short}
                </div>
                <div style={{
                  fontSize: 8, fontWeight: 600, letterSpacing: "0.02em",
                  color: status === "done" ? T.green : status === "running" ? T.orange : T.textTertiary,
                  textAlign: "center",
                }}>
                  {stage.label}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
