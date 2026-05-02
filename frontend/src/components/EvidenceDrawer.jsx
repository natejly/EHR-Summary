import React from "react";

const T = {
  text: "#1d1d1f",
  textSecondary: "#6e6e73",
  textTertiary: "#aeaeb2",
  accent: "#0071e3",
  green: "#34c759",
  red: "#ff3b30",
  orange: "#ff9f0a",
  purple: "#af52de",
  teal: "#32ade6",
  border: "rgba(0,0,0,0.08)",
  surface: "#ffffff",
  bg: "#f5f5f7",
};

const KIND_META = {
  condition:    { color: T.red,    bg: "rgba(255,59,48,0.08)",   label: "Condition" },
  medication:   { color: T.green,  bg: "rgba(52,199,89,0.1)",    label: "Medication" },
  observation:  { color: T.purple, bg: "rgba(175,82,222,0.1)",   label: "Observation" },
  allergy:      { color: T.orange, bg: "rgba(255,159,10,0.1)",   label: "Allergy" },
  procedure:    { color: T.teal,   bg: "rgba(50,173,230,0.1)",   label: "Procedure" },
  encounter:    { color: T.textSecondary, bg: "rgba(0,0,0,0.05)", label: "Encounter" },
  note_sentence:{ color: T.accent, bg: "rgba(0,113,227,0.08)",   label: "Note Sentence" },
};

function KindBadge({ kind }) {
  const meta = KIND_META[kind] ?? { color: T.textTertiary, bg: "rgba(0,0,0,0.05)", label: kind };
  return (
    <span style={{
      display: "inline-block", padding: "3px 10px", borderRadius: 20,
      fontSize: 11, fontWeight: 700,
      background: meta.bg, color: meta.color,
      border: `1px solid ${meta.color}30`,
      textTransform: "capitalize", letterSpacing: "0.02em",
    }}>
      {meta.label}
    </span>
  );
}

function Field({ label, value, mono, highlight }) {
  if (value == null || value === "") return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{
        fontSize: 10, fontWeight: 700, color: T.textTertiary,
        textTransform: "uppercase", letterSpacing: "0.07em",
      }}>
        {label}
      </div>
      <div style={{
        color: highlight ? T.accent : T.text,
        background: highlight ? "rgba(0,113,227,0.06)" : T.bg,
        padding: "9px 12px", borderRadius: 8,
        border: `1px solid ${highlight ? "rgba(0,113,227,0.15)" : T.border}`,
        fontFamily: mono ? "'SF Mono','Menlo','Monaco',monospace" : "inherit",
        fontSize: mono ? 11 : 13, lineHeight: 1.55, wordBreak: "break-word",
        fontStyle: highlight ? "italic" : "normal",
      }}>
        {value}
      </div>
    </div>
  );
}

export default function EvidenceDrawer({ item, onClose }) {
  const open = item != null;
  return (
    <>
      {/* Scrim */}
      <div
        onClick={onClose}
        style={{
          position: "fixed", inset: 0,
          background: open ? "rgba(0,0,0,0.18)" : "transparent",
          backdropFilter: open ? "blur(2px)" : "none",
          pointerEvents: open ? "auto" : "none",
          transition: "all 0.2s", zIndex: 100,
        }}
      />

      {/* Drawer panel */}
      <div style={{
        position: "fixed", top: 0, right: 0, bottom: 0, width: 360,
        background: T.surface,
        borderLeft: `1px solid ${T.border}`,
        boxShadow: open ? "-8px 0 24px rgba(0,0,0,0.08)" : "none",
        transform: open ? "translateX(0)" : "translateX(100%)",
        transition: "transform 0.25s cubic-bezier(0.4,0,0.2,1), box-shadow 0.25s",
        zIndex: 101, display: "flex", flexDirection: "column", overflow: "hidden",
      }}>
        {/* Header */}
        <div style={{
          padding: "16px 20px", borderBottom: `1px solid ${T.border}`,
          display: "flex", alignItems: "center", gap: 10, flexShrink: 0,
          background: T.bg,
        }}>
          <div style={{ flex: 1, fontSize: 13, fontWeight: 700, color: T.text, letterSpacing: "-0.01em" }}>
            Evidence Detail
          </div>
          <button
            onClick={onClose}
            style={{
              width: 26, height: 26, borderRadius: "50%",
              background: "rgba(0,0,0,0.06)", border: "none",
              color: T.textSecondary, fontSize: 13, cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "background 0.1s",
            }}
            onMouseEnter={e => e.currentTarget.style.background = "rgba(0,0,0,0.1)"}
            onMouseLeave={e => e.currentTarget.style.background = "rgba(0,0,0,0.06)"}
          >
            ✕
          </button>
        </div>

        {item ? (
          <div style={{ flex: 1, overflowY: "auto", padding: "18px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
            {/* Kind + ID */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <KindBadge kind={item.kind} />
              <span style={{
                fontSize: 10, fontWeight: 600, color: T.textTertiary,
                fontFamily: "'SF Mono','Menlo',monospace",
              }}>
                {item.id}
              </span>
            </div>

            <div style={{ height: 1, background: T.border }} />

            <Field label="Display" value={item.display} />
            {(item.value != null || item.unit) && (
              <Field label="Value" value={[item.value, item.unit].filter(Boolean).join(" ")} />
            )}
            <Field label="Date / Effective" value={item.effective} />
            <Field label="Source Reference" value={item.source_ref} />
            {item.text && (
              <Field label="Sentence from Note" value={`"${item.text}"`} highlight />
            )}
            {item.code && (
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{
                  fontSize: 10, fontWeight: 700, color: T.textTertiary,
                  textTransform: "uppercase", letterSpacing: "0.07em",
                }}>
                  Code
                </div>
                <pre style={{
                  fontSize: 11, fontFamily: "'SF Mono','Menlo',monospace",
                  color: T.textSecondary, background: T.bg,
                  padding: "10px 12px", borderRadius: 8,
                  border: `1px solid ${T.border}`,
                  overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-all",
                }}>
                  {JSON.stringify(item.code, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ) : (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: T.textTertiary, fontSize: 13 }}>
            No evidence selected
          </div>
        )}
      </div>
    </>
  );
}
