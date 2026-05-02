import React from "react";

const T = {
  text: "#1d1d1f",
  textSecondary: "#6e6e73",
  textTertiary: "#aeaeb2",
  accent: "#0071e3",
  accentBg: "rgba(0,113,227,0.07)",
  border: "rgba(0,0,0,0.08)",
  surface: "#ffffff",
  bg: "#f5f5f7",
};

const SECTIONS = [
  { key: "hpi",             label: "History of Present Illness", icon: "📋" },
  { key: "active_problems", label: "Active Problems",            icon: "🔴" },
  { key: "medications",     label: "Medications",                icon: "💊" },
  { key: "labs",            label: "Labs",                       icon: "🧪" },
  { key: "vitals",          label: "Vitals",                     icon: "📊" },
  { key: "plan",            label: "Plan",                       icon: "📝" },
];

export default function FactSheet({ factSheet, onEvidenceClick }) {
  if (!factSheet) {
    return <div style={{ padding: 24, color: T.textTertiary, fontSize: 13 }}>Fact sheet not available.</div>;
  }
  const sections = factSheet.sections ?? {};

  return (
    <div style={{ padding: "18px 22px", display: "flex", flexDirection: "column", gap: 14, overflowY: "auto", height: "100%" }}>
      {SECTIONS.map(({ key, label, icon }) => {
        const items = sections[key] ?? [];
        return (
          <div key={key} style={{
            background: T.surface,
            border: `1px solid ${T.border}`,
            borderRadius: 12,
            overflow: "hidden",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          }}>
            {/* Section header */}
            <div style={{
              padding: "10px 16px",
              background: T.bg,
              borderBottom: items.length > 0 ? `1px solid ${T.border}` : "none",
              display: "flex", alignItems: "center", gap: 8,
            }}>
              <span style={{ fontSize: 14 }}>{icon}</span>
              <span style={{
                fontSize: 12, fontWeight: 700, color: T.text,
                letterSpacing: "-0.01em",
              }}>
                {label}
              </span>
              <span style={{
                marginLeft: "auto", fontSize: 10, color: T.textTertiary,
                background: "rgba(0,0,0,0.04)", padding: "1px 7px",
                borderRadius: 10, border: `1px solid ${T.border}`,
              }}>
                {items.length}
              </span>
            </div>

            {/* Items */}
            {items.length === 0 ? (
              <div style={{ padding: "10px 16px", fontSize: 12, color: T.textTertiary, fontStyle: "italic" }}>
                Not documented.
              </div>
            ) : (
              items.map((item, i) => (
                <div key={i} style={{
                  padding: "11px 16px",
                  borderBottom: i < items.length - 1 ? `1px solid ${T.border}` : "none",
                  display: "flex", flexDirection: "column", gap: 6,
                }}>
                  <div style={{ fontSize: 13, color: T.text, lineHeight: 1.5 }}>
                    {item.text}
                  </div>
                  {item.evidence_ids?.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {item.evidence_ids.map(eid => (
                        <button
                          key={eid}
                          onClick={() => onEvidenceClick?.(eid)}
                          title={`View evidence: ${eid}`}
                          style={{
                            fontSize: 10, fontWeight: 700,
                            padding: "2px 8px", borderRadius: 20,
                            background: T.accentBg, color: T.accent,
                            border: "1px solid rgba(0,113,227,0.2)",
                            cursor: "pointer", transition: "all 0.1s",
                          }}
                          onMouseEnter={e => { e.currentTarget.style.background = "rgba(0,113,227,0.14)"; }}
                          onMouseLeave={e => { e.currentTarget.style.background = T.accentBg; }}
                        >
                          {eid}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        );
      })}
    </div>
  );
}
