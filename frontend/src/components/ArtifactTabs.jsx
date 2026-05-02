import React, { useState } from "react";
import SummaryViewer from "./SummaryViewer.jsx";
import FactSheet from "./FactSheet.jsx";

const T = {
  text: "#1d1d1f",
  textSecondary: "#6e6e73",
  textTertiary: "#aeaeb2",
  accent: "#0071e3",
  accentBg: "rgba(0,113,227,0.08)",
  green: "#34c759",
  greenBg: "rgba(52,199,89,0.09)",
  red: "#ff3b30",
  redBg: "rgba(255,59,48,0.08)",
  orange: "#ff9f0a",
  orangeBg: "rgba(255,159,10,0.09)",
  purple: "#af52de",
  border: "rgba(0,0,0,0.08)",
  borderStrong: "rgba(0,0,0,0.12)",
  surface: "#ffffff",
  bg: "#f5f5f7",
  shadow: "0 1px 3px rgba(0,0,0,0.05)",
};

// ── Shared helpers ────────────────────────────────────────────────────────────

function NoData({ label }) {
  return (
    <div style={{ padding: 24, color: T.textTertiary, fontSize: 13, fontStyle: "italic" }}>
      {label} not available for this case.
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 12, overflow: "hidden",
      boxShadow: T.shadow, ...style,
    }}>
      {children}
    </div>
  );
}

function SectionHeader({ children }) {
  return (
    <div style={{
      padding: "9px 14px", background: T.bg,
      borderBottom: `1px solid ${T.border}`,
      fontSize: 10, fontWeight: 700, color: T.textTertiary,
      textTransform: "uppercase", letterSpacing: "0.07em",
    }}>
      {children}
    </div>
  );
}

function MonoChip({ children, color }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: "1px 6px", borderRadius: 5,
      background: color ? `${color}12` : "rgba(0,0,0,0.05)",
      color: color ?? T.textSecondary,
      border: `1px solid ${color ? `${color}30` : "rgba(0,0,0,0.08)"}`,
      fontFamily: "monospace",
    }}>
      {children}
    </span>
  );
}

function StatusBadge({ status }) {
  const map = {
    verified:     { color: T.green,  bg: T.greenBg,  label: "Verified" },
    contradicted: { color: T.red,    bg: T.redBg,    label: "Contradicted" },
    unsupported:  { color: T.textTertiary, bg: "rgba(0,0,0,0.05)", label: "Unsupported" },
  };
  const m = map[status] ?? map.unsupported;
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: "2px 9px", borderRadius: 20,
      background: m.bg, color: m.color,
      border: `1px solid ${m.color}30`, textTransform: "capitalize",
    }}>
      {m.label}
    </span>
  );
}

function EvidencePill({ eid, onClick }) {
  return (
    <button onClick={() => onClick(eid)} title={`View evidence: ${eid}`} style={{
      fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 20,
      background: T.accentBg, color: T.accent,
      border: "1px solid rgba(0,113,227,0.2)",
      cursor: "pointer", transition: "background 0.1s",
    }}
    onMouseEnter={e => e.currentTarget.style.background = "rgba(0,113,227,0.14)"}
    onMouseLeave={e => e.currentTarget.style.background = T.accentBg}
    >
      {eid}
    </button>
  );
}

// ── Tab content components ────────────────────────────────────────────────────

function NoteTab({ note }) {
  if (!note) return <NoData label="Input note" />;
  return (
    <pre style={{
      fontFamily: "'SF Mono','Menlo','Monaco',monospace",
      fontSize: 12, lineHeight: 1.8, color: T.textSecondary,
      whiteSpace: "pre-wrap", wordBreak: "break-word",
      padding: "18px 22px", margin: 0,
    }}>
      {note}
    </pre>
  );
}

function ClaimsTab({ claims }) {
  const items = claims?.claims ?? [];
  if (!items.length) return <NoData label="Claims" />;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {items.map(c => (
        <Card key={c.claim_id}>
          <div style={{ padding: "12px 16px", display: "flex", flexDirection: "column", gap: 6 }}>
            <div style={{ display: "flex", gap: 7, alignItems: "center", flexWrap: "wrap" }}>
              <MonoChip>{c.claim_id}</MonoChip>
              <MonoChip color={T.accent}>{c.type}</MonoChip>
            </div>
            <div style={{ fontSize: 13, color: T.text, lineHeight: 1.5 }}>
              <strong style={{ fontWeight: 600 }}>{c.subject}</strong>
              {" "}<span style={{ color: T.textTertiary }}>{c.predicate}</span>{" "}
              <strong style={{ fontWeight: 600 }}>{c.value}</strong>
            </div>
            {c.time_ref && (
              <div style={{ fontSize: 11, color: T.textTertiary }}>🕐 {c.time_ref}</div>
            )}
            {c.source_span && (
              <div style={{
                fontSize: 11, color: T.accent, fontStyle: "italic",
                borderLeft: `2px solid rgba(0,113,227,0.3)`,
                paddingLeft: 8, background: T.accentBg,
                borderRadius: "0 5px 5px 0", padding: "5px 8px",
              }}>
                "{c.source_span}"
              </div>
            )}
          </div>
        </Card>
      ))}
    </div>
  );
}

function VerificationsTab({ verifications, claims, onEvidenceClick }) {
  const claimsMap = {};
  (claims?.claims ?? []).forEach(c => { claimsMap[c.claim_id] = c; });
  const items = verifications?.verifications ?? [];
  if (!items.length) return <NoData label="Verifications" />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {items.map(v => {
        const claim = claimsMap[v.claim_id];
        return (
          <Card key={v.claim_id}>
            <div style={{ padding: "12px 16px", display: "flex", flexDirection: "column", gap: 7 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <StatusBadge status={v.status} />
                <MonoChip>{v.claim_id}</MonoChip>
              </div>
              {claim && (
                <div style={{ fontSize: 13, color: T.text }}>
                  <strong style={{ fontWeight: 600 }}>{claim.subject}</strong>
                  {" "}<span style={{ color: T.textTertiary }}>{claim.predicate}</span>{" "}
                  <strong style={{ fontWeight: 600 }}>{claim.value}</strong>
                </div>
              )}
              {v.rationale && (
                <div style={{ fontSize: 12, color: T.textSecondary, lineHeight: 1.55 }}>
                  {v.rationale}
                </div>
              )}
              {v.evidence_ids?.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {v.evidence_ids.map(eid => (
                    <EvidencePill key={eid} eid={eid} onClick={onEvidenceClick} />
                  ))}
                </div>
              )}
            </div>
          </Card>
        );
      })}
    </div>
  );
}

function ContextTab({ context }) {
  if (!context) return <NoData label="Context" />;
  const missing = context.missing_context ?? [];
  const contradictions = context.contradictions ?? [];
  const suggested = context.suggested_supporting_facts ?? [];
  if (!missing.length && !contradictions.length && !suggested.length) {
    return (
      <div style={{ padding: 20, fontSize: 13, color: T.textTertiary }}>
        Context agent produced no findings (may have been disabled).
      </div>
    );
  }

  function Section({ title, color, items }) {
    if (!items.length) return null;
    return (
      <div style={{ marginBottom: 18 }}>
        <div style={{
          fontSize: 11, fontWeight: 700, color: color ?? T.accent,
          textTransform: "uppercase", letterSpacing: "0.07em",
          marginBottom: 8,
        }}>
          {title} ({items.length})
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {items.map((item, i) => (
            <div key={i} style={{
              fontSize: 13, color: T.text, lineHeight: 1.5,
              background: T.surface, border: `1px solid ${T.border}`,
              borderRadius: 8, padding: "9px 13px",
              boxShadow: T.shadow,
            }}>
              {typeof item === "string" ? item : (
                <>
                  {item.description && <div style={{ fontWeight: 500 }}>{item.description}</div>}
                  {item.suggested_claim && (
                    <div style={{ fontSize: 11, color: T.textTertiary, fontFamily: "monospace", marginTop: 4 }}>
                      {JSON.stringify(item.suggested_claim)}
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <>
      <Section title="Missing Context" color={T.accent} items={missing} />
      <Section title="Contradictions" color={T.red} items={contradictions} />
      <Section title="Suggested Supporting Facts" color={T.green} items={suggested} />
    </>
  );
}

function CheckTab({ checkReport }) {
  if (!checkReport) return <NoData label="Check report" />;
  const { passed, violations = [], cited_evidence_ids = [], sentence_count = 0 } = checkReport;
  return (
    <>
      {/* Banner */}
      <div style={{
        padding: "14px 18px", borderRadius: 12, marginBottom: 18,
        background: passed ? T.greenBg : T.redBg,
        border: `1px solid ${passed ? "rgba(52,199,89,0.25)" : "rgba(255,59,48,0.2)"}`,
        display: "flex", alignItems: "center", gap: 14,
        boxShadow: T.shadow,
      }}>
        <span style={{ fontSize: 22 }}>{passed ? "✅" : "❌"}</span>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: passed ? T.green : T.red }}>
            {passed ? "Deterministic Check Passed" : "Deterministic Check Failed"}
          </div>
          <div style={{ fontSize: 11, color: T.textTertiary, marginTop: 2 }}>
            {sentence_count} sentence(s) · {violations.length} violation(s) · {cited_evidence_ids.length} cited IDs
          </div>
        </div>
      </div>

      {violations.length > 0 && (
        <>
          <div style={{ fontSize: 11, fontWeight: 700, color: T.red, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.07em" }}>
            Violations
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {violations.map((v, i) => (
              <Card key={i}>
                <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 5 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: T.red }}>{v.rule}</div>
                  <div style={{ fontSize: 12, color: T.textSecondary, fontStyle: "italic", borderLeft: `2px solid ${T.border}`, paddingLeft: 8 }}>
                    "{v.sentence}"
                  </div>
                  <div style={{ fontSize: 12, color: T.text }}>{v.detail}</div>
                </div>
              </Card>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function ReviewTab({ review }) {
  if (!review) return <NoData label="Review" />;
  const concerns = review.concerns ?? [];
  const revisions = review.recommended_revisions ?? [];

  if (!concerns.length && !revisions.length) {
    return (
      <div style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "14px 18px", borderRadius: 12,
        background: T.greenBg, border: "1px solid rgba(52,199,89,0.25)",
        fontSize: 13, color: T.green, fontWeight: 600,
      }}>
        ✓ No concerns raised by LLM review.
      </div>
    );
  }

  const sevColor = { high: T.red, medium: T.orange, low: T.accent };

  return (
    <>
      {concerns.length > 0 && (
        <>
          <div style={{ fontSize: 11, fontWeight: 700, color: T.textTertiary, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 10 }}>
            Concerns ({concerns.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 22 }}>
            {concerns.map((c, i) => {
              const col = sevColor[c.severity] ?? T.accent;
              return (
                <Card key={i}>
                  <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 6 }}>
                    <span style={{
                      display: "inline-block", alignSelf: "flex-start",
                      fontSize: 10, fontWeight: 700, padding: "2px 9px", borderRadius: 20,
                      background: `${col}12`, color: col, border: `1px solid ${col}30`,
                      textTransform: "capitalize",
                    }}>
                      {c.severity}
                    </span>
                    {c.sentence && (
                      <div style={{ fontSize: 12, color: T.textSecondary, fontStyle: "italic", borderLeft: `2px solid ${T.border}`, paddingLeft: 8 }}>
                        "{c.sentence}"
                      </div>
                    )}
                    <div style={{ fontSize: 13, color: T.text }}>{c.reason}</div>
                  </div>
                </Card>
              );
            })}
          </div>
        </>
      )}

      {revisions.length > 0 && (
        <>
          <div style={{ fontSize: 11, fontWeight: 700, color: T.textTertiary, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 10 }}>
            Recommended Revisions ({revisions.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {revisions.map((r, i) => (
              <Card key={i}>
                <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 9, fontWeight: 700, color: T.red, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 3 }}>Original</div>
                    <div style={{ fontSize: 12, color: T.textSecondary, textDecoration: "line-through", fontStyle: "italic", opacity: 0.7 }}>{r.original}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, fontWeight: 700, color: T.green, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 3 }}>Suggested</div>
                    <div style={{ fontSize: 12, color: T.text, fontStyle: "italic" }}>{r.suggested}</div>
                  </div>
                  {r.reason && <div style={{ fontSize: 11, color: T.textTertiary }}>{r.reason}</div>}
                </div>
              </Card>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function PatientSummaryView({ summaryMd, meta, note, evidenceMap, onFhirEvidenceClick }) {
  if (!summaryMd) return <NoData label="Patient summary" />;

  // Adapt patient_summary_meta → SummaryViewer's meta + checkReport-shape so
  // we get the same FK-grade stat strip and citation pane behaviour.
  const adaptedMeta = meta
    ? {
        // Reuse note_chars / summary_chars slots so the stat strip lights up.
        note_chars: meta.fk_words ?? 0,            // words (display purpose)
        summary_chars: meta.chars ?? summaryMd.length,
        achieved_ratio: null,                       // hide compression
        in_target_band: null,
      }
    : null;

  // Synthesize a "check report"-style pill that reports FK grade band instead.
  const fkPill = meta
    ? {
        passed: meta.in_target_band,
        violations: meta.in_target_band
          ? []
          : [{ rule: "fk_out_of_band" }],
      }
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden", background: T.bg }}>
      {/* Meta strip dedicated to patient summary */}
      {meta && (
        <div style={{
          display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center",
          padding: "10px 22px", borderBottom: `1px solid ${T.border}`,
          background: T.surface, flexShrink: 0,
        }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <span style={{ fontSize: 9, fontWeight: 700, color: T.textTertiary, textTransform: "uppercase", letterSpacing: "0.07em" }}>
              Flesch-Kincaid Grade
            </span>
            <span style={{ fontSize: 14, fontWeight: 700, color: meta.in_target_band ? T.green : T.orange }}>
              {Number(meta.fk_grade).toFixed(2)}
              <span style={{ fontSize: 10, color: T.textTertiary, fontWeight: 500, marginLeft: 6 }}>
                target {meta.target_grade_min}-{meta.target_grade_max}
              </span>
            </span>
          </div>
          <Stat label="Words" value={meta.fk_words ?? 0} />
          <Stat label="Sentences" value={meta.fk_sentences ?? 0} />
          <Stat label="Characters" value={meta.chars?.toLocaleString() ?? 0} />
          <span style={{
            fontSize: 11, fontWeight: 700, padding: "3px 9px", borderRadius: 20,
            background: meta.in_target_band ? T.greenBg : T.orangeBg,
            color: meta.in_target_band ? T.green : T.orange,
            border: `1px solid ${meta.in_target_band ? "rgba(52,199,89,0.3)" : "rgba(255,159,10,0.3)"}`,
          }}>
            {meta.in_target_band ? "✓ Reading level on target" : "⚠ Outside target band"}
          </span>
        </div>
      )}

      {/* Reuse SummaryViewer for the actual rendering */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        <SummaryViewer
          summaryMd={summaryMd}
          summaryMeta={adaptedMeta}
          checkReport={fkPill}
          note={note}
          evidenceMap={evidenceMap}
          onFhirEvidenceClick={onFhirEvidenceClick}
        />
      </div>
    </div>
  );
}

// Inline Stat helper (local copy so we don't pull T into SummaryViewer twice)
function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      <span style={{ fontSize: 9, fontWeight: 700, color: T.textTertiary, textTransform: "uppercase", letterSpacing: "0.07em" }}>
        {label}
      </span>
      <span style={{ fontSize: 12, fontWeight: 700, color: T.text }}>{value}</span>
    </div>
  );
}

function AuditTab({ audit }) {
  if (!audit) return <NoData label="Audit" />;
  const timings = audit.timings ?? [];
  const models = audit.models ?? {};
  const compression = audit.compression ?? {};

  const stats = [
    { label: "Status", value: audit.stage_failed ? `Failed: ${audit.stage_failed}` : "Success", ok: !audit.stage_failed },
    { label: "Context Agent", value: audit.context_agent_enabled ? "Enabled" : "Disabled", ok: audit.context_agent_enabled },
    compression.achieved_ratio != null && {
      label: "Compression",
      value: `${(compression.achieved_ratio * 100).toFixed(1)}% ${compression.in_target_band ? "✓" : "✗"}`,
      ok: compression.in_target_band,
    },
    audit.review_passed != null && { label: "Review", value: audit.review_passed ? "Passed" : "Failed", ok: audit.review_passed },
  ].filter(Boolean);

  return (
    <>
      {/* Stats grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 20 }}>
        {stats.map(stat => (
          <Card key={stat.label}>
            <div style={{ padding: "12px 14px" }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: T.textTertiary, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 4 }}>
                {stat.label}
              </div>
              <div style={{ fontSize: 13, fontWeight: 700, color: stat.ok ? T.green : T.red }}>
                {stat.value}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Models */}
      {Object.keys(models).length > 0 && (
        <Card style={{ marginBottom: 16 }}>
          <SectionHeader>Models Used</SectionHeader>
          <div style={{ padding: "4px 0" }}>
            {Object.entries(models).map(([role, model]) => (
              <div key={role} style={{
                display: "flex", gap: 12, padding: "8px 14px",
                borderBottom: `1px solid ${T.border}`, fontSize: 12,
              }}>
                <span style={{ color: T.textTertiary, minWidth: 150 }}>{role}</span>
                <span style={{ color: T.text, fontFamily: "monospace", fontWeight: 500 }}>{model}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Timings */}
      {timings.length > 0 && (
        <Card>
          <SectionHeader>Stage Timings</SectionHeader>
          <div>
            {timings.map((t, i) => (
              <div key={t.name} style={{
                display: "flex", justifyContent: "space-between",
                padding: "8px 14px",
                borderBottom: i < timings.length - 1 ? `1px solid ${T.border}` : "none",
                fontSize: 12,
              }}>
                <span style={{ color: T.textSecondary }}>{t.name}</span>
                <span style={{ color: t.skipped ? T.textTertiary : T.text, fontFamily: "monospace" }}>
                  {t.skipped ? "cached" : `${t.seconds?.toFixed(2)}s`}
                </span>
              </div>
            ))}
            <div style={{ display: "flex", justifyContent: "space-between", padding: "9px 14px", fontSize: 12, fontWeight: 700, borderTop: `1px solid ${T.borderStrong}`, background: T.bg }}>
              <span style={{ color: T.text }}>Total</span>
              <span style={{ color: T.accent, fontFamily: "monospace" }}>
                {timings.filter(t => !t.skipped).reduce((s, t) => s + (t.seconds ?? 0), 0).toFixed(2)}s
              </span>
            </div>
          </div>
        </Card>
      )}
    </>
  );
}

// ── Tab definitions ───────────────────────────────────────────────────────────

const TABS = [
  { id: "note",            label: "Note",            stageKey: null },
  { id: "claims",          label: "Claims",          stageKey: "stage2_extract" },
  { id: "verifications",   label: "Verifications",   stageKey: "stage3_verify" },
  { id: "context",         label: "Context",         stageKey: "stage4_context" },
  { id: "factsheet",       label: "Fact Sheet",      stageKey: "stage5_fact_sheet" },
  { id: "summary",         label: "Summary",         stageKey: "stage6_summarize" },
  { id: "check",           label: "Check",           stageKey: "stage7_check" },
  { id: "review",          label: "Review",          stageKey: "stage8_review" },
  { id: "patient",         label: "Patient Summary", stageKey: "stage9_patient_summary" },
  { id: "audit",           label: "Audit",           stageKey: null },
];

// ── Main component ────────────────────────────────────────────────────────────

export default function ArtifactTabs({ artifacts, evidenceMap, onCitationClick }) {
  const [activeTab, setActiveTab] = useState("summary");
  const stages = artifacts?.stages_present ?? [];

  const isAvailable = tab => !tab.stageKey || stages.includes(tab.stageKey);

  const handleEvidenceClick = eid => onCitationClick(eid);


  const renderContent = () => {
    switch (activeTab) {
      case "note":          return <NoteTab note={artifacts?.note} />;
      case "claims":        return <ClaimsTab claims={artifacts?.claims} />;
      case "verifications": return (
        <VerificationsTab
          verifications={artifacts?.verifications_augmented ?? artifacts?.verifications}
          claims={artifacts?.claims}
          onEvidenceClick={handleEvidenceClick}
        />
      );
      case "context":  return <ContextTab context={artifacts?.context} />;
      case "factsheet": return (
        <FactSheet factSheet={artifacts?.fact_sheet} onEvidenceClick={handleEvidenceClick} />
      );
      case "summary": return (
        <SummaryViewer
          summaryMd={artifacts?.summary_revised_md ?? artifacts?.summary_md}
          summaryMeta={artifacts?.summary_meta}
          checkReport={artifacts?.check_report}
          note={artifacts?.note}
          evidenceMap={evidenceMap}
          onFhirEvidenceClick={handleEvidenceClick}
        />
      );
      case "check":  return <CheckTab checkReport={artifacts?.check_report_revised ?? artifacts?.check_report} />;
      case "review": return <ReviewTab review={artifacts?.review} />;
      case "patient": return (
        <PatientSummaryView
          summaryMd={artifacts?.patient_summary_md}
          meta={artifacts?.patient_summary_meta}
          note={artifacts?.note}
          evidenceMap={evidenceMap}
          onFhirEvidenceClick={handleEvidenceClick}
        />
      );
      case "audit":  return <AuditTab audit={artifacts?.audit} />;
      default: return null;
    }
  };

  const needsOwnScroll = activeTab === "summary" || activeTab === "factsheet" || activeTab === "patient";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden", background: T.bg }}>
      {/* Tab bar */}
      <div style={{
        display: "flex", alignItems: "flex-end", gap: 0,
        padding: "0 22px",
        borderBottom: `1px solid ${T.border}`,
        background: T.surface,
        overflowX: "auto", flexShrink: 0,
        boxShadow: "0 1px 0 rgba(0,0,0,0.04)",
      }}>
        {TABS.map(tab => {
          const avail = isAvailable(tab);
          const active = activeTab === tab.id;
          const done = tab.stageKey && stages.includes(tab.stageKey);
          return (
            <button
              key={tab.id}
              onClick={() => avail && setActiveTab(tab.id)}
              disabled={!avail}
              title={!avail ? `${tab.label} not yet generated` : undefined}
              style={{
                padding: "11px 13px 10px",
                fontSize: 12, fontWeight: active ? 700 : 500,
                color: active ? T.accent : avail ? T.textSecondary : T.textTertiary,
                cursor: avail ? "pointer" : "not-allowed",
                background: "none", border: "none",
                borderBottom: active ? `2px solid ${T.accent}` : "2px solid transparent",
                whiteSpace: "nowrap",
                transition: "color 0.12s",
                userSelect: "none",
                display: "flex", alignItems: "center", gap: 5,
                opacity: !avail ? 0.45 : 1,
              }}
            >
              {tab.label}
              {done && (
                <span style={{
                  width: 5, height: 5, borderRadius: "50%",
                  background: active ? T.accent : T.green,
                  flexShrink: 0,
                }} />
              )}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: needsOwnScroll ? "hidden" : "auto", display: "flex", flexDirection: "column" }}>
        {needsOwnScroll
          ? renderContent()
          : <div style={{ padding: "18px 22px" }}>{renderContent()}</div>
        }
      </div>
    </div>
  );
}
