import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const T = {
  text: "#1d1d1f",
  textSecondary: "#6e6e73",
  textTertiary: "#aeaeb2",
  accent: "#0071e3",
  accentBg: "rgba(0,113,227,0.08)",
  green: "#34c759",
  greenBg: "rgba(52,199,89,0.1)",
  red: "#ff3b30",
  redBg: "rgba(255,59,48,0.08)",
  orange: "#ff9f0a",
  orangeBg: "rgba(255,159,10,0.1)",
  purple: "#af52de",
  purpleBg: "rgba(175,82,222,0.1)",
  border: "rgba(0,0,0,0.08)",
  surface: "#ffffff",
  bg: "#f5f5f7",
};

// ── Citation ID parsing ───────────────────────────────────────────────────────
// [E:cond:1]          → { label: "cond·1", isNote: false }
// [E:note:row318233:3]→ { label: "note·3", isNote: true  }
function parseCitationId(id) {
  const parts = id.split(":");
  if (parts.length < 3) return { label: id, isNote: false };
  const kind = parts[1];
  if (kind === "note") {
    return { label: `note·${parts[parts.length - 1]}`, isNote: true };
  }
  return { label: `${kind}·${parts[2]}`, isNote: false };
}

const CITATION_RE = /\[E:[^\]]+\]/g;

// ── Note Highlighter ──────────────────────────────────────────────────────────
function NoteHighlighter({ noteText, highlightText, activeId }) {
  const markRef = useRef(null);
  useEffect(() => {
    if (markRef.current) {
      markRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlightText, activeId]);

  if (!noteText) {
    return (
      <div style={{ padding: 24, color: T.textTertiary, fontSize: 13, fontStyle: "italic" }}>
        No input note available for this case.
      </div>
    );
  }

  const base = {
    fontFamily: "'SF Mono', 'Menlo', 'Monaco', monospace",
    fontSize: 12, lineHeight: 1.8, color: T.textSecondary,
    whiteSpace: "pre-wrap", wordBreak: "break-word",
    padding: "18px 20px", margin: 0,
  };

  if (!highlightText) return <pre style={base}>{noteText}</pre>;

  const idx = noteText.indexOf(highlightText);
  if (idx === -1) return <pre style={base}>{noteText}</pre>;

  return (
    <pre style={base}>
      {noteText.slice(0, idx)}
      <mark ref={markRef} style={{
        background: "rgba(255,159,10,0.25)",
        color: "#7c4a00",
        borderRadius: 4,
        padding: "1px 2px",
        fontWeight: 700,
        outline: "2px solid rgba(255,159,10,0.5)",
        animation: "noteFlash 0.4s ease-out",
      }}>
        {noteText.slice(idx, idx + highlightText.length)}
      </mark>
      {noteText.slice(idx + highlightText.length)}
    </pre>
  );
}

// ── Citation Chip ─────────────────────────────────────────────────────────────
function CitationChip({ id, isActive, isNote, label, onClick }) {
  const [hov, setHov] = useState(false);
  const active = isActive || hov;
  return (
    <button
      onClick={() => onClick(id)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      title={id}
      style={{
        display: "inline-block",
        padding: "0px 6px",
        marginLeft: 2, marginRight: 1,
        borderRadius: 4,
        fontSize: 10, fontWeight: 700,
        lineHeight: "17px",
        verticalAlign: "middle",
        whiteSpace: "nowrap",
        cursor: "pointer",
        border: "1px solid",
        transition: "all 0.12s",
        ...(isNote
          ? {
              background: active ? T.orangeBg : "rgba(255,159,10,0.06)",
              color: T.orange,
              borderColor: active ? "rgba(255,159,10,0.5)" : "rgba(255,159,10,0.2)",
              boxShadow: active ? "0 0 0 2px rgba(255,159,10,0.15)" : "none",
            }
          : {
              background: active ? T.accentBg : "rgba(0,113,227,0.05)",
              color: T.accent,
              borderColor: active ? "rgba(0,113,227,0.35)" : "rgba(0,113,227,0.15)",
              boxShadow: active ? "0 0 0 2px rgba(0,113,227,0.12)" : "none",
            }),
      }}
    >
      {label}
    </button>
  );
}

// ── Text node with citations injected ────────────────────────────────────────
function CitationText({ text, activeId, onCitationClick }) {
  const parts = [];
  let last = 0;
  const re = new RegExp(CITATION_RE.source, "g");
  let match;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const raw = match[0];
    const id = raw.slice(1, -1);
    const { label, isNote } = parseCitationId(id);
    parts.push(
      <CitationChip
        key={`${id}-${match.index}`}
        id={id} isActive={activeId === id}
        isNote={isNote} label={label}
        onClick={onCitationClick}
      />
    );
    last = match.index + raw.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}

function makeComponents(activeId, onCitationClick) {
  function walk(node, key) {
    if (typeof node === "string") {
      if (CITATION_RE.test(node))
        return <CitationText key={key} text={node} activeId={activeId} onCitationClick={onCitationClick} />;
      return node;
    }
    if (React.isValidElement(node)) {
      const kids = React.Children.map(node.props.children, (c, i) => walk(c, `${key}-${i}`));
      return React.cloneElement(node, { key }, kids);
    }
    return node;
  }
  return {
    p({ children }) { return <p>{React.Children.map(children, (c, i) => walk(c, i))}</p>; },
    li({ children }) { return <li>{React.Children.map(children, (c, i) => walk(c, i))}</li>; },
  };
}

// ── Main component ────────────────────────────────────────────────────────────
export default function SummaryViewer({
  summaryMd, summaryMeta, checkReport,
  note, evidenceMap, onFhirEvidenceClick,
}) {
  const [activeId, setActiveId] = useState(null);
  const [highlightText, setHighlightText] = useState(null);
  const [noteLabel, setNoteLabel] = useState(null);

  const handleCitationClick = useCallback((id) => {
    if (activeId === id) {
      setActiveId(null); setHighlightText(null); setNoteLabel(null);
      return;
    }
    setActiveId(id);
    const item = evidenceMap?.get(id);
    if (item?.kind === "note_sentence" && item.text) {
      setHighlightText(item.text);
      setNoteLabel(item.text.length > 55 ? item.text.slice(0, 55) + "…" : item.text);
    } else {
      setHighlightText(null); setNoteLabel(null);
      onFhirEvidenceClick?.(id);
    }
  }, [activeId, evidenceMap, onFhirEvidenceClick]);

  const components = useMemo(
    () => makeComponents(activeId, handleCitationClick),
    [activeId, handleCitationClick]
  );

  const ratio = summaryMeta?.achieved_ratio != null
    ? (summaryMeta.achieved_ratio * 100).toFixed(1) + "%" : null;
  const inBand = summaryMeta?.in_target_band;

  if (!summaryMd) {
    return <div style={{ padding: 24, color: T.textTertiary, fontSize: 13 }}>No summary available.</div>;
  }

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden", background: T.bg }}>
      <style>{`
        @keyframes noteFlash {
          0%   { outline-color: rgba(255,159,10,0.8); }
          100% { outline-color: rgba(255,159,10,0.5); }
        }
        .sum-prose h2 {
          font-size: 13px; font-weight: 700; letter-spacing: -0.01em;
          color: ${T.text}; margin: 18px 0 6px;
          padding-bottom: 5px; border-bottom: 1px solid ${T.border};
        }
        .sum-prose p  { margin-bottom: 9px; }
        .sum-prose ul, .sum-prose ol { padding-left: 18px; margin-bottom: 9px; }
        .sum-prose li { margin-bottom: 3px; }
        .sum-prose em { color: ${T.textTertiary}; }
        .sum-prose strong { color: ${T.text}; font-weight: 600; }
      `}</style>

      {/* ── Left: Summary ─────────────────────────────── */}
      <div style={{
        flex: "0 0 56%", borderRight: `1px solid ${T.border}`,
        display: "flex", flexDirection: "column", overflow: "hidden",
        background: T.surface,
      }}>
        {/* Meta strip */}
        <div style={{
          display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center",
          padding: "10px 18px", borderBottom: `1px solid ${T.border}`,
          background: T.bg, flexShrink: 0,
        }}>
          {ratio && <Stat label="Compression" value={ratio} />}
          {summaryMeta?.note_chars != null && (
            <Stat label="Note" value={`${summaryMeta.note_chars.toLocaleString()} ch`} />
          )}
          {summaryMeta?.summary_chars != null && (
            <Stat label="Summary" value={`${summaryMeta.summary_chars.toLocaleString()} ch`} />
          )}
          {inBand != null && <Pill ok={inBand} yes="In band" no="Out of band" />}
          {checkReport && (
            <Pill
              ok={checkReport.passed}
              yes="Check ✓"
              no={`${checkReport.violations?.length ?? 0} violation(s)`}
            />
          )}
          <div style={{ marginLeft: "auto", fontSize: 10, color: T.textTertiary, display: "flex", gap: 10 }}>
            <span><span style={{ color: T.orange, fontWeight: 700 }}>note·n</span> = note sentence</span>
            <span><span style={{ color: T.accent, fontWeight: 700 }}>kind·n</span> = FHIR record</span>
          </div>
        </div>

        {/* Summary prose */}
        <div style={{ flex: 1, overflowY: "auto", padding: "18px 22px" }}>
          <div className="sum-prose" style={{ fontSize: 14, lineHeight: 1.8, color: T.textSecondary }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
              {summaryMd}
            </ReactMarkdown>
          </div>
        </div>
      </div>

      {/* ── Right: Original Note ───────────────────────── */}
      <div style={{
        flex: "0 0 44%", display: "flex", flexDirection: "column",
        overflow: "hidden", background: T.surface,
      }}>
        {/* Header */}
        <div style={{
          padding: "10px 18px", borderBottom: `1px solid ${T.border}`,
          background: T.bg, flexShrink: 0,
          display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap",
        }}>
          <span style={{
            fontSize: 10, fontWeight: 700, color: T.textTertiary,
            textTransform: "uppercase", letterSpacing: "0.07em",
          }}>
            Original Note
          </span>
          {noteLabel && (
            <span style={{
              fontSize: 10, color: T.orange,
              background: T.orangeBg,
              border: "1px solid rgba(255,159,10,0.25)",
              borderRadius: 5, padding: "2px 8px",
              maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }} title={highlightText}>
              ↳ "{noteLabel}"
            </span>
          )}
          {activeId && !noteLabel && (
            <span style={{ fontSize: 10, color: T.textTertiary, fontStyle: "italic" }}>
              FHIR record — see drawer →
            </span>
          )}
          {activeId && (
            <button
              onClick={() => { setActiveId(null); setHighlightText(null); setNoteLabel(null); }}
              style={{
                marginLeft: "auto", fontSize: 11, color: T.textTertiary,
                background: "none", border: "none", cursor: "pointer",
                padding: "2px 6px", borderRadius: 5,
              }}
            >
              ✕
            </button>
          )}
        </div>

        {/* Note text */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          <NoteHighlighter noteText={note} highlightText={highlightText} activeId={activeId} />
        </div>
      </div>
    </div>
  );
}

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

function Pill({ ok, yes, no }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: "3px 9px", borderRadius: 20,
      background: ok ? T.greenBg : T.redBg,
      color: ok ? T.green : T.red,
      border: `1px solid ${ok ? "rgba(52,199,89,0.3)" : "rgba(255,59,48,0.25)"}`,
    }}>
      {ok ? yes : no}
    </span>
  );
}
