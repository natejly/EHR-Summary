import React, { useState, useEffect, useCallback } from "react";
import CaseList from "./components/CaseList.jsx";
import PipelineStepper from "./components/PipelineStepper.jsx";
import ArtifactTabs from "./components/ArtifactTabs.jsx";
import EvidenceDrawer from "./components/EvidenceDrawer.jsx";

const T = {
  bg: "#f5f5f7",
  surface: "#ffffff",
  sidebar: "#f5f5f7",
  border: "rgba(0,0,0,0.08)",
  borderStrong: "rgba(0,0,0,0.12)",
  text: "#1d1d1f",
  textSecondary: "#6e6e73",
  textTertiary: "#aeaeb2",
  accent: "#0071e3",
  shadow: "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
};

const styles = {
  layout: {
    display: "flex",
    height: "100vh",
    overflow: "hidden",
    background: T.bg,
  },
  sidebar: {
    width: 260,
    minWidth: 220,
    borderRight: `1px solid ${T.border}`,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    background: T.sidebar,
  },
  sidebarHeader: {
    padding: "20px 16px 14px",
    borderBottom: `1px solid ${T.border}`,
    flexShrink: 0,
  },
  logo: {
    fontSize: 15,
    fontWeight: 700,
    color: T.text,
    letterSpacing: "-0.01em",
    display: "flex",
    alignItems: "center",
    gap: 7,
  },
  logoSub: {
    fontSize: 11,
    color: T.textTertiary,
    marginTop: 2,
    fontWeight: 400,
    letterSpacing: "0.01em",
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    background: T.bg,
  },
  topBar: {
    padding: "12px 22px",
    borderBottom: `1px solid ${T.border}`,
    background: T.surface,
    flexShrink: 0,
    display: "flex",
    alignItems: "center",
    gap: 14,
    boxShadow: T.shadow,
  },
  caseTitle: {
    fontSize: 13,
    fontWeight: 600,
    color: T.text,
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    letterSpacing: "-0.01em",
  },
  emptyState: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    color: T.textTertiary,
    background: T.bg,
  },
  emptyIcon: { fontSize: 44, opacity: 0.5 },
  emptyText: { fontSize: 14, color: T.textSecondary, fontWeight: 500 },
  emptyHint: { fontSize: 12, color: T.textTertiary },
  contentArea: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  loadingDot: {
    display: "inline-block",
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: T.accent,
    animation: "pulse 1.2s ease-in-out infinite",
  },
};

export default function App() {
  const [cases, setCases] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [artifacts, setArtifacts] = useState(null);
  const [loading, setLoading] = useState(false);
  const [runningCase, setRunningCase] = useState(null);
  const [runStages, setRunStages] = useState([]);
  const [evidenceItem, setEvidenceItem] = useState(null);

  const fetchCases = useCallback(async () => {
    try {
      const res = await fetch("/api/cases");
      setCases(await res.json());
    } catch (e) {
      console.error("Failed to fetch cases:", e);
    }
  }, []);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const loadCase = useCallback(async (caseId) => {
    setSelectedCase(caseId);
    setArtifacts(null);
    setEvidenceItem(null);
    setLoading(true);
    try {
      const res = await fetch(`/api/cases/${caseId}/artifacts`);
      if (!res.ok) throw new Error(await res.text());
      setArtifacts(await res.json());
    } catch (e) {
      console.error("Failed to load artifacts:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  const runCase = useCallback(async (caseId) => {
    if (runningCase) return;
    setRunningCase(caseId);
    setRunStages([]);
    setSelectedCase(caseId);
    setArtifacts(null);
    setEvidenceItem(null);
    try {
      const res = await fetch(`/api/run/${caseId}`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        alert(`Run failed: ${err.detail}`);
        setRunningCase(null);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = JSON.parse(line.slice(5).trim());
          if (payload.type === "stage_done") setRunStages((p) => [...p, payload.stage]);
          if (payload.type === "done") {
            setRunningCase(null);
            await fetchCases();
            await loadCase(caseId);
          }
        }
      }
    } catch (e) {
      console.error("SSE stream error:", e);
      setRunningCase(null);
    }
  }, [runningCase, fetchCases, loadCase]);

  const evidenceMap = React.useMemo(() => {
    const m = new Map();
    for (const ev of artifacts?.evidence_store?.evidence ?? []) m.set(ev.id, ev);
    return m;
  }, [artifacts]);

  const currentCaseMeta = cases.find((c) => c.case_id === selectedCase);

  return (
    <>
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes spin { to{transform:rotate(360deg)} }
      `}</style>
      <div style={styles.layout}>
        {/* Sidebar */}
        <div style={styles.sidebar}>
          <div style={styles.sidebarHeader}>
            <div style={styles.logo}>
              <span style={{ fontSize: 18 }}>⚕</span>
              EHR Pipeline
            </div>
            <div style={styles.logoSub}>Clinical Note Summarizer</div>
          </div>
          <CaseList
            cases={cases}
            selectedCase={selectedCase}
            runningCase={runningCase}
            onSelect={loadCase}
            onRun={runCase}
            onRefresh={fetchCases}
          />
        </div>

        {/* Main */}
        <div style={styles.main}>
          {selectedCase ? (
            <>
              <div style={styles.topBar}>
                <div style={styles.caseTitle}>{selectedCase}</div>
                {currentCaseMeta && (
                  <PipelineStepper
                    stagesPresent={
                      runningCase === selectedCase
                        ? runStages
                        : (artifacts?.stages_present ?? currentCaseMeta.stages_present)
                    }
                    isRunning={runningCase === selectedCase}
                  />
                )}
              </div>
              <div style={styles.contentArea}>
                {loading ? (
                  <div style={styles.emptyState}>
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      {[0, 1, 2].map((i) => (
                        <div
                          key={i}
                          style={{
                            ...styles.loadingDot,
                            animationDelay: `${i * 0.15}s`,
                          }}
                        />
                      ))}
                    </div>
                    <div style={{ fontSize: 13, color: "#6e6e73" }}>Loading artifacts…</div>
                  </div>
                ) : artifacts ? (
                <ArtifactTabs
                  key={selectedCase}
                  artifacts={artifacts}
                  evidenceMap={evidenceMap}
                  onCitationClick={(id) => setEvidenceItem(evidenceMap.get(id) ?? null)}
                />
                ) : runningCase === selectedCase ? (
                  <div style={styles.emptyState}>
                    <div style={{ fontSize: 13, color: "#6e6e73" }}>
                      Pipeline running — stages light up above as they complete.
                    </div>
                  </div>
                ) : null}
              </div>
            </>
          ) : (
            <div style={styles.emptyState}>
              <div style={styles.emptyIcon}>🩺</div>
              <div style={styles.emptyText}>Select a case to get started</div>
              <div style={styles.emptyHint}>Choose a MIMIC case from the sidebar</div>
            </div>
          )}
        </div>

        <EvidenceDrawer item={evidenceItem} onClose={() => setEvidenceItem(null)} />
      </div>
    </>
  );
}
