import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null };

/**
 * Last-resort guard: a render error used to unmount the whole React root and
 * leave a blank page with no way back.  Show the failure and offer a reload.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("MaiForge UI error", error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <div
        style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#0e0f13",
          color: "#f2eef4",
          fontFamily:
            "'Zen Maru Gothic', 'Noto Sans SC', system-ui, -apple-system, sans-serif",
          padding: 24,
        }}
      >
        <div style={{ maxWidth: 560, width: "100%" }}>
          <div
            style={{
              fontSize: 12,
              letterSpacing: "0.22em",
              color: "#ff8fbd",
              marginBottom: 12,
            }}
          >
            MAIFORGE
          </div>
          <h1 style={{ fontSize: 24, margin: "0 0 10px", lineHeight: 1.3 }}>
            页面渲染出错
          </h1>
          <p style={{ margin: "0 0 16px", color: "#b8b3c2", lineHeight: 1.7 }}>
            数据没有丢失，刷新页面即可继续。如果反复出现，请把下面的错误信息发给我。
          </p>
          <pre
            style={{
              margin: "0 0 20px",
              padding: "12px 14px",
              background: "#171922",
              border: "1px solid #2a2d3a",
              borderRadius: 6,
              color: "#ffb4c9",
              fontSize: 12.5,
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {this.state.error.message}
          </pre>
          <button
            type="button"
            onClick={() => window.location.reload()}
            style={{
              height: 38,
              padding: "0 20px",
              border: "none",
              borderRadius: 4,
              background: "#ff5f9e",
              color: "#fff",
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            重新加载
          </button>
        </div>
      </div>
    );
  }
}
