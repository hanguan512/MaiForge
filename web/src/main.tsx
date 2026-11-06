import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "@fontsource/zcool-kuaile/latin-400.css";
import "@fontsource/zcool-kuaile/chinese-simplified-400.css";
import "@fontsource/zen-maru-gothic/latin-700.css";
import "@fontsource/zen-maru-gothic/latin-900.css";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </BrowserRouter>
  </React.StrictMode>,
);
