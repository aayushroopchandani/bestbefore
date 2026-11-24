import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { takeKey } from "./api";
import { App } from "./App";

takeKey(); // before anything renders: the key out of the address bar, into session storage

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
