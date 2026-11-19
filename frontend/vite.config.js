import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, forward API calls to the FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/ask": "http://localhost:8000", "/health": "http://localhost:8000" } },
});
