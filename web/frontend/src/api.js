/**
 * API base URL for FastAPI backend.
 *
 * Development (Vite proxy): leave VITE_API_BASE_URL empty and use relative paths.
 * Development (direct):       VITE_API_BASE_URL=http://127.0.0.1:8000
 * Production build:             VITE_API_BASE_URL=<deployed backend HTTPS URL>
 */
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export function apiUrl(path) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return API_BASE_URL ? `${API_BASE_URL}${normalizedPath}` : normalizedPath;
}

export { API_BASE_URL };
