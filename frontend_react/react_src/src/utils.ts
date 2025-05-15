import { VITE_DEFAULT_PORT } from "./constants/dev";

export function isProd() {
  return import.meta.env.MODE === "production";
}
export function getBaseUrl() {
  // Truly local dev:
  let baseUrl = "";

  // Built Production
  if (isProd()) {
    // expected valid url https://app.datarobot.com/custom_applications/{appId}/api
    const fullUrl = window.location.origin + window.location.pathname;
    baseUrl = fullUrl.split("/").splice(0, 5).join("/");
  }

  // Running dev from Codespaces
  if (baseUrl === "") {
    // expected valid url https://<domain>/notebook-sessions/6824cb3764f338604977c9b8/ports/5173/
    const fullUrl = window.location.origin + window.location.pathname; 
    if (fullUrl.includes("notebook-sessions") && fullUrl.includes(`ports/${VITE_DEFAULT_PORT}`)) {
      baseUrl = fullUrl.endsWith("/") ? fullUrl.slice(0, -1) : fullUrl;
    }
  }
  return baseUrl;
}
