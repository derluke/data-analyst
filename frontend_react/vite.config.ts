import path from "path";
import { defineConfig, UserConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { nodePolyfills } from "vite-plugin-node-polyfills";
import { VITE_DEFAULT_PORT } from "./src/constants/dev";


let base: string = '';
// if NOTEBOOK_ID is set, use /notebook-sessions/${NOTEBOOK_ID}/ports/5173/ as the base
if (process.env.NOTEBOOK_ID) {
  const notebookId = process.env.NOTEBOOK_ID;
  base = `/notebook-sessions/${notebookId}/ports/${VITE_DEFAULT_PORT}/`;
}
const proxyBase: string = base === '' ? '/' : base;

export default defineConfig(({ command }) => {
  const config: UserConfig = {
    base,
    server: {
      host: true,
      allowedHosts: ["localhost", "127.0.0.1", ".datarobot.com"],
      proxy: {
        [`${proxyBase}api/`]: {
          target: 'http://localhost:8080',
          changeOrigin: true,
          rewrite: (path) => path.replace(new RegExp(`^${proxyBase}`), ''),
        }
      }
    },
    plugins: [
      react(),
      tailwindcss(),
      nodePolyfills({
        exclude: [],
        // for plotly.js
        protocolImports: true,
      }),
      {
        name: 'strip-base',
        apply: 'serve',
        configureServer({ middlewares }) {
          middlewares.use((req, _res, next) => {
            if (base !== '' && !req.url?.startsWith(base)) {
              req.url = base.slice(0, -1) + req.url;
            }
            next();
          });
        },
      },
    ],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
        "~": path.resolve(__dirname, "./src"),
      },
    },
  };

  // Add optimizations for production builds
  if (command === 'build') {
    config.build = {
      outDir: '../app_backend/static',
      emptyOutDir: true,
      // Exclude test files from the build
      rollupOptions: {
        external: [
          // Exclude test setup files
          /setupTests\.(cjs|js|ts)/,
          // Exclude test files and mocks
          /__tests__\//,
          /__mocks__\//,
          /\.test\.(js|ts|jsx|tsx)$/,
          /\.spec\.(js|ts|jsx|tsx)$/,
          /jest\.config\.cjs$/,
        ],
      },
    };
  }

  return config;
});
