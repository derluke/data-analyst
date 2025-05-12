import { getBaseUrl, isProd } from "@/utils";
import axios, { CreateAxiosDefaults } from "axios";

type ConfigDefaults = {
  baseURL: string;
  headers: {
    Accept: string;
    "Content-type": string;
    "x-user-email"?: string;
  };
  withCredentials: boolean;
};

const getAxiosConfig = (): CreateAxiosDefaults<ConfigDefaults> => {
  const config: ConfigDefaults = {
    baseURL: `${getBaseUrl()}/api`,
    headers: {
      Accept: "application/json",
      "Content-type": "application/json",
    },
    withCredentials: true,
  };
  if (!isProd()) {
    config.headers["x-user-email"] = "user@domain.com";
  }
  return config;
};

const apiClient = axios.create(getAxiosConfig());

export default apiClient;

const drClient = axios.create({
  baseURL: `${window.location.origin}/api/v2`,
  headers: {
    Accept: "application/json",
    "Content-type": "application/json",
  },
  withCredentials: true,
});

export { drClient, apiClient };
