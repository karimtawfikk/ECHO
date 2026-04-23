import axios from "axios";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010/api/v1";

export const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: false,
});