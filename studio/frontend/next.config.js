/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // BACKEND_URL is read server-side at RUNTIME in tRPC procedures (see server/backend.ts) — not inlined here,
  // so the same build works against any backend host/port set via the environment.
};
module.exports = nextConfig;
