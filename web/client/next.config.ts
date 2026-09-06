import type { NextConfig } from "next";

// The FastAPI runs server (web/server/app.py) owns /api and /media.
const API = process.env.VESPER_API ?? "http://localhost:8777";

const nextConfig: NextConfig = {
  rewrites() {
    return Promise.resolve([
      { source: "/api/:path*", destination: `${API}/api/:path*` },
      { source: "/media/:path*", destination: `${API}/media/:path*` },
      { source: "/site/:path*", destination: `${API}/site/:path*` },
      { source: "/demo/:path*", destination: `${API}/demo/:path*` },
      { source: "/download/:path*", destination: `${API}/download/:path*` },
    ]);
  },
};

export default nextConfig;
