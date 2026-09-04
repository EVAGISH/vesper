import type { NextConfig } from "next";

// The FastAPI runs server (web/server/app.py) owns /api and /media.
const API = process.env.VESPER_API ?? "http://localhost:8777";

const nextConfig: NextConfig = {
  rewrites() {
    return Promise.resolve([
      { source: "/api/:path*", destination: `${API}/api/:path*` },
      { source: "/media/:path*", destination: `${API}/media/:path*` },
      { source: "/site/:path*", destination: `${API}/site/:path*` },
    ]);
  },
};

export default nextConfig;
