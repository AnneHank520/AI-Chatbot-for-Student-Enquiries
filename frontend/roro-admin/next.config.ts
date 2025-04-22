import type { NextConfig } from "next";

// const nextConfig: NextConfig = {
//   /* config options here */
// };

/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {

    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://rekro-backend:5001/api/:path*',   // If you want to run this file locally, usehttp://localhost:5001/api/:path*
      },
    ];
  },
};

module.exports = nextConfig;
// export default nextConfig;
