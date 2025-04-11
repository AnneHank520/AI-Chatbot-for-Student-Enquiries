import type { NextConfig } from "next";

// const nextConfig: NextConfig = {
//   /* config options here */
// };

/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    // 在生产环境下忽略 ESLint 错误
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://rekro-backend:5001/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
// export default nextConfig;
