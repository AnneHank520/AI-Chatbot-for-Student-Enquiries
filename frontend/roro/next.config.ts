import type { NextConfig } from "next";

// const nextConfig: NextConfig = {
//   /* config options here */
// };

/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    // Ignoring ESLint errors in a production environment
    ignoreDuringBuilds: true,
  },
};

module.exports = nextConfig;
// export default nextConfig;
