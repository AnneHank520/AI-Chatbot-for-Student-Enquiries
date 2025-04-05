'use client'

import { Providers } from './providers'
import './globals.css'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <div className="min-h-screen flex flex-col">
            {/* 新的顶部导航栏 */}
            <header className="bg-[#183728] text-white p-4 shadow-md">
              <div className="container mx-auto">
                <h1 className="text-2xl font-bold">Rekro Management Backend</h1>
              </div>
            </header>
            
            {/* 主内容区域 */}
            <main className="flex-1 bg-gray-50">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  )
}
