import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Script from 'next/script'
import './globals.css'
import Header from '@/components/Header'
import Footer from '@/components/Footer'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Gustavo Tsuji - Senior Software Engineer',
  description:
    'Experienced Developer (18+ years) with a double major in CS and Administration from USP. Backend scalability, cloud efficiency, and mentoring.',
  authors: [{ name: 'Gustavo Kendi Tsuji' }],
  keywords: [
    'software engineer',
    'backend developer',
    'cloud architecture',
    'nodejs',
    'java',
    'aws',
  ],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} bg-gray-50`}>
        {/* Google Analytics */}
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-D5Y6ZMWLR4"
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-D5Y6ZMWLR4');
          `}
        </Script>
        <Header />
        <main className="min-h-screen bg-gray-50">{children}</main>
        <Footer />
      </body>
    </html>
  )
}
