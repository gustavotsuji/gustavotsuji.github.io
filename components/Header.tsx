import Link from 'next/link'
import HeaderClient from './HeaderClient'

export default function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800">
      <nav className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link
            href="/"
            className="text-2xl font-bold text-gray-900 dark:text-white hover:text-primary-600 transition-colors"
          >
            GT
          </Link>

          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/about"
              className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
            >
              About
            </Link>
            <Link
              href="/blog"
              className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
            >
              Blog
            </Link>
            <Link
              href="/#contact"
              className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
            >
              Contact
            </Link>
          </div>

          {/* Mobile menu controls hydrated client-side */}
          <HeaderClient />
        </div>
      </nav>
    </header>
  )
}
