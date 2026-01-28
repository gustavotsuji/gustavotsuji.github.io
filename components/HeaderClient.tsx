'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

export default function HeaderClient() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  useEffect(() => {
    const el = document.querySelector('header') as HTMLElement | null
    if (!el) return
    const setVar = () => {
      const height = `${el.getBoundingClientRect().height}px`
      document.documentElement.style.setProperty('--site-header-height', height)
    }
    setVar()
    window.addEventListener('resize', setVar)
    return () => window.removeEventListener('resize', setVar)
  }, [isMenuOpen])

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        className="md:hidden p-2 text-gray-700 dark:text-gray-300"
        aria-label="Toggle menu"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {isMenuOpen ? (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          ) : (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          )}
        </svg>
      </button>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden mt-4 pb-4 space-y-4">
          <Link
            href="/about"
            className="block text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
          >
            About
          </Link>
          <Link
            href="/blog"
            className="block text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
          >
            Blog
          </Link>
          <Link
            href="/#contact"
            className="block text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
          >
            Contact
          </Link>
        </div>
      )}
    </>
  )
}
