'use client'
import React from 'react'

interface Language {
  code: string
  label: string
  short: string
}

interface LanguageSelectorProps {
  readonly selectedLang?: string
  // Handler that accepts language code and updates the language
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly onSelectLang?: any
}

const LANGUAGES: Language[] = [
  { code: 'pt', label: 'Português', short: 'PT' },
  { code: 'en', label: 'English', short: 'EN' },
  { code: 'es', label: 'Español', short: 'ES' },
  { code: 'ja', label: '日本語', short: 'JA' },
  { code: 'fr', label: 'Français', short: 'FR' },
]

export default function LanguageSelector({ selectedLang, onSelectLang }: LanguageSelectorProps) {
  if (!onSelectLang) return null

  return (
    <div className="overflow-x-auto pb-2 scrollbar-hide">
      <div className="flex gap-2 justify-start sm:justify-end min-w-max sm:min-w-0">
        {LANGUAGES.map((lang) => (
          <button
            key={lang.code}
            aria-pressed={selectedLang === lang.code}
            aria-label={lang.label}
            className={`px-3 py-2 rounded text-sm font-medium border transition-colors whitespace-nowrap flex-shrink-0 ${
              selectedLang === lang.code
                ? 'bg-primary-600 text-white border-primary-600'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-700 hover:bg-primary-100 dark:hover:bg-primary-900/30'
            }`}
            onClick={() => onSelectLang?.(lang.code)}
          >
            <span className="hidden sm:inline">{lang.label}</span>
            <span className="sm:hidden">{lang.short}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
