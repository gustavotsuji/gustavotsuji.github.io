const nextJest = require('next/jest')

const createJestConfig = nextJest({
  // Provide the path to your Next.js app to load next.config.js and .env files in your test environment
  dir: './',
})

// Add any custom config to be passed to Jest
const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  testEnvironment: 'jest-environment-jsdom',
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
    '^react-markdown$': '<rootDir>/__mocks__/react-markdown.js',
  },
  coverageProvider: 'v8',
  collectCoverageFrom: [
    // Ensure new components/pages/libs are included in coverage
    'components/HeaderClient.{ts,tsx}',
    'components/MarkdownImage.{ts,tsx}',
    'app/blog/**/page.{ts,tsx}',
    'lib/posts.{ts,tsx}',
    'app/**/*.{js,jsx,ts,tsx}',
    'components/**/*.{js,jsx,ts,tsx}',
    'lib/**/*.{js,jsx,ts,tsx}',
    // Exclude configuration files from coverage (e.g. jest.config.js, *.config.js)
    '!**/*config*.{js,jsx,ts,tsx}',
    '!**/*.config.{js,jsx,ts,tsx}',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/.next/**',
    '!**/out/**',
    '!**/coverage/**',
  ],
  testMatch: ['**/__tests__/**/*.[jt]s?(x)', '**/?(*.)+(spec|test).[jt]s?(x)'],
  testPathIgnorePatterns: ['/node_modules/', '/.next/', '/out/'],
  // Some dependencies ship ESM which Jest can't parse by default. Transform react-markdown.
  transformIgnorePatterns: ['/node_modules/(?!(react-markdown)/)'],
}

// createJestConfig is exported this way to ensure that next/jest can load the Next.js config which is async
module.exports = createJestConfig(customJestConfig)
