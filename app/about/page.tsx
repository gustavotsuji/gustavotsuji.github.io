import type { Metadata } from 'next'
import Link from 'next/link'

export const metadata: Metadata = {
  title: 'About - Gustavo Tsuji',
  description:
    'Learn more about Gustavo Tsuji - Senior Software Engineer with 18+ years of experience',
}

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 pt-24 pb-20">
      <div className="container mx-auto px-4">
        <div className="max-w-5xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-6">About Me</h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
              Experienced Developer with 18+ years building scalable systems and leading engineering
              teams
            </p>
          </div>

          {/* Main Content */}
          <div className="grid md:grid-cols-2 gap-12 items-start mb-16">
            <div className="space-y-6">
              <img
                src="https://s.gravatar.com/avatar/67eb167c18902ec6d32e0117432665cc?s=600"
                alt="Gustavo Tsuji"
                className="rounded-lg shadow-2xl w-full"
              />

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                  Contact Information
                </h3>
                <ul className="space-y-3 text-gray-700 dark:text-gray-300">
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 mr-3 text-primary-600 dark:text-primary-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                      />
                    </svg>
                    <a
                      href="mailto:gustavokt@gmail.com"
                      className="hover:text-primary-600 dark:hover:text-primary-400"
                    >
                      gustavokt@gmail.com
                    </a>
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 mr-3 text-primary-600 dark:text-primary-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                      />
                    </svg>
                    <a
                      href="tel:+5511994168215"
                      className="hover:text-primary-600 dark:hover:text-primary-400"
                    >
                      +55 (11) 99416-8215
                    </a>
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 mr-3 text-primary-600 dark:text-primary-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                    </svg>
                    São Paulo, SP - Brazil
                  </li>
                </ul>

                <div className="mt-6 flex space-x-4">
                  <a
                    href="https://www.linkedin.com/in/gustavo-tsuji-7100462b"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold transition-colors text-center"
                  >
                    LinkedIn
                  </a>
                  <a
                    href="https://github.com/gustavotsuji"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 px-4 py-2 bg-gray-800 hover:bg-gray-900 text-white rounded-lg font-semibold transition-colors text-center"
                  >
                    GitHub
                  </a>
                </div>
              </div>
            </div>

            <div className="space-y-8">
              <div>
                <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                  Professional Background
                </h2>
                <div className="prose dark:prose-invert max-w-none text-gray-700 dark:text-gray-300 space-y-4">
                  <p className="text-lg">
                    Experienced Developer with <strong>18+ years</strong> of professional experience
                    and a double major in <strong>Computer Science</strong> and{' '}
                    <strong>Business Administration</strong> from University of São Paulo (USP).
                  </p>

                  <p>
                    My academic research in{' '}
                    <strong>Information Retrieval (Hibernate Search)</strong> and
                    <strong> Machine Learning</strong> supports my pragmatic approach to solving
                    complex data problems.
                  </p>

                  <p>
                    Currently focused on <strong>backend scalability</strong>,{' '}
                    <strong>cloud efficiency</strong>, and <strong>mentoring</strong> the next
                    generation of engineers. Passionate about building robust, scalable systems and
                    optimizing infrastructure costs while maintaining high performance and
                    reliability.
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Education</h3>
                <div className="space-y-4">
                  <div className="border-l-4 border-primary-600 pl-4">
                    <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Bachelor in Business Administration
                    </h4>
                    <p className="text-gray-600 dark:text-gray-400">
                      University of São Paulo (FEA-USP) • December 2016
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                      Thesis: "Segmentação e classificação de Big Data" (Machine Learning/Spark)
                    </p>
                  </div>

                  <div className="border-l-4 border-primary-600 pl-4">
                    <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                      Bachelor in Computer Science
                    </h4>
                    <p className="text-gray-600 dark:text-gray-400">
                      University of São Paulo (IME-USP) • December 2008
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
                      Thesis: "Integrando recuperação de informação em banco de dados com Hibernate
                      Search"
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                  Technical Skills
                </h3>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">
                      Languages & Frameworks
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {[
                        'Node.js',
                        'JavaScript',
                        'TypeScript',
                        'Java',
                        'Spring Boot',
                        'NestJS',
                        'Python',
                      ].map((skill) => (
                        <span
                          key={skill}
                          className="px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">
                      Cloud & Infrastructure
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {['AWS', 'Docker', 'Kubernetes', 'Terraform', 'CI/CD', 'Microservices'].map(
                        (skill) => (
                          <span
                            key={skill}
                            className="px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium"
                          >
                            {skill}
                          </span>
                        )
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">
                      Databases & Data
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {['PostgreSQL', 'MySQL', 'Redis', 'MongoDB', 'Oracle', 'Database Design'].map(
                        (skill) => (
                          <span
                            key={skill}
                            className="px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium"
                          >
                            {skill}
                          </span>
                        )
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">
                      Leadership & Others
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {[
                        'Technical Leadership',
                        'Mentoring',
                        'System Architecture',
                        'Agile',
                        'Code Review',
                      ].map((skill) => (
                        <span
                          key={skill}
                          className="px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Languages</h3>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>
                    🇧🇷 <strong>Portuguese:</strong> Native
                  </li>
                  <li>
                    🇺🇸 <strong>English:</strong> Full Professional
                  </li>
                  <li>
                    🇪🇸 <strong>Spanish:</strong> Limited
                  </li>
                  <li>
                    🇯🇵 <strong>Japanese:</strong> Basic
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* CTA */}
          <div className="bg-gradient-to-br from-primary-600 to-primary-700 rounded-lg p-8 text-center text-white">
            <h2 className="text-3xl font-bold mb-4">Let's Work Together</h2>
            <p className="text-lg mb-6 opacity-90">
              Interested in collaborating on a project or discussing opportunities?
            </p>
            <Link
              href="/#contact"
              className="inline-block px-8 py-3 bg-white text-primary-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Get in Touch
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
