const experiences = [
  {
    company: 'Grupo OLX',
    role: 'Senior Software Engineer',
    period: 'Jan 2024 - Present',
    description: 'Ad Integration & Autos ProPerformance Teams',
    highlights: [
      'Delivered backend implementation of Google Reviews, driving a +13-15% increase in CTR and +15-25% in Conversion Rate',
      'Led cloud cost and infrastructure optimization initiatives, reducing compute costs by ~60% (AWS Graviton)',
      'Modernized the Integrations ecosystem by upgrading to Node.js v24',
      'Mentored peers and hosted knowledge-sharing sessions on Architecture, DevOps, and Database optimization',
    ],
    tags: ['Node.js', 'AWS', 'PostgreSQL', 'Docker', 'System Architecture'],
  },
  {
    company: 'Associação Nova Escola',
    role: 'Technical Lead',
    period: 'Jul 2021 - Nov 2023',
    description: 'Education NGO',
    highlights: [
      'Designed and implemented a content-based recommendation system for the web portal',
      'Engineered an automation pipeline to sync Google Docs content directly into MySQL',
      "Pioneered an AI-driven documentation process using OpenAI's Whisper and ChatGPT",
      'Mentored junior and mid-level developers through 1:1 meetings and Personal Development Plans',
    ],
    tags: ['Node.js', 'TypeORM', 'Nest.js', 'AWS', 'MySQL'],
  },
  {
    company: 'Coteminas',
    role: 'Javascript Backend Developer',
    period: 'Sep 2019 - Jul 2021',
    description: 'Bedding, bath and home decor',
    highlights: [
      'Engineered and tuned the AMMO ecommerce environment to sustain peak loads of 150k requests per minute during Black Friday',
      'Spearheaded the migration of legacy systems to a scalable microservices architecture using NestJS',
      'Implemented data-driven observability pipelines using AWS Athena',
    ],
    tags: ['Node.js', 'NestJS', 'Kubernetes', 'AWS', 'PostgreSQL'],
  },
  {
    company: 'Mercado Livre',
    role: 'Senior Java Developer',
    period: 'Aug 2016 - Jul 2017',
    description: 'E-commerce Leader in LatAm',
    highlights: [
      'Orchestrated critical, large-scale database migrations using Oracle Golden Gate',
      'Optimized internal job processes and integration scripts',
      'Collaborated in a cross-functional, bi-national environment (Brazil & Argentina)',
    ],
    tags: ['Java', 'Oracle', 'MySQL', 'AWS', 'Redis'],
  },
]

export default function Experience() {
  return (
    <section id="experience" className="py-20 bg-gray-50 dark:bg-gray-800">
      <div className="container mx-auto px-4">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-12 text-center">
            Professional Experience
          </h2>

          <div className="space-y-8">
            {experiences.map((exp, index) => (
              <div
                key={index}
                className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6 md:p-8 hover:shadow-xl transition-shadow"
              >
                <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                  <div>
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                      {exp.company}
                    </h3>
                    <p className="text-lg text-primary-600 dark:text-primary-400 font-semibold">
                      {exp.role}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 italic">
                      {exp.description}
                    </p>
                  </div>
                  <div className="mt-2 md:mt-0">
                    <span className="inline-block px-4 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-full text-sm font-medium">
                      {exp.period}
                    </span>
                  </div>
                </div>

                <ul className="space-y-2 mb-4">
                  {exp.highlights.map((highlight, idx) => (
                    <li key={idx} className="flex items-start text-gray-700 dark:text-gray-300">
                      <svg
                        className="w-5 h-5 text-primary-600 dark:text-primary-400 mr-2 flex-shrink-0 mt-0.5"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span>{highlight}</span>
                    </li>
                  ))}
                </ul>

                <div className="flex flex-wrap gap-2">
                  {exp.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-full text-xs font-medium"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-12 text-center">
            <a
              href="https://linkedin.com/in/gustavo-tsuji-7100462b"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold transition-colors"
            >
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z" />
              </svg>
              View Full Experience on LinkedIn
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
