export default function About() {
  return (
    <section id="about" className="py-20 bg-white dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            About Me
          </h2>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <img
                src="https://s.gravatar.com/avatar/67eb167c18902ec6d32e0117432665cc?s=400"
                alt="Gustavo Tsuji"
                className="rounded-lg shadow-xl w-full"
              />
            </div>

            <div className="space-y-4 text-gray-700 dark:text-gray-300">
              <p className="text-lg">
                Experienced Developer with <strong>18+ years</strong> of professional experience and
                a double major in <strong>Computer Science</strong> and{' '}
                <strong>Business Administration</strong> from University of São Paulo (USP).
              </p>

              <p>
                My academic research in <strong>Information Retrieval (Hibernate Search)</strong>{' '}
                and
                <strong> Machine Learning</strong> supports my pragmatic approach to solving complex
                data problems.
              </p>

              <p>
                Currently focused on <strong>backend scalability</strong>,{' '}
                <strong>cloud efficiency</strong>, and <strong>mentoring</strong> the next
                generation of engineers. Passionate about building robust, scalable systems and
                optimizing infrastructure costs while maintaining high performance and reliability.
              </p>

              <div className="pt-4">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                  Key Skills
                </h3>
                <div className="flex flex-wrap gap-2">
                  {[
                    'Node.js',
                    'Java',
                    'TypeScript',
                    'AWS',
                    'PostgreSQL',
                    'Docker',
                    'Kubernetes',
                    'System Architecture',
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
        </div>
      </div>
    </section>
  )
}
