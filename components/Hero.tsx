export default function Hero() {
  return (
    <section
      id="home"
      className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white pt-20 px-4"
    >
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl sm:text-5xl md:text-7xl font-bold mb-6 animate-fade-in">
            Gustavo <span className="text-primary-400">Tsuji</span>
          </h1>
          <p className="text-lg sm:text-xl md:text-2xl text-gray-300 mb-4">
            Senior Software Engineer
          </p>
          <p className="text-base sm:text-lg text-gray-400 mb-8 max-w-2xl mx-auto px-4">
            18+ years building scalable backend systems, optimizing cloud infrastructure, and
            mentoring engineering teams. Double major in Computer Science and Business
            Administration from USP.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center px-4">
            <a
              href="/#contact"
              className="w-full sm:w-auto px-8 py-3 bg-primary-600 hover:bg-primary-700 rounded-lg font-semibold transition-colors inline-block text-center"
            >
              Get in Touch
            </a>
            <a
              href="/blog"
              className="w-full sm:w-auto px-8 py-3 border-2 border-primary-600 hover:bg-primary-600/10 rounded-lg font-semibold transition-colors inline-block text-center"
            >
              Read My Blog
            </a>
          </div>

          {/* Scroll indicator */}
          <div className="mt-16 animate-bounce">
            <a href="/#blog" aria-label="Scroll to blog section">
              <svg
                className="w-6 h-6 mx-auto text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 14l-7 7m0 0l-7-7m7 7V3"
                />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
