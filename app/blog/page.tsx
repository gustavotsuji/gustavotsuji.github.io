import BlogLanguageSelector from '@/components/BlogLanguageSelector'
import { getAllPosts } from '@/lib/posts'

export const metadata = {
  title: 'Blog - Gustavo Tsuji',
  description: 'Technical articles about software engineering, cloud architecture, and leadership',
}

export default function BlogPage() {
  const posts = getAllPosts()
  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 pt-24 pb-20">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          <div className="mb-12">
            <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">Blog</h1>
            <p className="text-xl text-gray-600 dark:text-gray-400">
              Thoughts on software engineering, cloud architecture, and technical leadership
            </p>
          </div>
          <BlogLanguageSelector allPosts={posts} />
        </div>
      </div>
    </div>
  )
}
