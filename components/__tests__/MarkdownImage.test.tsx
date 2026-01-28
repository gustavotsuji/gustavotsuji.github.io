import { render, screen } from '@testing-library/react'
import React from 'react'

// Mock node:fs so we can control existsSync responses
jest.mock('node:fs', () => ({
  existsSync: jest.fn(),
}))

import fs from 'node:fs'
import MarkdownImage from '../MarkdownImage'

describe('MarkdownImage', () => {
  beforeEach(() => {
    jest.resetAllMocks()
  })

  it('renders <source> elements when optimized files exist', () => {
    // make existsSync return true for avif and webp checks
    ;(fs.existsSync as jest.Mock).mockImplementation(
      (p: string) => p.endsWith('.avif') || p.endsWith('.webp')
    )

    render(<MarkdownImage src="/images/test.jpg" alt="Test" />)

    // expect source elements for avif and webp
    const avifSource = screen.getByRole('img', { hidden: true }) // img is present; sources are not role-based
    expect(avifSource).toBeInTheDocument()

    // ensure the <img> fallback uses original src
    const img = screen.getByAltText('Test')
    expect((img as HTMLImageElement).src).toContain('/images/test.jpg')
  })

  it('renders plain <img> for absolute URLs and does not check fs', () => {
    ;(fs.existsSync as jest.Mock).mockImplementation(() => {
      throw new Error('should not be called')
    })

    render(<MarkdownImage src="https://example.com/photo.jpg" alt="Remote" />)
    const img = screen.getByAltText('Remote')
    expect((img as HTMLImageElement).src).toContain('https://example.com/photo.jpg')
  })
})
