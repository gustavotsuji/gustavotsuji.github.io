import { render, screen, fireEvent } from '@testing-library/react'
import LanguageSelector from '../LanguageSelector'

describe('LanguageSelector', () => {
  const mockOnSelectLang = jest.fn()

  beforeEach(() => {
    mockOnSelectLang.mockClear()
  })

  it('renders all language buttons', () => {
    render(<LanguageSelector selectedLang="en" onSelectLang={mockOnSelectLang} />)

    expect(screen.getByRole('button', { name: /português/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /english/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /español/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /日本語/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /français/i })).toBeInTheDocument()
  })

  it('highlights selected language button', () => {
    render(<LanguageSelector selectedLang="pt" onSelectLang={mockOnSelectLang} />)

    const ptButton = screen.getByRole('button', { name: /português/i })
    expect(ptButton).toHaveAttribute('aria-pressed', 'true')
    expect(ptButton).toHaveClass('bg-primary-600')
  })

  it('does not highlight unselected language buttons', () => {
    render(<LanguageSelector selectedLang="en" onSelectLang={mockOnSelectLang} />)

    const ptButton = screen.getByRole('button', { name: /português/i })
    expect(ptButton).toHaveAttribute('aria-pressed', 'false')
    expect(ptButton).not.toHaveClass('bg-primary-600')
  })

  it('calls onSelectLang when language is clicked', () => {
    render(<LanguageSelector selectedLang="en" onSelectLang={mockOnSelectLang} />)

    const jaButton = screen.getByRole('button', { name: /日本語/i })
    fireEvent.click(jaButton)

    expect(mockOnSelectLang).toHaveBeenCalledWith('ja')
  })

  it('calls onSelectLang with correct code for each language', () => {
    render(<LanguageSelector selectedLang="en" onSelectLang={mockOnSelectLang} />)

    const languages = [
      { name: /português/i, code: 'pt' },
      { name: /english/i, code: 'en' },
      { name: /español/i, code: 'es' },
      { name: /日本語/i, code: 'ja' },
      { name: /français/i, code: 'fr' },
    ]

    languages.forEach(({ name, code }) => {
      mockOnSelectLang.mockClear()
      const button = screen.getByRole('button', { name })
      fireEvent.click(button)
      expect(mockOnSelectLang).toHaveBeenCalledWith(code)
    })
  })

  it('renders nothing when onSelectLang is not provided', () => {
    const { container } = render(<LanguageSelector selectedLang="en" />)

    expect(container.firstChild).toBeNull()
  })

  it('renders with different selected languages', () => {
    const { rerender } = render(
      <LanguageSelector selectedLang="en" onSelectLang={mockOnSelectLang} />
    )

    let enButton = screen.getByRole('button', { name: /english/i })
    expect(enButton).toHaveAttribute('aria-pressed', 'true')

    rerender(<LanguageSelector selectedLang="ja" onSelectLang={mockOnSelectLang} />)

    const jaButton = screen.getByRole('button', { name: /日本語/i })
    expect(jaButton).toHaveAttribute('aria-pressed', 'true')

    enButton = screen.getByRole('button', { name: /english/i })
    expect(enButton).toHaveAttribute('aria-pressed', 'false')
  })

  it('displays full language names on desktop', () => {
    render(<LanguageSelector selectedLang="en" onSelectLang={mockOnSelectLang} />)

    const ptButton = screen.getByRole('button', { name: /português/i })
    const ptLabel = ptButton.querySelector(String.raw`.hidden.sm\:inline`)

    expect(ptLabel).toHaveTextContent('Português')
  })

  it('has accessibility attributes', () => {
    render(<LanguageSelector selectedLang="en" onSelectLang={mockOnSelectLang} />)

    const buttons = screen.getAllByRole('button')

    buttons.forEach((button) => {
      expect(button).toHaveAttribute('aria-label')
      expect(button).toHaveAttribute('aria-pressed')
    })
  })

  it('supports both callback and dispatch function handlers', () => {
    const dispatchMock: React.Dispatch<string> = jest.fn()

    render(<LanguageSelector selectedLang="en" onSelectLang={dispatchMock} />)

    const jaButton = screen.getByRole('button', { name: /日本語/i })
    fireEvent.click(jaButton)

    expect(dispatchMock).toHaveBeenCalledWith('ja')
  })
})
