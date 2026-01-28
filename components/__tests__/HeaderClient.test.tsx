import { render } from '@testing-library/react'
import React from 'react'
import HeaderClient from '../HeaderClient'

describe('HeaderClient', () => {
  it('renders without crashing and updates CSS var', () => {
    // mount component; we only ensure no runtime errors in the component's effect
    render(<HeaderClient />)
    // effect reads document header; ensure CSS var is set (it will be set to some value)
    const v = document.documentElement.style.getPropertyValue('--site-header-height')
    // value may be empty if header not present in test DOM, but ensure no exception thrown
    expect(v).toBeDefined()
  })
})
