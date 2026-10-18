import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SimulatePage } from './simulate'
import './index.css'
import './simulate.css'

const root = document.getElementById('root')
if (!root) throw new Error('Missing #root')

createRoot(root).render(
  <StrictMode>
    <SimulatePage />
  </StrictMode>,
)
