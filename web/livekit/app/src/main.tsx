import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import App from './App'
import Design1Gemini from './designs/Design1Gemini'
import Design2MinimalOrb from './designs/Design2MinimalOrb'
import Design3Waveform from './designs/Design3Waveform'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/design/gemini" element={<Design1Gemini />} />
        <Route path="/design/minimal" element={<Design2MinimalOrb />} />
        <Route path="/design/waveform" element={<Design3Waveform />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
