import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Route, Routes, Navigate } from 'react-router-dom'
import {
  ClerkProvider,
  SignedIn,
  SignedOut,
  SignInButton,
  UserButton,
} from '@clerk/clerk-react'
import LiveSpeechDialogue from './designs/LiveSpeechDialogue'
import './index.css'

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY as string | undefined

if (!PUBLISHABLE_KEY) {
  // Fail loud in dev; a missing key would otherwise render a blank ClerkProvider.
  console.error('VITE_CLERK_PUBLISHABLE_KEY is not set — sign-in will not work.')
}

const signedOutShell: React.CSSProperties = {
  minHeight: '100dvh',
  background: '#020617',
  color: '#e2e8f0',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 16,
  fontFamily: 'system-ui, -apple-system, sans-serif',
}

const signedOutTitle: React.CSSProperties = {
  fontSize: 28,
  fontWeight: 600,
  letterSpacing: '0.15em',
}

const signedOutSubtitle: React.CSSProperties = {
  color: '#64748b',
  fontSize: 14,
}

const userButtonWrap: React.CSSProperties = {
  position: 'fixed',
  top: 12,
  right: 16,
  zIndex: 50,
}

/** Gate the call UI behind Clerk auth. Signed-out users see a sign-in prompt. */
function Gate() {
  return (
    <>
      <SignedIn>
        <LiveSpeechDialogue />
      </SignedIn>
      <SignedOut>
        <div style={signedOutShell}>
          <h1 style={signedOutTitle}>Rehearse</h1>
          <p style={signedOutSubtitle}>Sign in to start a coaching session.</p>
          <SignInButton mode="modal" />
        </div>
      </SignedOut>
    </>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY ?? ''}>
      {/* Account control, visible only when signed in. */}
      <SignedIn>
        <div style={userButtonWrap}>
          <UserButton />
        </div>
      </SignedIn>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Gate />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ClerkProvider>
  </React.StrictMode>,
)
