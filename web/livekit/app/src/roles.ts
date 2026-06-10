/**
 * Single source of truth for human-facing role labels.
 *
 * The runtime/DataChannel speaker keys are 'agent' (the provider) and 'user'
 * (the caller) — those are stable and must not change. The *display name* is
 * what humans read; change it here and it updates everywhere in the UI. The name
 * has evolved (Coach → Provider → Guide); keeping it in one place makes the next
 * change a one-line edit instead of a sweep.
 */

export type RoleKey = 'agent' | 'user'

export const ROLE_LABELS: Record<RoleKey, string> = {
  agent: 'Guide',
  user: 'Caller',
}

export const roleLabel = (key: RoleKey): string => ROLE_LABELS[key]
