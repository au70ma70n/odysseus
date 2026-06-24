/**
 * Shared URL-hash helpers for session routing.
 * Used by sessions.js (load/hashchange) and chatRenderer.js (link clicks).
 */

const _SESSION_UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const _ENTITY_PREFIX_RE = /^(document|note|image|email|event|task|skill|research)-/;

/** Extract a session UUID from a raw hash fragment (no leading #). */
export function sessionIdFromHash(raw) {
  const h = String(raw || '').replace(/^#/, '').trim();
  if (!h) return null;
  if (_ENTITY_PREFIX_RE.test(h)) return null;
  const prefixed = h.match(/^session-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i);
  if (prefixed) return prefixed[1];
  if (_SESSION_UUID_RE.test(h)) return h;
  return null;
}

/** Extract session UUID from an href (relative #hash or absolute URL with hash). */
export function sessionIdFromHref(href) {
  const raw = String(href || '');
  if (!raw) return null;
  const hashIdx = raw.indexOf('#');
  if (hashIdx < 0) return null;
  return sessionIdFromHash(raw.slice(hashIdx + 1));
}

/**
 * Hash fragments that are not session IDs but should be left alone
 * (e.g. #cookbook fallback, #document-… handled elsewhere).
 */
export function isPreservedAppHash(raw) {
  const h = String(raw || '').replace(/^#/, '').trim();
  if (!h) return true;
  if (_ENTITY_PREFIX_RE.test(h)) return true;
  // Single-segment routes like #cookbook — not session UUIDs.
  if (/^[a-z][a-z0-9_-]*$/i.test(h) && !h.includes('-')) return true;
  return false;
}
