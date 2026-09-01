const BASE = import.meta.env.VITE_API_BASE ?? "";

export async function sendChat(query) {
  const res = await fetch(`${BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export async function getHealth() {
  try {
    const res = await fetch(`${BASE}/api/v1/health`);
    return res.ok ? res.json() : null;
  } catch {
    return null;
  }
}
