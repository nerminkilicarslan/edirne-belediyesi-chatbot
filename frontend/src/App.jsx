import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";
import { sendChat, getHealth } from "./api";

const QUICK_REPLIES = [
  { label: "💧 Su faturası", value: "Su faturası nasıl ödenir?" },
  { label: "🏠 Emlak vergisi", value: "Emlak, çevre temizlik ve reklam vergisi nasıl ödenir?" },
  { label: "🚌 KentKart kaybı", value: "KentKart'ımı kaybettim, ne yapmalıyım?" },
  { label: "🎓 Öğrenci KentKart", value: "Üniversite öğrencileri için KentKart nasıl çıkarılır?" },
  { label: "💍 Nikâh belgeleri", value: "Nikâh için gerekli belgeler nelerdir?" },
  { label: "🤝 Sosyal yardım", value: "Sosyal yardım başvurusu nasıl yapılır?" },
  { label: "📄 Askıda fatura", value: "Askıda fatura nedir ve nasıl faydalanabilirim?" },
  { label: "📞 İletişim", value: "Belediyeyle nasıl iletişime geçebilirim?" },
  { label: "🚗 Otopark ücretleri", value: "Otopark ücretleri nelerdir?" },
  { label: "💊 Nöbetçi eczane", value: "Nöbetçi eczaneleri nereden görebilirim?" },
  { label: "🎭 Kırkpınar", value: "Kırkpınar Güreşleri hakkında bilgiye nereden ulaşabilirim?" },
  { label: "🏗️ İmar durumu", value: "İmar durumu belgesi almak istiyorum. Ne yapmalıyım?" },
];

function nowStr() {
  return new Date().toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" });
}

function getTier(confidence) {
  if (confidence >= 0.65) return "HIGH";
  if (confidence >= 0.42) return "MEDIUM";
  if (confidence >= 0.28) return "LOW";
  return "VERY_LOW";
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <div className="typing-dots">
      <span />
      <span />
      <span />
    </div>
  );
}

function ConfidencePill({ confidence }) {
  if (confidence === undefined || confidence === null) return null;
  const tier = getTier(confidence);
  if (tier === "VERY_LOW") return null;
  const labels = { HIGH: "Güvenilir", MEDIUM: "Orta", LOW: "Düşük" };
  return (
    <span className={`conf-pill conf-pill--${tier.toLowerCase()}`}>
      {labels[tier]} · {Math.round(confidence * 100)}%
    </span>
  );
}

function SourcePanel({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;

  return (
    <div className="src-panel">
      <button
        className="src-toggle"
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <span>📎 {sources.length} kaynak</span>
        <span className={`src-chevron ${open ? "open" : ""}`}>›</span>
      </button>
      {open && (
        <div className="src-list">
          {sources.slice(0, 4).map((s, i) => {
            const rawSrc = s.metadata?.source ?? "";
            const isUrl = rawSrc.startsWith("http");
            let label = s.metadata?.question ?? "";
            if (!label) {
              try {
                label = isUrl ? new URL(rawSrc).pathname : rawSrc;
              } catch {
                label = rawSrc;
              }
            }
            if (!label) label = "Bilgi kaynağı";

            return isUrl ? (
              <a
                key={i}
                href={rawSrc}
                target="_blank"
                rel="noreferrer"
                className="src-item"
              >
                <span className="src-score">{Math.round(s.score * 100)}%</span>
                <span className="src-label">{label}</span>
                <span className="src-ext">↗</span>
              </a>
            ) : (
              <div key={i} className="src-item src-item--plain">
                <span className="src-score">{Math.round(s.score * 100)}%</span>
                <span className="src-label">{label}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Bubble({ msg, onQuickReply, qrDisabled }) {
  const isBot = msg.role === "bot";
  return (
    <div
      className={`msg ${isBot ? "msg--bot" : "msg--user"}${msg.is_fallback ? " msg--fallback" : ""}`}
    >
      {isBot && <div className="msg__avatar">AI</div>}
      <div className="msg__body">
        <div className="msg__bubble">
          {isBot ? (
            <div className="msg__md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
            </div>
          ) : (
            <span>{msg.text}</span>
          )}
        </div>

        <div className="msg__meta">
          {msg.time && <span className="msg__time">{msg.time}</span>}
          {isBot && <ConfidencePill confidence={msg.confidence} />}
        </div>

        {isBot && <SourcePanel sources={msg.sources} />}

        {msg.showQuickReplies && (
          <div className="quick-replies">
            {QUICK_REPLIES.map((qr, i) => (
              <button
                key={i}
                className="qr-btn"
                onClick={() => onQuickReply(qr.value)}
                disabled={qrDisabled}
                type="button"
              >
                {qr.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Header({ online }) {
  return (
    <header className="hdr">
      <div className="hdr__inner">
        <div className="hdr__brand">
          <div className="hdr__logo">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path
                d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"
                stroke="#fff"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <polyline
                points="9,22 9,12 15,12 15,22"
                stroke="#fff"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div>
            <div className="hdr__name">Edirne Belediyesi</div>
            <div className="hdr__sub">Yapay Zekâ Asistanı</div>
          </div>
        </div>

        <div className="hdr__right">
          <nav className="hdr__nav">
            <a href="https://edirne.bel.tr" target="_blank" rel="noreferrer">
              Resmi Site
            </a>
            <a href="https://edirne.bel.tr/hizmetler" target="_blank" rel="noreferrer">
              Hizmetler
            </a>
            <a href="https://edirne.bel.tr/iletisim" target="_blank" rel="noreferrer">
              İletişim
            </a>
          </nav>
          <div className={`status-badge ${online ? "status-badge--ok" : "status-badge--err"}`}>
            <span className="status-dot" />
            {online ? "Çevrimiçi" : "Bağlanıyor…"}
          </div>
        </div>
      </div>
    </header>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState(true);

  const [messages, setMessages] = useState(() => [
    {
      role: "bot",
      text: "Merhaba! Ben Edirne Belediyesi'nin yapay zekâ asistanıyım. 👋\n\nBelediye hizmetleri, başvurular, işlemler ve duyurular hakkında size yardımcı olmaktan memnuniyet duyarım.",
      time: nowStr(),
    },
    {
      role: "bot",
      text: "Aşağıdaki konulardan birini seçebilir veya sorunuzu yazabilirsiniz:",
      showQuickReplies: true,
      time: nowStr(),
    },
  ]);

  const endRef = useRef(null);
  const inputRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !busy, [input, busy]);

  useEffect(() => {
    getHealth().then((data) => setOnline(!!data));
  }, []);

  useEffect(() => {
    if (open) {
      setTimeout(() => {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
        inputRef.current?.focus();
      }, 80);
    }
  }, [open]);

  function scrollBottom() {
    setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 60);
  }

  const doSend = useCallback(
    async (text) => {
      const clean = text.trim();
      if (!clean || busy) return;
      setInput("");

      setMessages((m) => [...m, { role: "user", text: clean, time: nowStr() }]);
      setBusy(true);
      scrollBottom();

      try {
        const data = await sendChat(clean);
        setMessages((m) => [
          ...m,
          {
            role: "bot",
            text: data.answer || "Üzgünüm, bir yanıt üretilemedi.",
            time: nowStr(),
            confidence: data.confidence,
            sources: data.sources,
            is_fallback: data.is_fallback,
          },
        ]);
      } catch {
        setMessages((m) => [
          ...m,
          {
            role: "bot",
            text: "⚠️ Sunucuya bağlanılamadı. Backend çalışıyor mu kontrol edin.",
            time: nowStr(),
          },
        ]);
      } finally {
        setBusy(false);
        scrollBottom();
      }
    },
    [busy]
  );

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      doSend(input);
    }
  }

  function openAndSend(value) {
    setOpen(true);
    doSend(value);
  }

  return (
    <div className="page">
      <Header online={online} />

      <main className="hero">
        <div className="hero__inner">
          <span className="hero__badge">🏛️ Edirne Belediyesi Resmi Asistanı</span>

          <h1 className="hero__h1">
            Belediye Hizmetleri Hakkında
            <br />
            <span className="hero__accent">Anında Bilgi Alın</span>
          </h1>

          <p className="hero__p">
            Yapay zekâ destekli asistanımız belediye hizmetleri, başvuru süreçleri, vergi
            ödemeleri ve daha fazlası hakkında 7/24 yardımcı olmaya hazır.
          </p>

          <div className="stats-row">
            <div className="stat">
              <div className="stat__num">158+</div>
              <div className="stat__lbl">Bilgi Kaynağı</div>
            </div>
            <div className="stat__divider" />
            <div className="stat">
              <div className="stat__num">11</div>
              <div className="stat__lbl">Hizmet Kategorisi</div>
            </div>
            <div className="stat__divider" />
            <div className="stat">
              <div className="stat__num">7/24</div>
              <div className="stat__lbl">Erişilebilir</div>
            </div>
          </div>

          <div className="topic-grid">
            {QUICK_REPLIES.slice(0, 8).map((qr, i) => (
              <button
                key={i}
                className="topic-card"
                onClick={() => openAndSend(qr.value)}
                type="button"
              >
                {qr.label}
              </button>
            ))}
          </div>
        </div>
      </main>

      {/* FAB */}
      <button
        className="fab"
        onClick={() => setOpen((v) => !v)}
        aria-label="Sohbet aç/kapat"
        type="button"
      >
        {open ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path
              d="M18 6 6 18M6 6l12 12"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path
              d="M21 12c0 4.418-4.03 8-9 8a10.6 10.6 0 0 1-2.4-.27L4 21l1.35-3.24A7.6 7.6 0 0 1 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8Z"
              stroke="currentColor"
              strokeWidth="1.6"
            />
            <path
              d="M8 12h.01M12 12h.01M16 12h.01"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          </svg>
        )}
      </button>

      {/* Chat panel */}
      {open && (
        <section className="panel" role="dialog" aria-label="AI Sohbet">
          <div className="panel__head">
            <div className="panel__head-info">
              <div className="panel__avatar-lg">AI</div>
              <div>
                <div className="panel__title">Edirne AI Asistan</div>
                <div className="panel__online">
                  <span className="online-dot" />
                  Çevrimiçi
                </div>
              </div>
            </div>
            <button
              className="panel__close"
              onClick={() => setOpen(false)}
              type="button"
              aria-label="Kapat"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                <path
                  d="M18 6 6 18M6 6l12 12"
                  stroke="currentColor"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>

          <div className="panel__msgs">
            {messages.map((m, i) => (
              <Bubble
                key={i}
                msg={m}
                onQuickReply={doSend}
                qrDisabled={busy}
              />
            ))}
            {busy && (
              <div className="msg msg--bot">
                <div className="msg__avatar">AI</div>
                <div className="msg__body">
                  <div className="msg__bubble">
                    <TypingDots />
                  </div>
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          <div className="panel__foot">
            <div className="panel__input-row">
              <input
                ref={inputRef}
                className="panel__input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Sorunuzu yazın…"
                disabled={busy}
                maxLength={500}
              />
              <button
                className="panel__send"
                onClick={() => doSend(input)}
                disabled={!canSend}
                type="button"
                aria-label="Gönder"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
            <p className="panel__disclaimer">
              Cevaplar Edirne Belediyesi bilgi tabanından üretilmektedir.
            </p>
          </div>
        </section>
      )}
    </div>
  );
}
