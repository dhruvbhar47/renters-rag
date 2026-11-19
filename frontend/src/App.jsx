import { useState } from "react";

const EXAMPLES = [
  "How is my tenancy deposit protected?",
  "What can I do if my landlord didn't protect my deposit?",
  "What notice does a landlord have to give to end a tenancy?",
];

// Turn "[1]" markers in the answer into links that light up the matching source below.
function AnswerText({ text, active, onActive }) {
  return text.split("\n").filter(Boolean).map((line, li) => (
    <p key={li}>
      {line.split(/(\[\d+\])/g).map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (!match) return part;
        const n = Number(match[1]);
        return (
          <a
            key={i}
            href={`#source-${n}`}
            className={"cite" + (active === n ? " is-active" : "")}
            aria-label={`Source ${n}`}
            onMouseEnter={() => onActive(n)}
            onMouseLeave={() => onActive(null)}
            onFocus={() => onActive(n)}
            onBlur={() => onActive(null)}
          >
            {n}
          </a>
        );
      })}
    </p>
  ));
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [active, setActive] = useState(null);

  async function ask(q) {
    const text = (q ?? question).trim();
    if (text.length < 3 || loading) return;
    setQuestion(text);
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong. Try again.");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <header>
        <h1>Renting in the UK: ask, then check the source</h1>
        <p className="lede">
          Answers come only from gov.uk renting guidance. Every claim links to the passage it came from, and if the
          guidance doesn't cover your question, you'll be told so.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask();
        }}
      >
        <label htmlFor="q">Your question</label>
        <textarea
          id="q"
          rows={3}
          maxLength={500}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask();
            }
          }}
          placeholder="For example: how long does my landlord have to protect my deposit?"
        />
        <button type="submit" disabled={loading || question.trim().length < 3}>
          {loading ? "Searching the guidance" : "Ask"}
        </button>
      </form>

      {!result && !loading && !error && (
        <section aria-label="Example questions" className="examples">
          <h2>Try one of these</h2>
          <ul>
            {EXAMPLES.map((ex) => (
              <li key={ex}>
                <button type="button" className="link" onClick={() => ask(ex)}>
                  {ex}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}

      {result && (
        <section aria-live="polite" className="result">
          <div className={"answer" + (result.answerable ? "" : " is-refusal")}>
            {result.answerable ? <AnswerText text={result.answer} active={active} onActive={setActive} /> : <p>{result.answer}</p>}
          </div>

          {result.sources.length > 0 && (
            <ol className="sources" aria-label="Sources">
              {result.sources.map((s) => (
                <li key={s.n} id={`source-${s.n}`} className={active === s.n ? "is-active" : ""}>
                  <span className="n">{s.n}</span>
                  <div>
                    <a href={s.url} target="_blank" rel="noreferrer">
                      {s.title}
                    </a>
                    {s.updated_at && <span className="meta">Page updated {s.updated_at}</span>}
                    <blockquote>{s.snippet}…</blockquote>
                  </div>
                </li>
              ))}
            </ol>
          )}
          <p className="meta">Answered in {(result.latency_ms / 1000).toFixed(1)}s</p>
        </section>
      )}

      <footer>
        General information from gov.uk guidance, not legal advice. Rules differ between England, Wales, Scotland and
        Northern Ireland, and they change: always check the linked page.
      </footer>
    </main>
  );
}
