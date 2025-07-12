import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [topSnippets, setTopSnippets] = useState([]);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAnswer("");
    setTopSnippets([]);
    setLogs([]);
    setError(null);
    setLoading(true);

    try {
      // Step 1: Trigger backend
      await fetch("http://127.0.0.1:5000/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      // Step 2: Start polling logs
      const pollInterval = setInterval(async () => {
        try {
          const res = await fetch("http://127.0.0.1:5000/api/logs");
          const data = await res.json();
          setLogs(data.logs || []);

          // Check if final step is done
          if (data.logs?.some((line) => line.includes("✅"))) {
            clearInterval(pollInterval);

            // Fetch final result after completion
            const resFinal = await fetch("http://127.0.0.1:5000/api/logs");
            const finalData = await resFinal.json();

            const resAnswer = await fetch(
              "http://127.0.0.1:5000/api/last_result"
            );
            const parsed = await resAnswer.json();
            setAnswer(parsed.llm_answer || "No answer returned.");
            setTopSnippets(parsed.top_snippets || []);
            setLoading(false);
          }
        } catch (pollErr) {
          clearInterval(pollInterval);
          setError("Error polling logs.");
          setLoading(false);
        }
      }, 1000);
    } catch (err) {
      setError("Error fetching results.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold mb-4 text-center">
          AI-Powered Paper Q&A
        </h1>

        <form onSubmit={handleSubmit} className="flex gap-2 mb-6">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 p-2 border border-gray-300 rounded"
          />
          <button
            type="submit"
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            disabled={loading}
          >
            {loading ? "Running..." : "Search"}
          </button>
        </form>

        {error && <p className="text-center text-red-500">{error}</p>}

        {logs.length > 0 && (
          <div className="bg-black text-green-300 font-mono p-4 rounded mb-6 max-h-72 overflow-y-auto">
            {logs.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </div>
        )}

        {answer && (
          <div className="bg-white rounded shadow p-4 mb-6">
            <h2 className="font-semibold text-lg mb-2">Answer</h2>
            <p className="text-gray-800 whitespace-pre-line">{answer}</p>
          </div>
        )}

        {topSnippets.length > 0 && (
          <div>
            <h2 className="font-semibold text-lg mb-2">Top Sources</h2>
            <div className="space-y-4">
              {topSnippets.map((snippet, i) => (
                <div key={i} className="bg-white p-4 rounded shadow">
                  <a
                    href={snippet.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 font-medium hover:underline"
                  >
                    {snippet.title}
                  </a>
                  <p className="text-gray-700 text-sm mt-1">
                    {snippet.snippet}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
