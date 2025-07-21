import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { jsPDF } from "jspdf";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [topSnippets, setTopSnippets] = useState([]);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoChunks, setVideoChunks] = useState([]);
  const [apiReady, setApiReady] = useState(false);
  const [players, setPlayers] = useState({});
  const [email, setEmail] = useState("");
  const [emailStatus, setEmailStatus] = useState(null);

  
const generatePDFBlob = (query, answer, topSnippets) => {
  try {
    const doc = new jsPDF();
    let y = 20;
    const pageHeight = doc.internal.pageSize.height;
    const margin = 15;
    const maxWidth = 180;
    const lineHeight = 6;
    
    // Helper function to check if we need a new page
    const checkNewPage = (linesNeeded = 1) => {
      if (y + (linesNeeded * lineHeight) > pageHeight - 20) {
        doc.addPage();
        y = 20;
      }
    };
    
    // Title
    doc.setFontSize(18);
    doc.setFont(undefined, 'bold');
    doc.text('Research Report', margin, y);
    y += 12;
    
    // Date
    doc.setFontSize(9);
    doc.setFont(undefined, 'normal');
    doc.setTextColor(100);
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, y);
    y += 15;
    
    // Reset color
    doc.setTextColor(0);
    
    // Query Section
    checkNewPage(3);
    doc.setFontSize(14);
    doc.setFont(undefined, 'bold');
    doc.text('Query', margin, y);
    y += 8;
    
    doc.setFontSize(11);
    doc.setFont(undefined, 'normal');
    const queryText = query || 'No query provided';
    const queryLines = doc.splitTextToSize(queryText, maxWidth);
    
    checkNewPage(queryLines.length);
    queryLines.forEach(line => {
      doc.text(line, margin, y);
      y += lineHeight;
    });
    y += 8;
    
    // Answer Section
    if (answer) {
      checkNewPage(3);
      doc.setFontSize(14);
      doc.setFont(undefined, 'bold');
      doc.text('Answer', margin, y);
      y += 8;
      
      doc.setFontSize(11);
      doc.setFont(undefined, 'normal');
      
      // Clean the answer text (remove markdown formatting for PDF)
      const cleanAnswer = answer
        .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown
        .replace(/\*(.*?)\*/g, '$1')     // Remove italic markdown
        .replace(/\[(.*?)\]\(.*?\)/g, '$1') // Convert links to just text
        .replace(/#{1,6}\s/g, '')        // Remove headers
        .trim();
      
      const answerLines = doc.splitTextToSize(cleanAnswer, maxWidth);
      
      answerLines.forEach(line => {
        checkNewPage();
        doc.text(line, margin, y);
        y += lineHeight;
      });
      y += 12;
    }
    
    // Sources Section
    if (topSnippets && topSnippets.length > 0) {
      checkNewPage(3);
      doc.setFontSize(14);
      doc.setFont(undefined, 'bold');
      doc.text('Sources', margin, y);
      y += 10;
      
      topSnippets.forEach((snippet, i) => {
        // Source number
        checkNewPage(2);
        doc.setFontSize(12);
        doc.setFont(undefined, 'bold');
        doc.text(`${i + 1}.`, margin, y);
        y += 7;
        
        // Source title
        doc.setFontSize(10);
        doc.setFont(undefined, 'bold');
        const title = snippet.title || `Source ${i + 1}`;
        const titleLines = doc.splitTextToSize(title, maxWidth - 5);
        
        titleLines.forEach(line => {
          checkNewPage();
          doc.text(line, margin + 5, y);
          y += lineHeight;
        });
        
        // Source content
        if (snippet.snippet) {
          doc.setFont(undefined, 'normal');
          const contentLines = doc.splitTextToSize(snippet.snippet, maxWidth - 5);
          
          contentLines.forEach(line => {
            checkNewPage();
            doc.text(line, margin + 5, y);
            y += lineHeight;
          });
        }
        
        // Source URL
        if (snippet.url) {
          doc.setTextColor(50);
          doc.setFontSize(9);
          const urlLines = doc.splitTextToSize(snippet.url, maxWidth - 5);
          
          urlLines.forEach(line => {
            checkNewPage();
            doc.text(line, margin + 5, y);
            y += lineHeight;
          });
          doc.setTextColor(0);
        }
        
        y += 8; // Space between sources
      });
    }
    
    return doc.output("blob");
  } catch (error) {
    console.error('Error generating PDF:', error);
    // Fallback to simple PDF if enhanced version fails
    const doc = new jsPDF();
    doc.setFontSize(12);
    doc.text('Error generating detailed report', 10, 20);
    doc.text(`Query: ${query || 'N/A'}`, 10, 40);
    doc.text('Please try again or contact support.', 10, 60);
    return doc.output("blob");
  }
};


  const blobToBase64 = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result); // includes base64 header
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });


    const handleEmailSend = async () => {
      setEmailStatus(null);
    
      try {
        const pdfBlob = generatePDFBlob(query, answer, topSnippets);
        const pdfBase64 = await blobToBase64(pdfBlob); // includes the data URI prefix
    
        const res = await fetch("http://127.0.0.1:5000/api/send-pdf", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email,
            pdf_data: pdfBase64, // Flask expects this
            subject: "Your PDF Export",
          }),
        });
    
        if (res.ok) {
          setEmailStatus("Report sent successfully!");
        } else {
          setEmailStatus("Failed to send report.");
        }
      } catch (err) {
        console.error("Error sending report:", err);
        setEmailStatus("Error sending report.");
      }
    };


  useEffect(() => {
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.body.appendChild(tag);
  
    window.onYouTubeIframeAPIReady = () => {
      setApiReady(true);
    };
  }, []);

  useEffect(() => {
    if (!apiReady || videoChunks.length === 0) return;
  
    const newPlayers = {};
    videoChunks.forEach((video, i) => {
      newPlayers[i] = new window.YT.Player(`player-${i}`, {
        height: "180",
        width: "320",
        videoId: video.video_id,
        events: {
          onReady: (event) => {
            console.log(`Player ${i} ready`);
          },
        },
      });
    });
    setPlayers(newPlayers);
  }, [apiReady, videoChunks]);



  useEffect(() => {
    console.log("[DEBUG] videoChunks rendering:", videoChunks);
  }, [videoChunks]);

  function formatSeconds(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    return `${min}:${sec.toString().padStart(2, "0")}`;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAnswer("");
    setTopSnippets([]);
    setLogs([]);
    setError(null);
    setVideoChunks([]);
    setLoading(true);

    try {
      await fetch("http://127.0.0.1:5000/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const pollInterval = setInterval(async () => {
        try {
          const res = await fetch("http://127.0.0.1:5000/api/logs");
          const data = await res.json();
          setLogs(data.logs || []);

          if (data.logs?.some((line) => line.includes("✅"))) {
            clearInterval(pollInterval);

            const resAnswer = await fetch(
              "http://127.0.0.1:5000/api/last_result"
            );
            const parsed = await resAnswer.json();

            console.log("[DEBUG] last_result response:", parsed);

            setAnswer(parsed.llm_answer || "No answer returned.");
            setTopSnippets(parsed.top_snippets || []);
            setVideoChunks(parsed.videos || []); // ✅ assign correctly
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
          AI Powered Report Generator
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
          <>
            <div className="bg-white rounded shadow p-4 mb-6">
              <h2 className="font-semibold text-lg mb-2">Answer</h2>
              <div className="prose max-w-none text-gray-800 prose-a:text-blue-600 hover:prose-a:underline">
                <ReactMarkdown
                  components={{
                    a: ({ node, ...props }) => (
                      <a {...props} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline" />
                    ),
                  }}
                >
                  {answer}
                </ReactMarkdown>
              </div>
            </div>

            <div className="bg-white rounded shadow p-4 mb-6">
              <h2 className="font-semibold text-lg mb-2">Email this Report</h2>
              <div className="flex gap-2">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email"
                  className="flex-1 p-2 border border-gray-300 rounded"
                  aria-label="Recipient email"
                />
                <button
                  onClick={handleEmailSend}
                  className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
                  disabled={!email}
                >
                  Send Report
                </button>
              </div>
              {emailStatus && (
                <p className="text-sm mt-2 text-center text-gray-600">{emailStatus}</p>
              )}
            </div>
          </>
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

        {videoChunks.length > 0 && (
          <div className="mt-8">
            <h2 className="font-semibold text-lg mb-4">Video Insights</h2>
            <div className="space-y-6">
              {videoChunks.map((video, i) => (
                <div key={i} className="flex gap-4 bg-white p-4 rounded shadow">
                  {/* Embedded video */}
                  <div className="w-[320px] h-[180px] flex-shrink-0">
                  <div
  id={`player-${i}`}
  className="w-[320px] h-[180px] flex-shrink-0"
/>
                  </div>
                  

                  {/* Right side: title + summary + key moments */}
                  <div className="flex flex-col justify-center max-w-md">
                    <p className="text-md font-semibold text-gray-800 mb-1">
                      {video.title}
                    </p>
                    <p className="text-sm italic text-gray-600 mb-3">
                      {video.summary}
                    </p>
                    <div className="space-y-2">
                    {video.moments.map((moment, j) => (
                      <div
                        key={j}
                        className="text-sm text-blue-600 hover:underline cursor-pointer"
                        onClick={() => {
                          const player = players[i];
                          if (player && player.seekTo) {
                            player.seekTo(moment.start, true);
                            player.playVideo();
                          }
                        }}
                      >
                        ⏱️ {formatSeconds(moment.start)} – {moment.summary}
                      </div>
                    ))}
                    </div>
                  </div>
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
