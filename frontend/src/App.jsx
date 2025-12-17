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
  
  // Modal states
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [email, setEmail] = useState("");
  const [feedbackText, setFeedbackText] = useState("");
  
  // Toast notification state
  const [toast, setToast] = useState(null);

  const showToast = (message, type = "success") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  const generatePDFBlob = (query, answer, topSnippets) => {
    try {
      const doc = new jsPDF();
      let y = 20;
      const pageHeight = doc.internal.pageSize.height;
      const margin = 15;
      const maxWidth = 180;
      const lineHeight = 6;
      
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
        
        const cleanAnswer = answer
          .replace(/\*\*(.*?)\*\*/g, '$1')
          .replace(/\*(.*?)\*/g, '$1')
          .replace(/\[(.*?)\]\(.*?\)/g, '$1')
          .replace(/#{1,6}\s/g, '')
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
          checkNewPage(2);
          doc.setFontSize(12);
          doc.setFont(undefined, 'bold');
          doc.text(`${i + 1}.`, margin, y);
          y += 7;
          
          doc.setFontSize(10);
          doc.setFont(undefined, 'bold');
          const title = snippet.title || `Source ${i + 1}`;
          const titleLines = doc.splitTextToSize(title, maxWidth - 5);
          
          titleLines.forEach(line => {
            checkNewPage();
            doc.text(line, margin + 5, y);
            y += lineHeight;
          });
          
          if (snippet.snippet) {
            doc.setFont(undefined, 'normal');
            const contentLines = doc.splitTextToSize(snippet.snippet, maxWidth - 5);
            
            contentLines.forEach(line => {
              checkNewPage();
              doc.text(line, margin + 5, y);
              y += lineHeight;
            });
          }
          
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
          
          y += 8;
        });
      }
      
      return doc.output("blob");
    } catch (error) {
      console.error('Error generating PDF:', error);
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
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  const handleEmailSend = async () => {
    if (!email) {
      showToast("Please enter an email address", "error");
      return;
    }

    try {
      const pdfBlob = generatePDFBlob(query, answer, topSnippets);
      const pdfBase64 = await blobToBase64(pdfBlob);

      const res = await fetch("http://127.0.0.1:5000/api/send-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          pdf_data: pdfBase64,
          subject: "Your Research Report",
        }),
      });

      if (res.ok) {
        showToast("Report sent successfully!", "success");
        setShowEmailModal(false);
        setEmail("");
      } else {
        showToast("Failed to send report", "error");
      }
    } catch (err) {
      console.error("Error sending report:", err);
      showToast("Error sending report", "error");
    }
  };

  const handleFeedbackSend = async () => {
    if (!feedbackText.trim()) {
      showToast("Please enter your feedback", "error");
      return;
    }

    try {
      const mailtoLink = `mailto:sebastienbrown1@gmail.com?subject=AI Report Generator Feedback&body=${encodeURIComponent(feedbackText)}`;
      window.location.href = mailtoLink;
      
      showToast("Opening your email client...", "success");
      setShowFeedbackModal(false);
      setFeedbackText("");
    } catch (err) {
      console.error("Error opening email client:", err);
      showToast("Error opening email client", "error");
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

  function formatSeconds(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    return `${min}:${sec.toString().padStart(2, "0")}`;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      showToast("Please enter a query", "error");
      return;
    }

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

            const resAnswer = await fetch("http://127.0.0.1:5000/api/last_result");
            const parsed = await resAnswer.json();

            setAnswer(parsed.llm_answer || "No answer returned.");
            setTopSnippets(parsed.top_snippets || []);
            setVideoChunks(parsed.videos || []);
            setLoading(false);
            showToast("Search completed!", "success");
          }
        } catch (pollErr) {
          clearInterval(pollInterval);
          setError("Error polling logs.");
          setLoading(false);
          showToast("Error during search", "error");
        }
      }, 1000);
    } catch (err) {
      setError("Error fetching results.");
      setLoading(false);
      showToast("Error fetching results", "error");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Toast Notification */}
      {toast && (
        <div className="fixed top-4 right-4 z-50 animate-slideIn">
          <div className={`px-6 py-3 rounded-lg shadow-lg flex items-center gap-3 ${
            toast.type === "success" 
              ? "bg-emerald-500 text-white" 
              : "bg-red-500 text-white"
          }`}>
            <span className="text-lg">
              {toast.type === "success" ? "✓" : "⚠"}
            </span>
            <span className="font-medium">{toast.message}</span>
          </div>
        </div>
      )}

      {/* Top Toolbar */}
      <div className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center shadow-md">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              ReportIO - An AI Research Assistant
            </h1>
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={() => setShowEmailModal(true)}
              disabled={!answer}
              className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              Email Report
            </button>
            
            <button
              onClick={() => setShowFeedbackModal(true)}
              className="px-4 py-2 bg-white border-2 border-slate-200 text-slate-700 rounded-lg font-medium hover:border-slate-300 hover:bg-slate-50 transition-all flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              </svg>
              Feedback
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Search Section */}
        <div className="mb-8">
          <form onSubmit={handleSubmit} className="relative">
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask me anything... I'll search and analyze for you"
                className="w-full px-6 py-4 pr-32 text-lg border-2 border-slate-200 rounded-xl focus:border-blue-500 focus:outline-none focus:ring-4 focus:ring-blue-100 transition-all shadow-sm bg-white"
                disabled={loading}
              />
              <button
                type="submit"
                className="absolute right-2 top-2 px-6 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center gap-2"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Searching...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    Search
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg mb-6">
            <div className="flex items-center gap-3">
              <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-700 font-medium">{error}</p>
            </div>
          </div>
        )}

        {/* Logs Terminal */}
        {logs.length > 0 && (
          <div className="bg-slate-900 rounded-xl shadow-2xl mb-8 overflow-hidden border border-slate-700">
            <div className="bg-slate-800 px-4 py-2 border-b border-slate-700 flex items-center gap-2">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <span className="text-slate-400 text-sm font-mono ml-3">Processing...</span>
            </div>
            <div className="p-4 max-h-80 overflow-y-auto font-mono text-sm">
              {logs.map((line, i) => (
                <div key={i} className="text-emerald-400 mb-1 leading-relaxed">
                  {line}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Answer Section */}
        {answer && (
          <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-8 mb-8 animate-fadeIn">
            <div className="flex items-center gap-3 mb-4 pb-4 border-b border-slate-200">
              <div className="w-10 h-10 bg-gradient-to-br from-emerald-400 to-teal-500 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-800">Research Results</h2>
            </div>
            <div className="prose prose-lg max-w-none text-slate-700 prose-headings:text-slate-800 prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline prose-strong:text-slate-900">
              <ReactMarkdown
                components={{
                  a: ({ node, ...props }) => (
                    <a {...props} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium" />
                  ),
                }}
              >
                {answer}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {/* Sources Section */}
        {topSnippets.length > 0 && (
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-violet-400 to-purple-500 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-800">Top Sources</h2>
            </div>
            <div className="space-y-4">
              {topSnippets.map((snippet, i) => (
                <div 
                  key={i} 
                  className="bg-white p-6 rounded-xl shadow-md border border-slate-200 hover:shadow-lg hover:border-blue-300 transition-all group"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-lg flex items-center justify-center flex-shrink-0 text-blue-600 font-bold">
                      {i + 1}
                    </div>
                    <div className="flex-1">
                      <a
                        href={snippet.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 font-semibold text-lg hover:text-blue-700 hover:underline group-hover:text-blue-700 transition-colors"
                      >
                        {snippet.title}
                      </a>
                      <p className="text-slate-600 text-sm mt-2 leading-relaxed">
                        {snippet.snippet}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Video Insights Section */}
        {videoChunks.length > 0 && (
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-rose-400 to-pink-500 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-800">Video Insights</h2>
            </div>
            <div className="space-y-6">
              {videoChunks.map((video, i) => (
                <div 
                  key={i} 
                  className="bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden hover:shadow-xl transition-all"
                >
                  <div className="flex flex-col lg:flex-row gap-6 p-6">
                    {/* Video Player */}
                    <div className="flex-shrink-0">
                      <div className="w-full lg:w-[320px] h-[180px] rounded-lg overflow-hidden shadow-md">
                        <div id={`player-${i}`} className="w-full h-full" />
                      </div>
                    </div>

                    {/* Video Info */}
                    <div className="flex-1">
                      <h3 className="text-lg font-bold text-slate-800 mb-2">
                        {video.title}
                      </h3>
                      <p className="text-sm italic text-slate-600 mb-4 leading-relaxed">
                        {video.summary}
                      </p>
                      
                      {/* Key Moments */}
                      {video.moments && video.moments.length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Key Moments
                          </h4>
                          <div className="space-y-2">
                            {video.moments.map((moment, j) => (
                              <button
                                key={j}
                                className="w-full text-left px-3 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg transition-colors flex items-start gap-2 group"
                                onClick={() => {
                                  const player = players[i];
                                  if (player && player.seekTo) {
                                    player.seekTo(moment.start, true);
                                    player.playVideo();
                                  }
                                }}
                              >
                                <span className="font-mono font-medium text-slate-500 group-hover:text-blue-600 transition-colors">
                                  {formatSeconds(moment.start)}
                                </span>
                                <span className="flex-1 group-hover:underline">
                                  {moment.summary}
                                </span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!loading && !answer && !error && (
          <div className="text-center py-20">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-slate-800 mb-2">
              Ready to Research
            </h3>
            <p className="text-slate-600 max-w-md mx-auto">
              Enter your question above and I'll search the web, analyze sources, and provide you with comprehensive insights.
            </p>
          </div>
        )}
      </div>

      {/* Email Modal */}
      {showEmailModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 animate-fadeIn">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full transform animate-scaleIn">
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 rounded-t-2xl flex items-center justify-between">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                Email Report
              </h3>
              <button
                onClick={() => {
                  setShowEmailModal(false);
                  setEmail("");
                }}
                className="text-white hover:bg-white hover:bg-opacity-20 rounded-lg p-1 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="p-6">
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                Email Address
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                className="w-full px-4 py-3 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none focus:ring-4 focus:ring-blue-100 transition-all"
                autoFocus
              />
              <p className="text-sm text-slate-500 mt-2">
                We'll send a PDF report with your research results to this email address.
              </p>
            </div>

            <div className="px-6 pb-6 flex gap-3">
              <button
                onClick={() => {
                  setShowEmailModal(false);
                  setEmail("");
                }}
                className="flex-1 px-4 py-3 border-2 border-slate-200 text-slate-700 rounded-lg font-medium hover:bg-slate-50 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleEmailSend}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all"
              >
                Send Report
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Feedback Modal */}
      {showFeedbackModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 animate-fadeIn">
          <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full transform animate-scaleIn">
            <div className="bg-gradient-to-r from-slate-700 to-slate-800 px-6 py-4 rounded-t-2xl flex items-center justify-between">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                </svg>
                Send Feedback
              </h3>
              <button
                onClick={() => {
                  setShowFeedbackModal(false);
                  setFeedbackText("");
                }}
                className="text-white hover:bg-white hover:bg-opacity-20 rounded-lg p-1 transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="p-6">
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                Your Feedback
              </label>
              <textarea
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                placeholder="Share your thoughts, report bugs, or suggest improvements..."
                className="w-full px-4 py-3 border-2 border-slate-200 rounded-lg focus:border-slate-500 focus:outline-none focus:ring-4 focus:ring-slate-100 transition-all resize-none"
                rows={6}
                autoFocus
              />
              <p className="text-sm text-slate-500 mt-2">
                Your feedback helps us improve the AI Research Assistant. Thank you!
              </p>
            </div>

            <div className="px-6 pb-6 flex gap-3">
              <button
                onClick={() => {
                  setShowFeedbackModal(false);
                  setFeedbackText("");
                }}
                className="flex-1 px-4 py-3 border-2 border-slate-200 text-slate-700 rounded-lg font-medium hover:bg-slate-50 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleFeedbackSend}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white rounded-lg font-medium shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all"
              >
                Send Feedback
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Custom Styles */}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes slideIn {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        
        @keyframes scaleIn {
          from {
            transform: scale(0.9);
            opacity: 0;
          }
          to {
            transform: scale(1);
            opacity: 1;
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
        
        .animate-slideIn {
          animation: slideIn 0.3s ease-out;
        }
        
        .animate-scaleIn {
          animation: scaleIn 0.2s ease-out;
        }
      `}</style>
    </div>
  );
}

export default App;