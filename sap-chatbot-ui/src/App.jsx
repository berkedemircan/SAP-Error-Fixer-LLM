import React, { useState, useRef, useEffect } from "react";

export default function App() {
    const [input, setInput] = useState("");
    const [messages, setMessages] = useState([]);
    const listRef = useRef(null);
    const API_URL = "http://localhost:8000/generate";
    const [loading, setLoading] = useState(false);
    const [menuOpen, setMenuOpen] = useState(false);

    // New Message-bottom
    useEffect(() => {
        if (listRef.current) {
            listRef.current.scrollTop = listRef.current.scrollHeight;
        }
    }, [messages]);

    // Locking Scroll via Menu
    useEffect(() => {
        const prev = document.body.style.overflow;
        document.body.style.overflow = menuOpen ? "hidden" : prev;
        return () => {
            document.body.style.overflow = prev;
        };
    }, [menuOpen]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userText = input.trim();
        setMessages((prev) => [...prev, { role: "user", text: userText }]);
        setInput("");

        setLoading(true);
        try {
            const res = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: userText }),
            });

            const data = await res.json();

            setMessages((prev) => [
                ...prev,
                { role: "assistant", text: data.response ?? "Boş yanıt." },
            ]);
        } catch (err) {
            console.error(err);
            setMessages((prev) => [
                ...prev,
                { role: "assistant", text: "Sunucuya bağlanırken bir hata oluştu." },
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                minHeight: "100vh",
            }}
        >
            {/* === Drop down menu : BUTON + OVERLAY + DRAWER === */}
            <button
                onClick={() => setMenuOpen((v) => !v)}
                aria-label={menuOpen ? "Menüyü kapat" : "Menüyü aç"}
                style={{
                    border: "none",
                    position: "fixed",
                    background:"transparent",
                    top: 16,
                    right: 16,
                    zIndex: 100001,
                    width: 44,
                    height: 44,
                    borderRadius: 16,
                    display: "grid",
                    placeItems: "center",
                    cursor: "pointer",
                    padding:0,
                    outline:"none",
                    boxShadow:"none",

                }}

                //Hover

                onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = "rgba(255,255,255,0.08)")
                }
                onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = "transparent")
                }




            >
        <span style={{ position: "relative", width: 22, height: 16 }}>
          <span
              style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  width: "100%",
                  height: 2,
                  background: "#ffffff",
                  transform: menuOpen ? "translateY(6px) rotate(45deg)" : "none",
                  transition: "transform .25s",
              }}
          />
          <span
              style={{
                  position: "absolute",
                  left: 0,
                  top: "50%",
                  width: "100%",
                  height: 2,
                  background: "#ffffff",
                  transform: "translateY(-50%)",
                  opacity: menuOpen ? 0 : 1,
                  transition: "opacity .2s",
              }}
          />
          <span
              style={{
                  position: "absolute",
                  left: 0,
                  bottom: 0,
                  width: "100%",
                  height: 2,
                  background: "#ffffff",
                  transform: menuOpen ? "translateY(-6px) rotate(-45deg)" : "none",
                  transition: "transform .25s",
              }}
          />
        </span>

            </button>

            <div
                onClick={() => setMenuOpen(false)}
                style={{
                    position: "fixed",
                    inset: 0,
                    background: "rgba(0,0,0,.4)",
                    opacity: menuOpen ? 1 : 0,
                    pointerEvents: menuOpen ? "auto" : "none",
                    transition: "opacity .2s",
                    zIndex: 100000,
                }}
            />

            <aside
                role="dialog"
                aria-modal="true"
                aria-label="Sağ menü"
                style={{
                    position: "fixed",
                    top: 0,
                    right: 0,
                    height: "100vh",
                    width: "92vw",
                    maxWidth: 380,
                    background: "#535151",
                    borderLeft: "1px solid #e5e7eb",
                    boxShadow: "-8px 0 24px rgba(0,0,0,.08)",
                    transform: menuOpen ? "translateX(0)" : "translateX(100%)",
                    transition: "transform .28s cubic-bezier(.2,.8,.2,1)",
                    zIndex: 100002,
                    display: "flex",
                    flexDirection: "column",
                    borderTopLeftRadius: 16,
                    borderBottomLeftRadius: 16,
                    overflow: "hidden",
                }}
            >
                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "12px 16px",
                        borderBottom: "1px solid #e5e7eb",
                    }}
                >
                    <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>Menu</h2>
                    <button
                        onClick={() => setMenuOpen(false)}
                        aria-label="Menüyü kapat"
                        style={{
                            border: "1px solid #535151",
                            background: "#535151",
                            padding: 6,
                            borderRadius: 10,
                            cursor: "pointer",
                            fontSize: 14,
                            width: 36,
                            height: 36,
                            display: "grid",
                            placeItems: "center",
                            outline: "none", // removing blue outline
                            boxShadow: "none",
                            transition: "background-color 0.2s ease, box-shadow 0.2s ease",
                        }}
                        onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#3f3f3f")} // hover
                        onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#535151")} // normal
                        onFocus={(e) =>
                            (e.currentTarget.style.boxShadow = "0 0 0 3px rgba(0,0,0,0.15)") // focus shadow
                        }
                        onBlur={(e) => (e.currentTarget.style.boxShadow = "none")} // focus removing
                    >
  <span style={{ position: "relative", width: 16, height: 16 }}>
    <span
        style={{
            position: "absolute",
            left: 0,
            top: "50%",
            width: "100%",
            height: 2,
            background: "#fff",
            transform: "rotate(45deg)",
            transformOrigin: "center",
        }}
    />
    <span
        style={{
            position: "absolute",
            left: 0,
            top: "50%",
            width: "100%",
            height: 2,
            background: "#fff",
            transform: "rotate(-45deg)",
            transformOrigin: "center",
        }}
    />
  </span>
                    </button>

                </div>

                <nav style={{ padding: 12, overflowY: "auto", flex: 1 }}>
                    <a href="#" style={linkStyle}>
                        Home Page
                    </a>
                    <a href="#" style={linkStyle}>
                        Models
                    </a>

                    <div style={{ padding: 12, display: "grid", gap: 16 }}>
                        <button
                            type="button"
                            style={{
                                border: "1px solid #e5e7eb",
                                background: "#fff",
                                color: "#111827",
                                borderRadius: 20,
                                padding: "8px 12px",
                                fontSize: 14,
                                cursor: "pointer",
                                width: "50%",
                            }}
                        >
                            Log In
                        </button>
                        <button
                            type="button"
                            style={{
                                border: "1px solid #111827",
                                background: "#ffffff",
                                color: "#000100",
                                borderRadius: 20,
                                padding: "8px 12px",
                                fontSize: 14,
                                cursor: "pointer",
                                width: "50%",
                            }}
                        >
                            Sign Up
                        </button>
                    </div>

                </nav>






                <div
                    style={{
                        borderTop: "1px solid #e5e7eb",
                        padding: 12,
                        fontSize: 13,
                        color: "#6b7280",
                        textAlign: "center",
                    }}
                >
                    © {new Date().getFullYear()} — LLM Panel
                </div>
            </aside>
            {/* === /RIGHT MENU === */}

            <div
                style={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    height: "100vh",
                    maxWidth: 600,
                    margin: "0 auto",
                    padding: "1rem",
                }}
            >
                {/* TOP SIDE  */}
                <div
                    style={{
                        position: "fixed",
                        top: 0,
                        left: 0,
                        width: "95%",
                        display: "flex",
                        justifyContent: "left",
                        alignItems: "center",
                        padding: "1rem",
                        zIndex: 1000,
                    }}
                >
                    <h1 style={{ margin: 0 }}>SAP Log Fixer</h1>
                </div>

                {/* MIDDLE SIDE */}
                <div
                    ref={listRef}
                    style={{
                        position: "fixed",
                        top: 80, // top  fixed header height
                        bottom: 100, // bottom  fixed form height
                        left: "50%",
                        transform: "translateX(-50%)",
                        width: "95%",
                        maxWidth: 600,
                        overflowY: "auto",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: messages.length === 0 ? "center" : "flex-start",
                        alignItems: "stretch",
                        gap: 10,
                        padding: "0 4px",
                    }}
                >
                    {messages.map((m, i) => (
                        <div
                            key={i}
                            style={{
                                display: "flex",
                                justifyContent: m.role === "user" ? "flex-end" : "flex-start",
                            }}
                        >
                            <div
                                style={{
                                    maxWidth: "85%",
                                    padding: "10px 12px",
                                    borderRadius: 14,
                                    lineHeight: 1.4,
                                    fontSize: 14,
                                    wordBreak: "break-word",
                                    background: m.role === "user" ? "#a3a3a4" : "#a3a3a4",
                                    ...(m.role === "user"
                                        ? { borderTopRightRadius: 6 }
                                        : { borderTopLeftRadius: 6 }),
                                }}
                            >
                                {m.text}
                            </div>
                        </div>
                    ))}

                    {loading && (
                        <div
                            style={{
                                alignSelf: "flex-start",
                                background: "#F1F1F1",
                                color: "#666",
                                fontSize: 12,
                                padding: "6px 10px",
                                borderRadius: 12,
                                borderTopLeftRadius: 6,
                                maxWidth: "60%",
                            }}
                        >
                            Error is fixing...
                        </div>
                    )}
                </div>

                {/* BOTTOM SIDE */}
                <form
                    onSubmit={handleSubmit}
                    style={{
                        position: "fixed",
                        bottom: "20px",
                        left: "50%",
                        transform: "translateX(-50%)",
                        width: "80%",
                        maxWidth: "600px",
                    }}
                >
                    <div style={{ position: "relative" }}>
            <textarea
                rows={2}
                placeholder="Enter your error log..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                    }
                }}
                style={{
                    width: "100%",
                    padding: "0.75rem 3rem 0.75rem 1rem",
                    borderRadius: "14px",
                    border: "1px solid #717173",
                    resize: "none",
                    boxSizing: "border-box",
                    fontSize: "16px",
                    outline: "none",
                    backgroundColor: "#272729",
                    boxShadow: "none",
                    transition: "all 0.2s ease",
                    color: "#fff",
                }}
            />

                        <button
                            type="submit"
                            disabled={loading}
                            title={loading ? "Sending..." : "Send"}
                            style={{
                                position: "absolute",
                                right: "5px",
                                bottom: "12px",
                                background: "#808080",
                                color: "#fff",
                                border: "none",
                                borderRadius: "50%",
                                width: "30px",
                                height: "30px",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontSize: "14px",
                                padding: 0,
                                transition: "background-color 0.3s ease",
                            }}
                            onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#666")}
                            onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#808080")}
                        >
                            {loading ? "…" : "⬆"}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

// out of component
const linkStyle = {
    display: "block",
    padding: "10px 12px",
    borderRadius: 12,
    color: "#ffffff",
    textDecoration: "none",
    marginBottom: 6,
    transition: "background .2s",
    width:"25%"
};
