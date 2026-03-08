import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
    getCacheStats,
    postQuery,
    flushCache,
    QueryResponse,
    CacheStats
} from "./api";

interface HistoryItem {
    query: string;
    cache_hit: boolean;
    timestamp: string;
}

function App() {
    const [query, setQuery] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [response, setResponse] = useState<QueryResponse | null>(null);
    const [stats, setStats] = useState<CacheStats | null>(null);
    const [history, setHistory] = useState<HistoryItem[]>([]);

    const fetchStats = async () => {
        try {
            const s = await getCacheStats();
            setStats(s);
        } catch (e) {
            console.error(e);
        }
    };

    useEffect(() => {
        fetchStats();
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const trimmed = query.trim();
        if (!trimmed) return;
        setLoading(true);
        setError(null);

        try {
            const res = await postQuery(trimmed);
            setResponse(res);
            setHistory(prev => [
                {
                    query: trimmed,
                    cache_hit: res.cache_hit,
                    timestamp: new Date().toISOString()
                },
                ...prev.slice(0, 9)
            ]);
            fetchStats();
        } catch (e: any) {
            console.error(e);
            setError(e?.response?.data?.detail || "Something went wrong");
        } finally {
            setLoading(false);
        }
    };

    const handleFlush = async () => {
        await flushCache();
        setResponse(null);
        setHistory([]);
        fetchStats();
    };

    return (
        <div className="min-h-screen flex flex-col">
            <main className="flex-1 flex flex-col lg:flex-row max-w-6xl mx-auto px-4 py-8 gap-8">
                <section className="flex-1 flex flex-col">
                    <header className="mb-6">
                        <h1 className="text-3xl font-semibold tracking-tight">
                            Trademarkia Semantic Explorer
                        </h1>
                        <p className="text-slate-400 mt-1">
                            Fuzzy-clustered semantic search with an intelligent cache. Type a
                            query and watch hits appear.
                        </p>
                    </header>

                    <form onSubmit={handleSubmit} className="relative mb-6">
                        <div className="relative group">
                            <input
                                value={query}
                                onChange={e => setQuery(e.target.value)}
                                placeholder="Ask about politics, religion, sports, hardware, gun control, space…"
                                className="w-full rounded-2xl bg-card/80 border border-slate-700/60 px-4 py-3 pr-32
                           text-slate-100 placeholder:text-slate-500
                           shadow-[0_0_0_1px_rgba(148,163,184,.2)] focus:outline-none
                           focus:border-accent focus:shadow-[0_0_0_1px_rgba(34,211,238,.6)]
                           transition-all"
                            />
                            <button
                                type="submit"
                                disabled={loading}
                                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-xl bg-accent text-slate-900
                           px-4 py-1.5 text-sm font-medium shadow-md hover:shadow-lg
                           disabled:opacity-60 disabled:cursor-not-allowed"
                            >
                                {loading ? "Searching…" : "Search"}
                            </button>
                        </div>
                    </form>

                    {error && (
                        <div className="mb-4 rounded-xl border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-100">
                            {error}
                        </div>
                    )}

                    <AnimatePresence mode="wait">
                        {response ? (
                            <motion.div
                                key={response.query + String(response.cache_hit)}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -8 }}
                                className="flex-1 rounded-2xl bg-card/80 border border-slate-700/70 px-4 py-4 overflow-auto"
                            >
                                <div className="flex items-center justify-between mb-3 gap-2">
                                    <div className="flex items-center gap-2 text-sm">
                                        <span
                                            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium
                        ${response.cache_hit
                                                    ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/40"
                                                    : "bg-sky-500/10 text-sky-300 border border-sky-500/40"
                                                }`}
                                        >
                                            <span
                                                className={`h-1.5 w-1.5 rounded-full ${response.cache_hit ? "bg-emerald-400" : "bg-sky-400"
                                                    }`}
                                            />
                                            {response.cache_hit ? "Cache hit" : "Cache miss"}
                                        </span>
                                        <span className="text-slate-500">
                                            sim={response.similarity_score.toFixed(3)} · cluster{" "}
                                            <span className="font-mono">
                                                {response.dominant_cluster}
                                            </span>
                                        </span>
                                    </div>
                                </div>

                                {response.matched_query &&
                                    response.matched_query !== response.query && (
                                        <p className="text-xs text-slate-400 mb-2">
                                            Matched cached query:
                                            <span className="ml-1 font-mono text-slate-200">
                                                “{response.matched_query}”
                                            </span>
                                        </p>
                                    )}

                                <pre className="mt-2 text-xs sm:text-sm font-mono whitespace-pre-wrap text-slate-100/90">
                                    {response.result}
                                </pre>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="empty"
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex-1 rounded-2xl border border-dashed border-slate-700/70
                           bg-card/40 px-4 py-6 text-sm text-slate-500 flex items-center justify-center text-center"
                            >
                                Type a query to see top semantic matches and live cache
                                behaviour.
                            </motion.div>
                        )}
                    </AnimatePresence>
                </section>

                <aside className="w-full lg:w-80 flex flex-col gap-4">
                    <div className="rounded-2xl bg-card/80 border border-slate-700/70 px-4 py-4">
                        <div className="flex items-center justify-between mb-2">
                            <h2 className="text-sm font-semibold text-slate-100">
                                Cache stats
                            </h2>
                            <button
                                onClick={handleFlush}
                                className="text-xs text-slate-400 hover:text-rose-300 transition-colors"
                            >
                                Flush
                            </button>
                        </div>
                        {stats ? (
                            <div className="space-y-1 text-xs text-slate-300">
                                <div className="flex justify-between">
                                    <span>Total entries</span>
                                    <span className="font-mono">{stats.total_entries}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Hits</span>
                                    <span className="font-mono text-emerald-300">
                                        {stats.hit_count}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Misses</span>
                                    <span className="font-mono text-rose-300">
                                        {stats.miss_count}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Hit rate</span>
                                    <span className="font-mono">
                                        {(stats.hit_rate * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Similarity τ</span>
                                    <span className="font-mono">
                                        {stats.similarity_threshold.toFixed(2)}
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <p className="text-xs text-slate-500">Loading stats…</p>
                        )}
                    </div>

                    <div className="rounded-2xl bg-card/80 border border-slate-700/70 px-4 py-4 flex-1">
                        <h2 className="text-sm font-semibold text-slate-100 mb-2">
                            Recent queries
                        </h2>
                        {history.length === 0 ? (
                            <p className="text-xs text-slate-500">No queries yet.</p>
                        ) : (
                            <ul className="space-y-2 text-xs">
                                {history.map(h => (
                                    <li
                                        key={h.timestamp + h.query}
                                        className="flex items-start justify-between gap-2"
                                    >
                                        <span className="flex-1 truncate text-slate-300">
                                            {h.query}
                                        </span>
                                        <span
                                            className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium
                        ${h.cache_hit
                                                    ? "bg-emerald-500/10 text-emerald-300"
                                                    : "bg-sky-500/10 text-sky-300"
                                                }`}
                                        >
                                            {h.cache_hit ? "hit" : "miss"}
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                </aside>
            </main>

            <footer className="border-t border-slate-800/60 py-3 text-center text-xs text-slate-500">
                Built for Trademarkia — 20 Newsgroups semantic search with fuzzy
                clustering & cache.
            </footer>
        </div>
    );
}

export default App;
