import axios from "axios";

const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL
});

export interface QueryResponse {
    query: string;
    cache_hit: boolean;
    matched_query: string | null;
    similarity_score: number;
    result: string;
    dominant_cluster: number;
}

export interface CacheStats {
    total_entries: number;
    hit_count: number;
    miss_count: number;
    hit_rate: number;
    similarity_threshold: number;
}

export async function postQuery(query: string): Promise<QueryResponse> {
    const { data } = await api.post<QueryResponse>("/query", { query });
    return data;
}

export async function getCacheStats(): Promise<CacheStats> {
    const { data } = await api.get<CacheStats>("/cache/stats");
    return data;
}

export async function flushCache(): Promise<void> {
    await api.delete("/cache");
}
