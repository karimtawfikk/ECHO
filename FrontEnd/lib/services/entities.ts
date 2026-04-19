// ─── Frontend API for fetching trending entities from the DB ─────────────────

import type { RecognitionEntity } from "../types";


const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8010";

export interface TrendingEntitiesResponse {
    pharaohs: RecognitionEntity[];
    landmarks: RecognitionEntity[];
    error?: string;
}

export async function fetchTrendingEntities(): Promise<TrendingEntitiesResponse> {
    const res = await fetch(`${API_BASE}/api/v1/entities/trending?limit=5`, {
        // no-store so we always get fresh data on each client-side visit
        cache: "no-store",
    });
    if (!res.ok) throw new Error(`Entities API error ${res.status}`);
    return res.json();
}
