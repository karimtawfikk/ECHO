
import { RecognitionResult, PendingResult } from "../types";

const SESSION_KEY = "echo_recognition_result";

const API_BASE =
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/api\/v1\/?$/, "") ?? "http://localhost:8010";

export async function recognizeImage(
    file: File
): Promise<RecognitionResult> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(
        `${API_BASE}/api/v1/recognize/?debug=false`,
        {
            method: "POST",
            body: formData,
        }
    );

    if (!res.ok) {
        const text = await res.text();
        throw new Error(`Recognition API error ${res.status}: ${text}`);
    }

    return res.json() as Promise<RecognitionResult>;
}


export function formatTitle(name: string): string {
    return name.replace(/_/g, " ");
}

export function saveResultToSession(payload: PendingResult): void {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(payload));
}

export function loadResultFromSession(): PendingResult | null {
    try {
        const raw = sessionStorage.getItem(SESSION_KEY);
        if (!raw) return null;
        return JSON.parse(raw) as PendingResult;
    } catch {
        return null;
    }
}

export function clearResultSession(): void {
    sessionStorage.removeItem(SESSION_KEY);
}
