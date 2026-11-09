export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = response.statusText || `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { error?: string; detail?: string };
      message = body.error || body.detail || message;
    } catch {
      // keep the status text
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Multipart upload with real upload-byte progress (fetch cannot report it). */
export function uploadForm<T>(
  path: string,
  body: FormData,
  onProgress?: (fraction: number) => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", path);
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress?.(event.loaded / event.total);
      }
    };
    request.onload = () => {
      const payload = request.response as { error?: string; detail?: string } | null;
      if (request.status >= 200 && request.status < 300) {
        onProgress?.(1);
        resolve(payload as T);
        return;
      }
      reject(
        new ApiError(
          payload?.error || payload?.detail || request.statusText || `HTTP ${request.status}`,
          request.status,
        ),
      );
    };
    request.onerror = () => reject(new ApiError("网络错误，请重试", 0));
    request.onabort = () => reject(new ApiError("上传已取消", 0));
    request.send(body);
  });
}

export function formatError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--:--";
  }
  const safe = Math.max(0, value);
  const minutes = Math.floor(safe / 60);
  const seconds = Math.floor(safe % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
