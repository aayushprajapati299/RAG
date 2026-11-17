/**
 * API client library with credentials support, CSRF token management,
 * and structured error handling.
 */

export interface User {
  id: string;
  email: string;
  full_name?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface UserLoginRequest {
  email: string;
  password: string;
}

export interface UserRegisterRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface CsrfTokenResponse {
  csrf_token: string;
}

export interface MessageResponse {
  message: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes?: number | null;
  page_count: number;
  created_at: string;
  updated_at: string;
}

export interface IngestionJob {
  id: string;
  document_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  error_message?: string | null;
  pages_processed: number;
  total_pages: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentDetail extends DocumentItem {
  latest_job?: IngestionJob | null;
  job_id?: string | null;
  document_id?: string | null;
}

export interface Citation {
  chunk_id: string;
  page_number: number;
  text_snippet?: string | null;
  score?: number | null;
}

export interface ChunkItem {
  id: string;
  document_id: string;
  page_number: number;
  chunk_index: number;
  content: string;
  char_count: number;
  created_at: string;
}

export interface MessageItem {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: Citation[] | null;
  created_at: string;
}

export interface ConversationItem {
  id: string;
  user_id: string;
  title: string;
  selected_document_id?: string | null;
  created_at: string;
  updated_at: string;
  message_count?: number | null;
}

export interface ConversationDetail extends ConversationItem {
  documents: DocumentItem[];
  messages: MessageItem[];
}

export interface ChatAnswerResponse {
  conversation_id: string;
  answer: string;
  citations: Citation[];
  user_message: MessageItem;
  assistant_message: MessageItem;
}

export interface StreamCompletePayload {
  message_id: string;
  conversation_id: string;
  answer: string;
  citations: Citation[];
  finish_reason: string;
}

export interface StreamCallbacks {
  onDelta?: (delta: string) => void;
  onComplete?: (data: StreamCompletePayload) => void;
  onError?: (error: Error) => void;
}

export class ApiError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

// In-memory CSRF token cache (never stored in localStorage)
let cachedCsrfToken: string | null = null;

export function setCachedCsrfToken(token: string | null): void {
  cachedCsrfToken = token;
}

export function getCachedCsrfToken(): string | null {
  return cachedCsrfToken;
}

export async function fetchCsrfToken(): Promise<string> {
  const res = await fetch("/api/auth/csrf", {
    method: "GET",
    credentials: "include",
    headers: {
      Accept: "application/json",
    },
  });

  if (!res.ok) {
    throw new ApiError("Failed to retrieve CSRF token", res.status);
  }

  const data: CsrfTokenResponse = await res.json();
  cachedCsrfToken = data.csrf_token;
  return data.csrf_token;
}

export async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const method = (options.method || "GET").toUpperCase();
  const isStateChanging = ["POST", "PUT", "DELETE", "PATCH"].includes(method);

  const headers = new Headers(options.headers || {});
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }

  // Ensure JSON Content-Type for request bodies that aren't FormData
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  // For state-changing requests, ensure we have a CSRF token
  if (isStateChanging) {
    if (!cachedCsrfToken) {
      try {
        await fetchCsrfToken();
      } catch {
        // If getting CSRF token fails, continue; backend might reject with 403
      }
    }
    if (cachedCsrfToken && !headers.has("X-CSRF-Token")) {
      headers.set("X-CSRF-Token", cachedCsrfToken);
    }
  }

  let res = await fetch(endpoint, {
    ...options,
    headers,
    credentials: "include", // Always include session cookies
  });

  // If CSRF token was invalid/expired (HTTP 403), refresh it and retry once
  if (res.status === 403 && isStateChanging) {
    try {
      const newToken = await fetchCsrfToken();
      headers.set("X-CSRF-Token", newToken);
      res = await fetch(endpoint, {
        ...options,
        headers,
        credentials: "include",
      });
    } catch {
      // Retry failed; proceed with original response handling
    }
  }

  // Handle errors
  if (!res.ok) {
    let errorMessage = `Request failed with status ${res.status}`;
    let errorDetails: unknown = null;

    try {
      const data = await res.json();
      errorDetails = data;
      if (typeof data.detail === "string") {
        errorMessage = data.detail;
      } else if (Array.isArray(data.detail)) {
        // Pydantic validation error list
        errorMessage = data.detail
          .map((err: { msg?: string; loc?: string[] }) => {
            const field = err.loc ? err.loc[err.loc.length - 1] : "";
            return field ? `${field}: ${err.msg}` : err.msg || "Validation error";
          })
          .join(", ");
      } else if (typeof data.message === "string") {
        errorMessage = data.message;
      } else if (data.error && typeof data.error === "string") {
        errorMessage = data.error;
      }
    } catch {
      try {
        const text = await res.text();
        if (text) errorMessage = text.slice(0, 200);
      } catch {
        // Ignore text parsing failure
      }
    }

    throw new ApiError(errorMessage, res.status, errorDetails);
  }

  if (res.status === 204) {
    return {} as T;
  }

  return res.json() as Promise<T>;
}

// Auth API client
export const authApi = {
  async register(payload: UserRegisterRequest): Promise<User> {
    const user = await apiFetch<User>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await fetchCsrfToken().catch(() => {});
    return user;
  },

  async login(payload: UserLoginRequest): Promise<User> {
    const user = await apiFetch<User>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await fetchCsrfToken().catch(() => {});
    return user;
  },

  async getMe(): Promise<User> {
    return apiFetch<User>("/api/auth/me", {
      method: "GET",
    });
  },

  async logout(): Promise<MessageResponse> {
    const res = await apiFetch<MessageResponse>("/api/auth/logout", {
      method: "POST",
    });
    cachedCsrfToken = null;
    return res;
  },

  async getCsrfToken(): Promise<string> {
    return fetchCsrfToken();
  },
};

// Document API client
export const documentsApi = {
  async list(conversationId?: string): Promise<DocumentItem[]> {
    const url = conversationId
      ? `/api/documents?conversation_id=${encodeURIComponent(conversationId)}`
      : "/api/documents";
    return apiFetch<DocumentItem[]>(url, {
      method: "GET",
    });
  },

  async get(documentId: string): Promise<DocumentDetail> {
    return apiFetch<DocumentDetail>(`/api/documents/${documentId}`, {
      method: "GET",
    });
  },

  async getStatus(documentId: string): Promise<IngestionJob> {
    return apiFetch<IngestionJob>(`/api/documents/${documentId}/status`, {
      method: "GET",
    });
  },

  async upload(file: File, conversationId?: string): Promise<DocumentDetail> {
    const formData = new FormData();
    formData.append("file", file);
    if (conversationId) {
      formData.append("conversation_id", conversationId);
    }

    return apiFetch<DocumentDetail>("/api/documents", {
      method: "POST",
      body: formData,
    });
  },

  async delete(documentId: string): Promise<MessageResponse> {
    return apiFetch<MessageResponse>(`/api/documents/${documentId}`, {
      method: "DELETE",
    });
  },
};

// Chat & Conversation API client
export const chatApi = {
  async listConversations(): Promise<ConversationItem[]> {
    return apiFetch<ConversationItem[]>("/api/chat/conversations", {
      method: "GET",
    });
  },

  async createConversation(
    title?: string,
    selectedDocumentId?: string,
    documentIds?: string[]
  ): Promise<ConversationItem> {
    const payload: Record<string, unknown> = {};
    if (title) payload.title = title;
    if (selectedDocumentId) payload.selected_document_id = selectedDocumentId;
    if (documentIds) payload.document_ids = documentIds;

    return apiFetch<ConversationItem>("/api/chat/conversations", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async getConversation(conversationId: string): Promise<ConversationDetail> {
    return apiFetch<ConversationDetail>(`/api/chat/conversations/${conversationId}`, {
      method: "GET",
    });
  },

  async updateConversation(
    conversationId: string,
    payload: {
      title?: string;
      selected_document_id?: string | null;
      document_ids?: string[];
    }
  ): Promise<ConversationDetail> {
    return apiFetch<ConversationDetail>(`/api/chat/conversations/${conversationId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  },

  async deleteConversation(conversationId: string): Promise<MessageResponse> {
    return apiFetch<MessageResponse>(`/api/chat/conversations/${conversationId}`, {
      method: "DELETE",
    });
  },

  async attachDocument(conversationId: string, documentId: string): Promise<ConversationDetail> {
    return apiFetch<ConversationDetail>(
      `/api/chat/conversations/${conversationId}/documents/${documentId}`,
      { method: "POST" }
    );
  },

  async detachDocument(conversationId: string, documentId: string): Promise<ConversationDetail> {
    return apiFetch<ConversationDetail>(
      `/api/chat/conversations/${conversationId}/documents/${documentId}`,
      { method: "DELETE" }
    );
  },

  async selectDocument(conversationId: string, documentId: string): Promise<ConversationDetail> {
    return apiFetch<ConversationDetail>(
      `/api/chat/conversations/${conversationId}/select-document/${documentId}`,
      { method: "PUT" }
    );
  },

  async getChunk(chunkId: string, documentId?: string): Promise<ChunkItem> {
    const endpoint = documentId
      ? `/api/documents/${documentId}/chunks/${chunkId}`
      : `/api/chunks/${chunkId}`;
    return apiFetch<ChunkItem>(endpoint, {
      method: "GET",
    });
  },

  async postMessage(
    conversationId: string,
    documentId: string,
    content: string
  ): Promise<ChatAnswerResponse> {
    return apiFetch<ChatAnswerResponse>(`/api/chat/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        document_id: documentId,
        content,
        stream: false,
      }),
    });
  },

  async streamMessage(
    conversationId: string,
    documentId: string,
    content: string,
    callbacks: StreamCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    if (!cachedCsrfToken) {
      try {
        await fetchCsrfToken();
      } catch {
        // Continue, backend handles auth/csrf
      }
    }

    const headers = new Headers({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    });
    if (cachedCsrfToken) {
      headers.set("X-CSRF-Token", cachedCsrfToken);
    }

    const payload = JSON.stringify({
      document_id: documentId,
      content,
      stream: true,
    });

    let res = await fetch(`/api/chat/conversations/${conversationId}/messages/stream`, {
      method: "POST",
      headers,
      credentials: "include",
      body: payload,
      signal,
    });

    // If CSRF token was invalid/expired (HTTP 403), refresh it and retry once
    if (res.status === 403) {
      try {
        const newToken = await fetchCsrfToken();
        headers.set("X-CSRF-Token", newToken);
        res = await fetch(`/api/chat/conversations/${conversationId}/messages/stream`, {
          method: "POST",
          headers,
          credentials: "include",
          body: payload,
          signal,
        });
      } catch {
        // Retry failed; fall through to response check
      }
    }

    if (!res.ok) {
      let errorMessage = `Stream request failed with status ${res.status}`;
      try {
        const data = await res.json();
        if (typeof data.detail === "string") {
          errorMessage = data.detail;
        } else if (data.detail && typeof data.detail.message === "string") {
          errorMessage = data.detail.message;
        } else if (typeof data.message === "string") {
          errorMessage = data.message;
        }
      } catch {
        // Non-JSON response
      }
      const error = new ApiError(errorMessage, res.status);
      callbacks.onError?.(error);
      throw error;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      const err = new ApiError("Streaming is not supported by your browser.", 500);
      callbacks.onError?.(err);
      throw err;
    }

    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        // Keep the last segment in buffer in case it is incomplete
        buffer = blocks.pop() || "";

        for (const block of blocks) {
          const trimmed = block.trim();
          if (!trimmed) continue;

          let eventName = "message";
          let dataStr = "";

          const lines = trimmed.split("\n");
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventName = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              dataStr += (dataStr ? "\n" : "") + line.slice(6);
            } else if (line === "data:") {
              dataStr += (dataStr ? "\n" : "");
            }
          }

          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (eventName === "text") {
              callbacks.onDelta?.(parsed.delta || "");
            } else if (eventName === "complete") {
              callbacks.onComplete?.(parsed);
            } else if (eventName === "error") {
              const err = new ApiError(
                parsed.message || parsed.error || "Provider error",
                500,
                parsed
              );
              callbacks.onError?.(err);
              return;
            }
          } catch {
            if (eventName === "text") {
              callbacks.onDelta?.(dataStr);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },
};

