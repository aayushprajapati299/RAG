import * as React from "react";
import {
  DocumentItem,
  DocumentDetail,
  IngestionJob,
  documentsApi,
  chatApi,
  ApiError,
} from "@/lib/api";

export type UploadStatus = "idle" | "uploading" | "processing" | "completed" | "error";

export function useDocuments(
  conversationId?: string | null,
  initialSelectedDocId?: string | null
) {
  const [documents, setDocuments] = React.useState<DocumentItem[]>([]);
  const [allUserDocuments, setAllUserDocuments] = React.useState<DocumentItem[]>([]);
  const [selectedDocument, setSelectedDocument] = React.useState<DocumentItem | null>(null);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [error, setError] = React.useState<string | null>(null);

  // Background ingestion job tracking
  const [activeJob, setActiveJob] = React.useState<IngestionJob | null>(null);
  const [uploadState, setUploadState] = React.useState<UploadStatus>("idle");
  const [uploadError, setUploadError] = React.useState<string | null>(null);

  const pollingRef = React.useRef<number | null>(null);
  const isMountedRef = React.useRef<boolean>(true);

  // Stop polling helper
  const stopPolling = React.useCallback(() => {
    if (pollingRef.current !== null) {
      window.clearTimeout(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  // Fetch document list for the active conversation + global user docs
  const fetchDocuments = React.useCallback(
    async (keepSelected = true) => {
      try {
        setError(null);
        // Load all documents owned by user for attachment support
        const allDocs = await documentsApi.list();
        if (!isMountedRef.current) return;
        setAllUserDocuments(allDocs);

        // Load documents scoped to current conversation (if one is active)
        let chatDocs: DocumentItem[] = [];
        if (conversationId) {
          try {
            chatDocs = await documentsApi.list(conversationId);
          } catch {
            // If conversation has no docs or is new
            chatDocs = [];
          }
        } else {
          chatDocs = allDocs;
        }

        if (!isMountedRef.current) return;
        setDocuments(chatDocs);

        // Determine active selection
        setSelectedDocument((prev) => {
          if (keepSelected && prev) {
            const match = chatDocs.find((d) => d.id === prev.id);
            if (match) return match;
          }
          if (initialSelectedDocId) {
            const match = chatDocs.find((d) => d.id === initialSelectedDocId);
            if (match) return match;
          }
          return chatDocs.length > 0 ? chatDocs[0] : null;
        });
      } catch (err) {
        if (!isMountedRef.current) return;
        const msg = err instanceof ApiError ? err.message : "Failed to load documents.";
        setError(msg);
      } finally {
        if (isMountedRef.current) {
          setIsLoading(false);
        }
      }
    },
    [conversationId, initialSelectedDocId]
  );

  // Synchronize when conversationId changes
  React.useEffect(() => {
    isMountedRef.current = true;
    setIsLoading(true);
    fetchDocuments(false);

    return () => {
      isMountedRef.current = false;
      stopPolling();
    };
  }, [conversationId, fetchDocuments, stopPolling]);

  // Select document and persist to conversation
  const selectDocument = React.useCallback(
    async (doc: DocumentItem | null) => {
      setSelectedDocument(doc);
      if (conversationId && doc) {
        try {
          await chatApi.selectDocument(conversationId, doc.id);
        } catch {
          // Non-critical background sync
        }
      }
    },
    [conversationId]
  );

  // Delete document
  const deleteDocument = React.useCallback(
    async (docId: string) => {
      await documentsApi.delete(docId);
      if (selectedDocument?.id === docId) {
        setSelectedDocument(null);
      }
      await fetchDocuments(false);
    },
    [selectedDocument, fetchDocuments]
  );

  // Attach existing document from library to active conversation
  const attachDocument = React.useCallback(
    async (docId: string) => {
      if (!conversationId) return;
      await chatApi.attachDocument(conversationId, docId);
      await fetchDocuments(true);
    },
    [conversationId, fetchDocuments]
  );

  // Poll IngestionJob status endpoint
  const pollJobStatus = React.useCallback(
    (documentId: string) => {
      stopPolling();

      const poll = async () => {
        try {
          const job = await documentsApi.getStatus(documentId);
          if (!isMountedRef.current) return;

          setActiveJob(job);

          if (job.status === "completed") {
            setUploadState("completed");
            stopPolling();
            // Refresh conversation documents
            await fetchDocuments(false);
            if (isMountedRef.current) {
              const refreshed = await documentsApi.list(conversationId || undefined);
              const found = refreshed.find((d) => d.id === documentId);
              if (found) {
                setSelectedDocument(found);
                if (conversationId) {
                  chatApi.selectDocument(conversationId, found.id).catch(() => {});
                }
              }
            }
          } else if (job.status === "failed") {
            setUploadState("error");
            setUploadError(job.error_message || "Document processing failed. Please verify file integrity.");
            stopPolling();
            await fetchDocuments(true).catch(() => {});
          } else {
            pollingRef.current = window.setTimeout(poll, 1500);
          }
        } catch (err) {
          if (!isMountedRef.current) return;
          const msg = err instanceof ApiError ? err.message : "Unable to verify document status.";
          setUploadState("error");
          setUploadError(msg);
          stopPolling();
        }
      };

      pollingRef.current = window.setTimeout(poll, 1500);
    },
    [conversationId, fetchDocuments, stopPolling]
  );

  // Upload handler
  const uploadDocument = React.useCallback(
    async (file: File): Promise<DocumentDetail> => {
      setUploadError(null);
      stopPolling();

      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
      if (![".pdf", ".txt"].includes(ext)) {
        const msg = "Unsupported file type. Only PDF and TXT files are accepted.";
        setUploadState("error");
        setUploadError(msg);
        throw new Error(msg);
      }

      const MAX_SIZE = 25 * 1024 * 1024; // 25 MB
      if (file.size > MAX_SIZE) {
        const msg = `File size exceeds the 25 MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`;
        setUploadState("error");
        setUploadError(msg);
        throw new Error(msg);
      }

      setUploadState("uploading");

      try {
        const detail = await documentsApi.upload(file, conversationId || undefined);
        if (!isMountedRef.current) return detail;

        setActiveJob(detail.latest_job || null);
        setUploadState("processing");

        // Start polling background ingestion job
        pollJobStatus(detail.id);

        return detail;
      } catch (err) {
        if (!isMountedRef.current) throw err;
        const msg = err instanceof ApiError ? err.message : "Failed to upload document.";
        setUploadState("error");
        setUploadError(msg);
        throw err;
      }
    },
    [conversationId, pollJobStatus, stopPolling]
  );

  const resetUploadState = React.useCallback(() => {
    stopPolling();
    setUploadState("idle");
    setUploadError(null);
    setActiveJob(null);
  }, [stopPolling]);

  return {
    documents,
    allUserDocuments,
    selectedDocument,
    selectDocument,
    deleteDocument,
    attachDocument,
    isLoading,
    error,
    activeJob,
    uploadState,
    uploadError,
    uploadDocument,
    resetUploadState,
    refreshDocuments: fetchDocuments,
  };
}
