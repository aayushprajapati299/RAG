import * as React from "react";
import {
  chatApi,
  ConversationItem,
  ConversationDetail,
  MessageItem,
  Citation,
  StreamCompletePayload,
} from "@/lib/api";

export interface UseChatReturn {
  conversations: ConversationItem[];
  activeConversation: ConversationItem | null;
  activeConversationId: string | null;
  activeConversationDetail: ConversationDetail | null;
  messages: MessageItem[];
  isLoadingConversations: boolean;
  isLoadingMessages: boolean;
  isStreaming: boolean;
  streamingDelta: string;
  error: string | null;
  selectConversation: (conversationId: string) => Promise<void>;
  createConversation: (title?: string) => Promise<ConversationItem | null>;
  deleteConversation: (conversationId: string) => Promise<void>;
  sendMessage: (documentId: string, content: string) => Promise<void>;
  stopStreaming: () => void;
  clearError: () => void;
  refreshConversations: () => Promise<void>;
}

export function useChat(): UseChatReturn {
  const [conversations, setConversations] = React.useState<ConversationItem[]>([]);
  const [activeConversationId, setActiveConversationId] = React.useState<string | null>(null);
  const [activeConversationDetail, setActiveConversationDetail] = React.useState<ConversationDetail | null>(null);
  const [messages, setMessages] = React.useState<MessageItem[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = React.useState<boolean>(false);
  const [isLoadingMessages, setIsLoadingMessages] = React.useState<boolean>(false);
  const [isStreaming, setIsStreaming] = React.useState<boolean>(false);
  const [streamingDelta, setStreamingDelta] = React.useState<string>("");
  const [error, setError] = React.useState<string | null>(null);

  const abortControllerRef = React.useRef<AbortController | null>(null);
  const streamingDeltaRef = React.useRef<string>("");

  // Keep ref synchronized with delta for stopStreaming closure
  React.useEffect(() => {
    streamingDeltaRef.current = streamingDelta;
  }, [streamingDelta]);

  // Select an existing conversation and fetch its message history & detail
  const selectConversation = React.useCallback(
    async (conversationId: string) => {
      // Abort any ongoing stream before switching
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setIsStreaming(false);
      setStreamingDelta("");
      setError(null);
      setActiveConversationId(conversationId);

      try {
        setIsLoadingMessages(true);
        const detail = await chatApi.getConversation(conversationId);
        setActiveConversationDetail(detail);
        setMessages(detail.messages || []);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to load message history.";
        setError(msg);
      } finally {
        setIsLoadingMessages(false);
      }
    },
    []
  );

  // Load all user-owned conversations
  const refreshConversations = React.useCallback(async () => {
    try {
      setIsLoadingConversations(true);
      const list = await chatApi.listConversations();
      setConversations(list);
      // Auto-select first conversation if none is active
      setActiveConversationId((currentActive) => {
        if (!currentActive && list.length > 0) {
          selectConversation(list[0].id);
          return list[0].id;
        }
        return currentActive;
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load conversations.";
      setError(msg);
    } finally {
      setIsLoadingConversations(false);
    }
  }, [selectConversation]);

  // Initial load
  React.useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  // Create a brand new conversation
  const createConversation = React.useCallback(
    async (title?: string): Promise<ConversationItem | null> => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setIsStreaming(false);
      setStreamingDelta("");
      setError(null);

      try {
        const newConv = await chatApi.createConversation(title);
        setConversations((prev) => [newConv, ...prev.filter((c) => c.id !== newConv.id)]);
        setActiveConversationId(newConv.id);
        setActiveConversationDetail({
          ...newConv,
          documents: [],
          messages: [],
        });
        setMessages([]);
        return newConv;
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to create conversation.";
        setError(msg);
        return null;
      }
    },
    []
  );

  // Delete a conversation and return to a valid chat state
  const deleteConversation = React.useCallback(
    async (conversationId: string): Promise<void> => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setIsStreaming(false);
      setStreamingDelta("");

      try {
        await chatApi.deleteConversation(conversationId);
        setConversations((prev) => {
          const remaining = prev.filter((c) => c.id !== conversationId);
          if (activeConversationId === conversationId) {
            if (remaining.length > 0) {
              selectConversation(remaining[0].id);
            } else {
              setActiveConversationId(null);
              setActiveConversationDetail(null);
              setMessages([]);
            }
          }
          return remaining;
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to delete conversation.";
        setError(msg);
      }
    },
    [activeConversationId, selectConversation]
  );

  // Stop currently active streaming generation
  const stopStreaming = React.useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    const currentDelta = streamingDeltaRef.current;
    if (currentDelta.trim()) {
      const partialMsg: MessageItem = {
        id: `stopped-${Date.now()}`,
        conversation_id: activeConversationId || "",
        role: "assistant",
        content: currentDelta + "\n\n*(Generation stopped by user)*",
        sources: [],
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, partialMsg]);
    }

    setStreamingDelta("");
    setIsStreaming(false);
  }, [activeConversationId]);

  // Send a question/message with streaming response
  const sendMessage = React.useCallback(
    async (documentId: string, content: string) => {
      const trimmed = content.trim();
      if (!trimmed || isStreaming) return;

      setError(null);

      // Ensure we have an active conversation
      let targetConvId = activeConversationId;
      if (!targetConvId) {
        const newConv = await createConversation();
        if (!newConv) return;
        targetConvId = newConv.id;
      }

      // Optimistically append user message
      const userMsg: MessageItem = {
        id: `user-${Date.now()}`,
        conversation_id: targetConvId,
        role: "user",
        content: trimmed,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      // Set up streaming state
      setIsStreaming(true);
      setStreamingDelta("");
      streamingDeltaRef.current = "";

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        await chatApi.streamMessage(
          targetConvId,
          documentId,
          trimmed,
          {
            onDelta: (delta: string) => {
              setStreamingDelta((prev) => prev + delta);
            },
            onComplete: (data: StreamCompletePayload) => {
              const assistantMsg: MessageItem = {
                id: data.message_id,
                conversation_id: data.conversation_id,
                role: "assistant",
                content: data.answer,
                sources: data.citations as Citation[],
                created_at: new Date().toISOString(),
              };
              setMessages((prev) => [...prev, assistantMsg]);
              setStreamingDelta("");
              setIsStreaming(false);
              abortControllerRef.current = null;
              // Refresh conversation list to capture updated title / timestamp
              refreshConversations();
            },
            onError: (err: Error) => {
              if (controller.signal.aborted) {
                return;
              }
              setError(err.message || "Failed to stream answer from provider.");
              setIsStreaming(false);
              abortControllerRef.current = null;
            },
          },
          controller.signal
        );
      } catch (err: unknown) {
        if (controller.signal.aborted) {
          return;
        }
        const errorMsg =
          err instanceof Error ? err.message : "Network error occurred while sending message.";
        setError(errorMsg);
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [activeConversationId, isStreaming, createConversation, refreshConversations]
  );

  const clearError = React.useCallback(() => {
    setError(null);
  }, []);

  const activeConversation = React.useMemo(() => {
    return conversations.find((c) => c.id === activeConversationId) || null;
  }, [conversations, activeConversationId]);

  return {
    conversations,
    activeConversation,
    activeConversationId,
    activeConversationDetail,
    messages,
    isLoadingConversations,
    isLoadingMessages,
    isStreaming,
    streamingDelta,
    error,
    selectConversation,
    createConversation,
    deleteConversation,
    sendMessage,
    stopStreaming,
    clearError,
    refreshConversations,
  };
}
