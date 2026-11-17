import * as React from "react";
import { Bot, User, Sparkles } from "lucide-react";
import { MessageItem } from "@/lib/api";
import { CitationList } from "@/components/CitationList";
import { Skeleton } from "@/components/ui/skeleton";

export interface MessageListProps {
  messages: MessageItem[];
  isStreaming: boolean;
  streamingDelta: string;
  isLoading?: boolean;
  documentTitle?: string;
  documentId?: string;
  onSuggestionClick?: (question: string) => void;
}

export function MessageList({
  messages,
  isStreaming,
  streamingDelta,
  isLoading = false,
  documentTitle,
  documentId,
  onSuggestionClick,
}: MessageListProps) {
  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or incoming streaming deltas
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingDelta, isStreaming]);

  // Format message time
  const formatTime = (isoString?: string) => {
    if (!isoString) return "";
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch {
      return "";
    }
  };

  const sampleSuggestions = [
    "What is the main topic or objective of this document?",
    "Summarize the key findings or methodology.",
    "What are the specific parameters or figures mentioned?",
  ];

  return (
    <div
      ref={containerRef}
      role="log"
      aria-live="polite"
      aria-label="Conversation message thread"
      className="flex-1 overflow-y-auto p-4 md:p-6 space-y-5"
    >
      {/* Loading Skeletons */}
      {isLoading && (
        <div
          role="status"
          aria-live="polite"
          aria-label="Loading conversation messages..."
          className="space-y-5 py-2"
        >
          {/* User message skeleton */}
          <div className="flex flex-col items-end space-y-1">
            <Skeleton className="h-10 w-2/3 max-w-md rounded-2xl rounded-tr-xs" />
            <Skeleton className="h-2.5 w-16 mr-1" />
          </div>
          {/* Assistant message skeleton */}
          <div className="flex items-start gap-2.5 max-w-[90%]">
            <Skeleton className="h-7 w-7 rounded-lg shrink-0 mt-0.5" />
            <div className="space-y-2 flex-1">
              <Skeleton className="h-20 w-full rounded-2xl rounded-tl-xs" />
              <div className="flex gap-2">
                <Skeleton className="h-5 w-16 rounded-md" />
                <Skeleton className="h-5 w-16 rounded-md" />
              </div>
            </div>
          </div>
          <span className="sr-only">Loading conversation history...</span>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && messages.length === 0 && !isStreaming && (
        <div className="flex flex-col items-center justify-center h-full py-10 px-4 text-center space-y-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary shadow-xs">
            <Sparkles className="h-6 w-6" aria-hidden="true" />
          </div>
          <div className="max-w-md space-y-1.5">
            <h3 className="text-base font-semibold text-foreground">
              {documentTitle ? `Ask about "${documentTitle}"` : "Ready for your questions"}
            </h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Every answer is grounded strictly in the document text using hybrid BM25 and
              sentence-transformer vector retrieval. Each claim includes page citations.
            </p>
          </div>

          {documentTitle && onSuggestionClick && (
            <div className="w-full max-w-md pt-2 space-y-2">
              <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                Suggested questions
              </p>
              <div className="flex flex-col gap-1.5">
                {sampleSuggestions.map((q, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => onSuggestionClick(q)}
                    className="text-left text-xs px-3 py-2 rounded-lg border border-border bg-card/60 hover:bg-muted text-card-foreground transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                  >
                    &ldquo;{q}&rdquo;
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Render Persisted Messages */}
      {!isLoading &&
        messages.map((message) => {
          const isUser = message.role === "user";

          if (isUser) {
            return (
              <div
                key={message.id}
                className="flex flex-col items-end group"
                data-testid="user-message"
              >
                <div className="flex items-end gap-2 max-w-[85%]">
                  <div className="rounded-2xl rounded-tr-xs bg-primary px-4 py-2.5 text-sm text-primary-foreground shadow-xs leading-relaxed whitespace-pre-wrap select-text">
                    {message.content}
                  </div>
                  <div
                    className="hidden sm:flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/20 text-primary text-[10px] font-bold"
                    aria-hidden="true"
                  >
                    <User className="h-3.5 w-3.5" />
                  </div>
                </div>
                <span className="text-[10px] text-muted-foreground mt-1 pr-8 select-none">
                  {formatTime(message.created_at)}
                </span>
              </div>
            );
          }

          // Assistant Message
          return (
            <div
              key={message.id}
              className="flex flex-col items-start group"
              data-testid="assistant-message"
            >
              <div className="flex items-start gap-2.5 max-w-[90%]">
                <div
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary mt-0.5 shadow-2xs"
                  aria-hidden="true"
                >
                  <Bot className="h-4 w-4" />
                </div>

                <div className="rounded-2xl rounded-tl-xs border border-border bg-card px-4 py-3 text-sm text-card-foreground shadow-xs space-y-2.5">
                  <div className="prose prose-sm dark:prose-invert max-w-none text-foreground leading-relaxed whitespace-pre-wrap select-text">
                    {message.content}
                  </div>

                  {/* Sources / Citations list */}
                  {message.sources && message.sources.length > 0 && (
                    <CitationList citations={message.sources} documentId={documentId} />
                  )}
                </div>
              </div>

              <div className="flex items-center gap-2 pl-9.5 mt-1 select-none">
                <span className="text-[10px] text-muted-foreground">Vectorless Assistant</span>
                {message.created_at && (
                  <span className="text-[10px] text-muted-foreground/60">
                    • {formatTime(message.created_at)}
                  </span>
                )}
              </div>
            </div>
          );
        })}

      {/* In-Flight Streaming Assistant Message */}
      {isStreaming && (
        <div className="flex flex-col items-start animate-in fade-in duration-100">
          <div className="flex items-start gap-2.5 max-w-[90%]">
            <div
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary mt-0.5 shadow-2xs"
              aria-hidden="true"
            >
              <Bot className="h-4 w-4 animate-pulse motion-reduce:animate-none" />
            </div>

            <div className="rounded-2xl rounded-tl-xs border border-border bg-card px-4 py-3 text-sm text-card-foreground shadow-xs space-y-2">
              <div className="text-foreground leading-relaxed whitespace-pre-wrap select-text">
                {streamingDelta}
                <span
                  className="inline-block w-1.5 h-4 ml-0.5 align-middle bg-primary animate-pulse motion-reduce:animate-none"
                  aria-hidden="true"
                />
              </div>

              {!streamingDelta && (
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground py-1">
                  <span className="inline-block h-2 w-2 rounded-full bg-primary/60 animate-ping motion-reduce:animate-none" />
                  <span>Synthesizing grounded response...</span>
                </div>
              )}
            </div>
          </div>
          <span className="text-[10px] text-muted-foreground pl-9.5 mt-1 select-none flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse motion-reduce:animate-none" />
            Streaming response...
          </span>
        </div>
      )}

      {/* Screen-reader live update */}
      <div className="sr-only" role="status" aria-live="polite">
        {isStreaming ? "Assistant is streaming an answer..." : ""}
      </div>

      {/* Anchor for automatic scroll */}
      <div ref={messagesEndRef} />
    </div>
  );
}
