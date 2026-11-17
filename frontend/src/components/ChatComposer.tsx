import * as React from "react";
import { Send, Square, AlertCircle, X, Sparkles, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/lib/toast";

export interface ChatComposerProps {
  onSendMessage: (content: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
  disabledReason?: string;
  error?: string | null;
  onClearError?: () => void;
  placeholder?: string;
  initialValue?: string;
}

export function ChatComposer({
  onSendMessage,
  onStop,
  isStreaming,
  disabled = false,
  disabledReason,
  error,
  onClearError,
  placeholder,
  initialValue = "",
}: ChatComposerProps) {
  const { toast } = useToast();
  const [content, setContent] = React.useState(initialValue);
  const [lastSentMessage, setLastSentMessage] = React.useState<string>("");
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const prevErrorRef = React.useRef<string | null>(null);
  const isSubmittingRef = React.useRef<boolean>(false);

  // Notify via toast on chat errors
  React.useEffect(() => {
    if (error && error !== prevErrorRef.current) {
      toast.error("Chat Error", error);
    }
    prevErrorRef.current = error || null;
  }, [error, toast]);

  // Synchronize when initialValue prop changes (e.g., from suggested question click)
  React.useEffect(() => {
    if (initialValue) {
      setContent(initialValue);
      if (textareaRef.current) {
        textareaRef.current.value = initialValue;
        textareaRef.current.focus();
      }
    }
  }, [initialValue]);

  // Submit current non-empty question exactly once
  const handleSubmit = (e?: React.FormEvent) => {
    if (e) {
      e.preventDefault();
    }
    const currentVal = (textareaRef.current?.value ?? content).trim();
    if (!currentVal || isStreaming || disabled || isSubmittingRef.current) {
      return;
    }

    isSubmittingRef.current = true;
    setLastSentMessage(currentVal);
    setContent("");
    if (textareaRef.current) {
      textareaRef.current.value = "";
      textareaRef.current.style.height = "auto";
    }

    try {
      onSendMessage(currentVal);
    } finally {
      setTimeout(() => {
        isSubmittingRef.current = false;
      }, 100);
    }
  };

  // Accessible keyboard handler: Enter to send, Shift+Enter for newline
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Respect IME composition sessions (e.g. CJK character input or voice typing)
    if (e.nativeEvent?.isComposing || e.keyCode === 229) {
      return;
    }

    if ((e.key === "Enter" || e.keyCode === 13) && !e.shiftKey) {
      e.preventDefault();
      const currentVal = (textareaRef.current?.value ?? content).trim();
      // Only submit non-empty questions when not in progress and not disabled
      if (!isStreaming && !disabled && currentVal.length > 0 && !isSubmittingRef.current) {
        handleSubmit();
      }
    }
    // Shift+Enter falls through to default textarea behavior (inserting a newline)
  };

  const handleRetry = () => {
    if (!lastSentMessage || isStreaming || disabled) return;
    onClearError?.();
    onSendMessage(lastSentMessage);
  };

  // Auto-resize textarea height as content expands
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);
    const target = e.target;
    target.style.height = "auto";
    const newHeight = Math.min(target.scrollHeight, 180);
    target.style.height = `${newHeight}px`;
  };

  const defaultPlaceholder = disabled
    ? disabledReason || "Select or upload a document to begin asking questions..."
    : "Ask a question about the document... (Enter to send, Shift+Enter for newline)";

  return (
    <div className="border-t bg-background p-3 md:p-4 space-y-2">
      {/* Network or Provider Error Banner with Retry Action */}
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="flex items-start justify-between gap-2.5 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive animate-in fade-in"
        >
          <div className="flex items-start gap-2.5 flex-1">
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
            <div className="space-y-1.5 flex-1">
              <p className="font-semibold">Provider / Retrieval Error</p>
              <p className="text-foreground/90 leading-normal">{error}</p>
              {lastSentMessage && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleRetry}
                  className="h-7 text-xs gap-1.5 cursor-pointer mt-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                >
                  <RotateCcw className="h-3 w-3" aria-hidden="true" />
                  <span>Retry Question</span>
                </Button>
              )}
            </div>
          </div>
          {onClearError && (
            <button
              type="button"
              onClick={onClearError}
              className="p-1 rounded-md text-destructive/70 hover:text-destructive hover:bg-destructive/20 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label="Dismiss error"
            >
              <X className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          )}
        </div>
      )}

      {/* Composer Input Area */}
      <form onSubmit={handleSubmit} className="relative flex flex-col gap-2">
        <label htmlFor="chat-textarea" className="sr-only">
          Question input for document Q&amp;A
        </label>

        <div className="relative flex items-end rounded-xl border border-input bg-card shadow-xs focus-within:ring-2 focus-within:ring-ring focus-within:border-primary transition-all">
          <textarea
            ref={textareaRef}
            id="chat-textarea"
            rows={1}
            value={content}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={placeholder || defaultPlaceholder}
            className="flex-1 max-h-[180px] min-h-[44px] resize-none bg-transparent px-3.5 py-2.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
            aria-label="Ask a question about your document"
            aria-describedby={error ? "chat-error-message" : undefined}
          />

          <div className="flex items-center p-2 gap-1.5 shrink-0">
            {isStreaming ? (
              <Button
                type="button"
                variant="destructive"
                size="sm"
                onClick={onStop}
                className="h-8 gap-1.5 px-3 text-xs font-semibold cursor-pointer shadow-xs animate-pulse motion-reduce:animate-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                aria-label="Stop generating response"
              >
                <Square className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
                <span>Stop</span>
              </Button>
            ) : (
              <Button
                type="submit"
                size="sm"
                disabled={disabled || isStreaming || !content.trim()}
                className="h-8 gap-1.5 px-3 text-xs font-semibold cursor-pointer shadow-xs disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                aria-label="Send message"
              >
                <Send className="h-3.5 w-3.5" aria-hidden="true" />
                <span className="hidden sm:inline">Send</span>
              </Button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-1.5 px-1 text-[11px] text-muted-foreground select-none">
          <span className="flex items-center gap-1">
            <Sparkles className="h-3 w-3 text-primary/70" aria-hidden="true" />
            Grounded in active document context
          </span>
          <span className="inline-flex items-center gap-1">
            Press <kbd className="font-mono bg-muted px-1.5 py-0.5 rounded border border-border text-[10px]">Enter ↵</kbd> to send,{" "}
            <kbd className="font-mono bg-muted px-1.5 py-0.5 rounded border border-border text-[10px]">Shift + Enter</kbd> for newline
          </span>
        </div>
      </form>
    </div>
  );
}
