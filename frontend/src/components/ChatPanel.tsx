import * as React from "react";
import {
  MessageSquare,
  Plus,
  FileText,
  AlertTriangle,
  FolderOpen,
} from "lucide-react";
import { DocumentItem, ConversationItem, MessageItem } from "@/lib/api";
import { useChat } from "@/hooks/useChat";
import { useToast } from "@/lib/toast";
import { MessageList } from "@/components/MessageList";
import { ChatComposer } from "@/components/ChatComposer";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export interface ChatPanelProps {
  selectedDocument: DocumentItem | null;
  activeConversation?: ConversationItem | null;
  messages?: MessageItem[];
  isLoadingMessages?: boolean;
  isStreaming?: boolean;
  streamingDelta?: string;
  error?: string | null;
  onSendMessage?: (content: string) => void;
  onStopStreaming?: () => void;
  onClearError?: () => void;
  onNewChat?: () => void;
  onSelectMobileTab?: (tab: "chats" | "documents" | "chat") => void;
}

export function ChatPanel({
  selectedDocument,
  activeConversation: propActiveConversation,
  messages: propMessages,
  isLoadingMessages: propIsLoadingMessages,
  isStreaming: propIsStreaming,
  streamingDelta: propStreamingDelta,
  error: propError,
  onSendMessage: propOnSendMessage,
  onStopStreaming: propOnStopStreaming,
  onClearError: propOnClearError,
  onNewChat: propOnNewChat,
  onSelectMobileTab,
}: ChatPanelProps) {
  const { toast } = useToast();
  const internalChat = useChat();

  // Prefer controlled props if passed from App, else fallback to internal hook
  const activeConversation = propActiveConversation !== undefined ? propActiveConversation : internalChat.activeConversation;
  const messages = propMessages !== undefined ? propMessages : internalChat.messages;
  const isLoadingMessages = propIsLoadingMessages !== undefined ? propIsLoadingMessages : internalChat.isLoadingMessages;
  const isStreaming = propIsStreaming !== undefined ? propIsStreaming : internalChat.isStreaming;
  const streamingDelta = propStreamingDelta !== undefined ? propStreamingDelta : internalChat.streamingDelta;
  const error = propError !== undefined ? propError : internalChat.error;
  const onClearError = propOnClearError || internalChat.clearError;
  const onStopStreaming = propOnStopStreaming || internalChat.stopStreaming;

  const [suggestedInput, setSuggestedInput] = React.useState<string>("");

  const handleSendMessage = (content: string) => {
    if (!selectedDocument) return;
    if (propOnSendMessage) {
      propOnSendMessage(content);
    } else {
      internalChat.sendMessage(selectedDocument.id, content);
    }
    setSuggestedInput("");
  };

  const handleCreateNewChat = async () => {
    if (propOnNewChat) {
      propOnNewChat();
    } else {
      const newConv = await internalChat.createConversation("New Conversation");
      if (newConv) {
        toast.info("New Conversation Started", "You can now ask questions about the selected document.");
      }
    }
    setSuggestedInput("");
  };

  const handleSuggestionClick = (question: string) => {
    setSuggestedInput(question);
  };

  return (
    <Card className="flex flex-col flex-1 shadow-sm overflow-hidden h-full max-h-[calc(100vh-8.5rem)] min-h-[550px]">
      {/* Conversation Header */}
      <CardHeader className="flex-row items-center justify-between border-b py-3 px-4 md:px-6 shrink-0 bg-card/60">
        <div className="flex items-center space-x-3 overflow-hidden flex-1 min-w-0">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <MessageSquare className="h-5 w-5" aria-hidden="true" />
          </div>

          <div className="space-y-0.5 truncate flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base font-semibold leading-none truncate">
                {activeConversation ? activeConversation.title : "Document Q&A"}
              </CardTitle>
            </div>

            {/* Active Document Indicator */}
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground truncate">
              {selectedDocument ? (
                <>
                  <FileText className="h-3 w-3 text-primary shrink-0" aria-hidden="true" />
                  <span className="truncate">
                    Querying: <span className="font-semibold text-foreground">{selectedDocument.filename}</span>
                  </span>
                  <Badge variant="secondary" className="text-[10px] py-0 px-1.5 font-normal shrink-0">
                    {selectedDocument.page_count} {selectedDocument.page_count === 1 ? "page" : "pages"} • {selectedDocument.file_type.toUpperCase()}
                  </Badge>
                </>
              ) : (
                <div className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400 font-medium truncate">
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span className="truncate">No document selected.</span>
                  {onSelectMobileTab && (
                    <button
                      type="button"
                      onClick={() => onSelectMobileTab("documents")}
                      className="underline hover:text-foreground inline-flex items-center gap-1 text-[11px] cursor-pointer"
                    >
                      Choose one <FolderOpen className="h-3 w-3" />
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-2 shrink-0 ml-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleCreateNewChat}
            disabled={isStreaming}
            className="text-xs h-8 gap-1.5 cursor-pointer shadow-2xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
            title="Start a new conversation thread"
          >
            <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="hidden sm:inline">New Chat</span>
          </Button>
        </div>
      </CardHeader>

      {/* Message History & Stream Area */}
      <CardContent className="flex-1 flex flex-col p-0 overflow-hidden relative">
        <MessageList
          messages={messages}
          isStreaming={isStreaming}
          streamingDelta={streamingDelta}
          isLoading={isLoadingMessages}
          documentTitle={selectedDocument?.filename}
          documentId={selectedDocument?.id}
          onSuggestionClick={handleSuggestionClick}
        />
      </CardContent>

      {/* Bottom Composer */}
      <ChatComposer
        onSendMessage={handleSendMessage}
        onStop={onStopStreaming}
        isStreaming={isStreaming}
        disabled={!selectedDocument}
        disabledReason={
          !selectedDocument
            ? "Please select or upload a document from the left to start asking questions."
            : undefined
        }
        error={error}
        onClearError={onClearError}
        placeholder={
          selectedDocument
            ? `Ask a question about ${selectedDocument.filename}...`
            : "Select or upload a document to ask questions..."
        }
        initialValue={suggestedInput}
      />
    </Card>
  );
}
