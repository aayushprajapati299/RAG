import * as React from "react";
import { MessageSquare, Plus, Trash2, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ConversationItem } from "@/lib/api";
import { ConfirmDialog } from "@/components/ConfirmDialog";

export interface ChatSidebarProps {
  conversations: ConversationItem[];
  activeConversationId: string | null;
  isLoading: boolean;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => Promise<void>;
}

export function ChatSidebar({
  conversations,
  activeConversationId,
  isLoading,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
}: ChatSidebarProps) {
  const [deleteTarget, setDeleteTarget] = React.useState<ConversationItem | null>(null);
  const [isDeleting, setIsDeleting] = React.useState<boolean>(false);

  const formatDate = (dateStr: string): string => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      });
    } catch {
      return dateStr;
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    try {
      await onDeleteConversation(deleteTarget.id);
      setDeleteTarget(null);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <>
      <Card className="flex flex-col shadow-sm h-full max-h-[calc(100vh-8.5rem)] min-h-[550px] overflow-hidden">
        {/* Header with New Chat Button */}
        <CardHeader className="p-3 border-b bg-card/60 space-y-2 shrink-0">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-sm font-semibold">
              <MessageSquare className="h-4 w-4 text-primary" aria-hidden="true" />
              <span>Chats</span>
            </CardTitle>
            <Badge variant="secondary" className="text-[10px] py-0 px-1.5 font-normal">
              {conversations.length} {conversations.length === 1 ? "chat" : "chats"}
            </Badge>
          </div>

          <Button
            type="button"
            onClick={onNewChat}
            className="w-full h-8 text-xs font-semibold gap-1.5 cursor-pointer shadow-2xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
            aria-label="Start a new chat conversation"
          >
            <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            <span>New Chat</span>
          </Button>
        </CardHeader>

        {/* Conversation List */}
        <CardContent className="flex-1 p-2 overflow-y-auto space-y-1">
          {isLoading && (
            <div
              role="status"
              aria-live="polite"
              className="space-y-2 py-1 px-1"
            >
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="flex items-center space-x-2 rounded-lg border border-border/70 p-2.5">
                  <Skeleton className="h-4 w-4 rounded shrink-0" />
                  <div className="space-y-1.5 flex-1">
                    <Skeleton className="h-3 w-4/5 rounded" />
                    <Skeleton className="h-2 w-1/2 rounded" />
                  </div>
                </div>
              ))}
              <span className="sr-only">Loading conversation history...</span>
            </div>
          )}

          {!isLoading && conversations.length === 0 && (
            <div className="flex flex-col items-center justify-center p-6 text-center text-muted-foreground space-y-2 my-auto">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-muted">
                <Clock className="h-4 w-4" aria-hidden="true" />
              </div>
              <p className="text-xs font-medium text-foreground">No chats yet</p>
              <p className="text-[11px] text-muted-foreground leading-normal">
                Click "New Chat" to begin asking questions about documents.
              </p>
            </div>
          )}

          {!isLoading &&
            conversations.map((conv) => {
              const isActive = conv.id === activeConversationId;
              return (
                <div
                  key={conv.id}
                  className={`group relative flex items-center justify-between rounded-lg px-2.5 py-2 text-xs transition-colors cursor-pointer border ${
                    isActive
                      ? "border-primary/40 bg-primary/10 text-primary font-semibold shadow-2xs"
                      : "border-transparent text-foreground hover:bg-muted/60"
                  }`}
                  onClick={() => onSelectConversation(conv.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelectConversation(conv.id);
                    }
                  }}
                  aria-pressed={isActive}
                  aria-label={`Chat ${conv.title || "Untitled Chat"}`}
                >
                  <div className="flex items-center gap-2 overflow-hidden flex-1 min-w-0 pr-1">
                    <MessageSquare
                      className={`h-3.5 w-3.5 shrink-0 ${
                        isActive ? "text-primary" : "text-muted-foreground"
                      }`}
                      aria-hidden="true"
                    />
                    <div className="truncate flex-1">
                      <p className="truncate text-xs leading-snug">
                        {conv.title || "Untitled Chat"}
                      </p>
                      <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground font-normal mt-0.5">
                        <span>{formatDate(conv.updated_at || conv.created_at)}</span>
                        {typeof conv.message_count === "number" && conv.message_count > 0 && (
                          <>
                            <span>•</span>
                            <span>{conv.message_count} {conv.message_count === 1 ? "msg" : "msgs"}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Delete Conversation Action */}
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteTarget(conv);
                    }}
                    className="h-6 w-6 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded shrink-0 opacity-80 group-hover:opacity-100 cursor-pointer focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label={`Delete chat "${conv.title || "Untitled Chat"}"`}
                    title="Delete chat"
                  >
                    <Trash2 className="h-3 w-3" aria-hidden="true" />
                  </Button>
                </div>
              );
            })}
        </CardContent>
      </Card>

      {/* Delete Chat Confirmation Dialog */}
      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title="Delete Conversation"
        description={`Are you sure you want to delete "${
          deleteTarget?.title || "this chat"
        }"? All messages in this conversation will be permanently removed. Uploaded documents will remain preserved.`}
        confirmText="Delete Chat"
        cancelText="Keep Chat"
        variant="destructive"
        isLoading={isDeleting}
        onConfirm={handleConfirmDelete}
        onClose={() => setDeleteTarget(null)}
      />
    </>
  );
}
