import * as React from "react";
import {
  BookOpen,
  FileText,
  FileCode,
  CheckCircle2,
  RefreshCw,
  FolderOpen,
  AlertCircle,
  Plus,
  Trash2,
  Paperclip,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { DocumentItem } from "@/lib/api";
import { ConfirmDialog } from "@/components/ConfirmDialog";

export interface DocumentListProps {
  documents: DocumentItem[];
  selectedDocument: DocumentItem | null;
  isLoading: boolean;
  error: string | null;
  onSelectDocument: (doc: DocumentItem) => void;
  onDeleteDocument: (docId: string) => Promise<void>;
  onRefresh: () => void;
  onTriggerUpload?: () => void;
  allUserDocuments?: DocumentItem[];
  onAttachDocument?: (docId: string) => Promise<void>;
}

export function DocumentList({
  documents,
  selectedDocument,
  isLoading,
  error,
  onSelectDocument,
  onDeleteDocument,
  onRefresh,
  onTriggerUpload,
  allUserDocuments = [],
  onAttachDocument,
}: DocumentListProps) {
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  const [deleteTarget, setDeleteTarget] = React.useState<DocumentItem | null>(null);
  const [isDeleting, setIsDeleting] = React.useState<boolean>(false);
  const [isAttaching, setIsAttaching] = React.useState<boolean>(false);
  const [showAttachMenu, setShowAttachMenu] = React.useState<boolean>(false);

  const handleRefreshClick = async () => {
    setIsRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    try {
      await onDeleteDocument(deleteTarget.id);
      setDeleteTarget(null);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleAttachClick = async (docId: string) => {
    if (!onAttachDocument) return;
    setIsAttaching(true);
    try {
      await onAttachDocument(docId);
      setShowAttachMenu(false);
    } finally {
      setIsAttaching(false);
    }
  };

  // Documents owned by user that are not yet attached to this chat
  const unattachedDocs = React.useMemo(() => {
    const currentIds = new Set(documents.map((d) => d.id));
    return allUserDocuments.filter((d) => !currentIds.has(d.id));
  }, [allUserDocuments, documents]);

  const formatFileSize = (bytes?: number | null): string => {
    if (!bytes) return "Unknown size";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateStr: string): string => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <>
      <Card className="shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <BookOpen className="h-4 w-4 text-primary" aria-hidden="true" />
              Chat Documents
            </CardTitle>
            <div className="flex items-center gap-1.5">
              <Badge variant="secondary" className="text-[10px]">
                {documents.length} {documents.length === 1 ? "Doc" : "Docs"}
              </Badge>
              {unattachedDocs.length > 0 && onAttachDocument && (
                <div className="relative">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowAttachMenu((prev) => !prev)}
                    className="h-7 text-xs gap-1 px-2 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    title="Attach an existing document from your library"
                  >
                    <Paperclip className="h-3 w-3" aria-hidden="true" />
                    <span>Attach ({unattachedDocs.length})</span>
                  </Button>
                  {showAttachMenu && (
                    <div className="absolute right-0 top-full mt-1 z-30 w-56 rounded-md border border-border bg-popover p-1 shadow-lg text-xs space-y-1 animate-in fade-in zoom-in-95">
                      <p className="px-2 py-1 text-[10px] font-semibold text-muted-foreground uppercase">
                        From your library:
                      </p>
                      {unattachedDocs.map((udoc) => (
                        <button
                          key={udoc.id}
                          type="button"
                          disabled={isAttaching}
                          onClick={() => handleAttachClick(udoc.id)}
                          className="w-full text-left px-2 py-1.5 rounded hover:bg-muted transition-colors truncate flex items-center justify-between cursor-pointer"
                        >
                          <span className="truncate">{udoc.filename}</span>
                          <span className="text-[10px] text-muted-foreground shrink-0 uppercase ml-1">
                            {udoc.file_type}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={handleRefreshClick}
                disabled={isLoading || isRefreshing}
                className="h-7 w-7 text-muted-foreground hover:text-foreground cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                aria-label="Refresh document list"
              >
                <RefreshCw
                  className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`}
                  aria-hidden="true"
                />
              </Button>
            </div>
          </div>
          <CardDescription>
            Documents attached to this chat. Select one to scope your questions.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-2">
          {/* Loading skeletons */}
          {isLoading && (
            <div
              role="status"
              aria-live="polite"
              aria-label="Loading document list..."
              className="space-y-2.5 py-1"
            >
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="flex items-start justify-between rounded-lg border border-border/70 p-3 space-x-3"
                >
                  <div className="flex items-start space-x-2.5 flex-1">
                    <Skeleton className="h-4 w-4 rounded shrink-0 mt-0.5" />
                    <div className="space-y-1.5 flex-1">
                      <Skeleton className="h-3.5 w-3/4 rounded" />
                      <Skeleton className="h-2.5 w-1/2 rounded" />
                      <Skeleton className="h-2 w-1/3 rounded" />
                    </div>
                  </div>
                </div>
              ))}
              <span className="sr-only">Loading documents from server...</span>
            </div>
          )}

          {/* Error state with retry action */}
          {!isLoading && error && (
            <div
              role="alert"
              className="flex items-start gap-2.5 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-xs text-destructive"
            >
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
              <div className="space-y-1 flex-1">
                <p className="font-semibold">Failed to load documents</p>
                <p className="text-[11px] text-destructive/90">{error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefreshClick}
                  className="h-7 text-xs mt-1.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                >
                  Retry loading
                </Button>
              </div>
            </div>
          )}

          {/* Empty state with action */}
          {!isLoading && !error && documents.length === 0 && (
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border/80 p-8 text-center bg-muted/10 space-y-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted text-muted-foreground">
                <FolderOpen className="h-5 w-5" aria-hidden="true" />
              </div>
              <div className="space-y-1">
                <p className="text-xs font-semibold text-foreground">No documents in this chat</p>
                <p className="text-[11px] text-muted-foreground max-w-[220px]">
                  Upload a PDF or TXT document above to attach it to this conversation and start asking questions.
                </p>
              </div>
              {onTriggerUpload && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onTriggerUpload}
                  className="h-7 text-xs gap-1.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                >
                  <Plus className="h-3 w-3" aria-hidden="true" />
                  <span>Upload document</span>
                </Button>
              )}
            </div>
          )}

          {/* Screen reader live update for count */}
          <div className="sr-only" role="status" aria-live="polite">
            {!isLoading && `${documents.length} documents available.`}
          </div>

          {/* Documents list */}
          {!isLoading &&
            !error &&
            documents.map((doc) => {
              const isSelected = selectedDocument?.id === doc.id;
              const isPdf = doc.file_type.toLowerCase() === "pdf";

              return (
                <div
                  key={doc.id}
                  className={`group relative flex items-start justify-between rounded-lg border p-3 text-left transition-all ${
                    isSelected
                      ? "border-primary bg-primary/5 shadow-xs"
                      : "border-border hover:bg-muted/40 cursor-pointer"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => onSelectDocument(doc)}
                    className="flex-1 flex items-start space-x-2.5 truncate text-left focus-visible:outline-none"
                    aria-pressed={isSelected}
                    aria-label={`Select document ${doc.filename}, ${doc.page_count} pages, ${doc.file_type}`}
                  >
                    {isPdf ? (
                      <FileText className="h-4 w-4 shrink-0 text-red-500 mt-0.5" aria-hidden="true" />
                    ) : (
                      <FileCode className="h-4 w-4 shrink-0 text-blue-500 mt-0.5" aria-hidden="true" />
                    )}
                    <div className="truncate text-left flex-1 min-w-0">
                      <p className="text-xs font-semibold leading-tight text-foreground truncate">
                        {doc.filename}
                      </p>
                      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground mt-1">
                        <span className="font-medium text-foreground/80 uppercase text-[10px]">
                          {doc.file_type}
                        </span>
                        <span>•</span>
                        <span>
                          {doc.page_count} {doc.page_count === 1 ? "page" : "pages"}
                        </span>
                        <span>•</span>
                        <span>{formatFileSize(doc.file_size_bytes)}</span>
                      </div>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        Uploaded {formatDate(doc.created_at)}
                      </p>
                    </div>

                    {isSelected && (
                      <CheckCircle2
                        className="h-4 w-4 shrink-0 text-primary mt-0.5 ml-2 mr-1"
                        aria-label="Active selection"
                      />
                    )}
                  </button>

                  {/* Delete Document Button with Confirmation Dialog */}
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteTarget(doc);
                    }}
                    className="h-7 w-7 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded shrink-0 opacity-80 group-hover:opacity-100 cursor-pointer focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label={`Delete document "${doc.filename}"`}
                    title="Delete document"
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                  </Button>
                </div>
              );
            })}
        </CardContent>
      </Card>

      {/* Confirmation Dialog before Deleting Document */}
      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title="Delete Document"
        description={`Are you sure you want to delete "${
          deleteTarget?.filename || "this document"
        }"? This will permanently remove its indexed text chunks, search vectors, and source files. Any active selection in chats will be cleared.`}
        confirmText="Delete Document"
        cancelText="Cancel"
        variant="destructive"
        isLoading={isDeleting}
        onConfirm={handleConfirmDelete}
        onClose={() => setDeleteTarget(null)}
      />
    </>
  );
}
