import * as React from "react";
import { FileText, X, Loader2, AlertCircle, Hash, Copy, Check } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Citation, ChunkItem, chatApi } from "@/lib/api";
import { useToast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export interface CitationListProps {
  citations?: Citation[] | null;
  documentId?: string;
}

export function CitationList({ citations, documentId }: CitationListProps) {
  const { toast } = useToast();
  const shouldReduceMotion = useReducedMotion();
  const [selectedCitation, setSelectedCitation] = React.useState<Citation | null>(null);
  const [chunkData, setChunkData] = React.useState<ChunkItem | null>(null);
  const [isLoadingChunk, setIsLoadingChunk] = React.useState<boolean>(false);
  const [chunkError, setChunkError] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState<boolean>(false);

  const modalRef = React.useRef<HTMLDivElement>(null);
  const closeButtonRef = React.useRef<HTMLButtonElement>(null);

  // Handle citation button click
  const handleOpenCitation = async (citation: Citation) => {
    setSelectedCitation(citation);
    setChunkData(null);
    setChunkError(null);
    setIsLoadingChunk(true);
    setCopied(false);

    try {
      const data = await chatApi.getChunk(citation.chunk_id, documentId);
      setChunkData(data);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Failed to load citation chunk text from server.";
      setChunkError(msg);
    } finally {
      setIsLoadingChunk(false);
    }
  };

  const handleClose = () => {
    setSelectedCitation(null);
    setChunkData(null);
    setChunkError(null);
    setCopied(false);
  };

  // Keyboard accessibility: ESC key to close modal
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && selectedCitation) {
        handleClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedCitation]);

  // Focus trap / focus to close button when opened
  React.useEffect(() => {
    if (selectedCitation && closeButtonRef.current) {
      closeButtonRef.current.focus();
    }
  }, [selectedCitation, isLoadingChunk]);

  const handleCopyText = async () => {
    if (!chunkData?.content) return;
    try {
      await navigator.clipboard.writeText(chunkData.content);
      setCopied(true);
      toast.success("Snippet Copied", "Citation text copied to clipboard.");
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Copy Failed", "Could not write snippet to clipboard.");
    }
  };

  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="pt-2 border-t border-border/60">
      <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
        <span className="font-medium text-[11px] select-none text-foreground/80 flex items-center gap-1">
          <FileText className="h-3 w-3" aria-hidden="true" />
          Sources:
        </span>
        {citations.map((citation, idx) => (
          <button
            key={`${citation.chunk_id}-${idx}`}
            type="button"
            onClick={() => handleOpenCitation(citation)}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-secondary/80 hover:bg-secondary text-secondary-foreground hover:text-foreground border border-border/80 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
            title={`View text excerpt from page ${citation.page_number} (Chunk ${citation.chunk_id.slice(0, 8)})`}
            aria-haspopup="dialog"
            aria-expanded={selectedCitation?.chunk_id === citation.chunk_id}
          >
            <span>Page {citation.page_number}</span>
            {citation.score !== undefined && citation.score !== null && (
              <span className="text-[10px] text-muted-foreground/80">
                ({(citation.score * 100).toFixed(0)}%)
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Accessible Modal Dialog for Citation Chunk Source */}
      <AnimatePresence>
        {selectedCitation && (
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="citation-modal-title"
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-xs"
            onClick={(e) => {
              if (e.target === e.currentTarget) handleClose();
            }}
          >
            <motion.div
              ref={modalRef}
              initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.95, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.95, y: 8 }}
              transition={{ duration: shouldReduceMotion ? 0 : 0.15, ease: "easeOut" }}
              className="relative w-full max-w-xl max-h-[85vh] flex flex-col rounded-xl border border-border bg-card p-6 shadow-xl text-card-foreground overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-start justify-between border-b border-border pb-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs font-semibold px-2 py-0.5">
                      Page {selectedCitation.page_number}
                    </Badge>
                    <h3
                      id="citation-modal-title"
                      className="text-base font-semibold leading-tight text-foreground"
                    >
                      Citation Source
                    </h3>
                  </div>
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <Hash className="h-3 w-3" aria-hidden="true" />
                    Chunk ID:{" "}
                    <code className="font-mono text-[11px] bg-muted px-1 rounded">
                      {selectedCitation.chunk_id}
                    </code>
                  </p>
                </div>

                <Button
                  ref={closeButtonRef}
                  variant="ghost"
                  size="sm"
                  onClick={handleClose}
                  className="h-8 w-8 p-0 rounded-full hover:bg-muted text-muted-foreground hover:text-foreground cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                  aria-label="Close citation details"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                </Button>
              </div>

              {/* Modal Body */}
              <div className="flex-1 overflow-y-auto py-4 space-y-3">
                {isLoadingChunk && (
                  <div
                    role="status"
                    aria-live="polite"
                    className="flex flex-col items-center justify-center py-12 space-y-3"
                  >
                    <Loader2 className="h-7 w-7 animate-spin text-primary" aria-hidden="true" />
                    <p className="text-xs text-muted-foreground">
                      Retrieving cited text chunk from source endpoint...
                    </p>
                  </div>
                )}

                {chunkError && (
                  <div
                    role="alert"
                    className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive flex items-start gap-2.5"
                  >
                    <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
                    <div className="space-y-1">
                      <p className="font-semibold">Could not load citation source</p>
                      <p className="text-muted-foreground">{chunkError}</p>
                    </div>
                  </div>
                )}

                {chunkData && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span className="font-medium">
                        Chunk Index #{chunkData.chunk_index} • {chunkData.char_count} chars
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleCopyText}
                        className="h-7 text-xs gap-1.5 px-2.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                        aria-label="Copy citation text to clipboard"
                      >
                        {copied ? (
                          <>
                            <Check className="h-3.5 w-3.5 text-emerald-500" aria-hidden="true" />
                            <span className="text-emerald-600 dark:text-emerald-400">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
                            <span>Copy snippet</span>
                          </>
                        )}
                      </Button>
                    </div>

                    <div className="relative rounded-lg border border-border/80 bg-muted/30 p-3.5 text-xs text-foreground font-mono leading-relaxed whitespace-pre-wrap select-text max-h-[350px] overflow-y-auto">
                      {chunkData.content}
                    </div>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="border-t border-border pt-3 flex justify-between items-center text-xs text-muted-foreground">
                <span>Retrieved via user-owned source endpoint</span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleClose}
                  className="h-8 text-xs cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                >
                  Close
                </Button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
