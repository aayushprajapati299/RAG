import * as React from "react";
import {
  UploadCloud,
  FileText,
  FileCode,
  CheckCircle2,
  AlertCircle,
  Loader2,
  X,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { UploadStatus } from "@/hooks/useDocuments";
import { IngestionJob } from "@/lib/api";
import { useToast } from "@/lib/toast";

interface DocumentUploadProps {
  uploadState: UploadStatus;
  uploadError: string | null;
  activeJob: IngestionJob | null;
  onUpload: (file: File) => Promise<unknown>;
  onReset: () => void;
}

export function DocumentUpload({
  uploadState,
  uploadError,
  activeJob,
  onUpload,
  onReset,
}: DocumentUploadProps) {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [isDragOver, setIsDragOver] = React.useState<boolean>(false);
  const fileInputRef = React.useRef<HTMLInputElement | null>(null);

  const prevUploadState = React.useRef<UploadStatus>(uploadState);

  // Trigger toast notifications on state transitions
  React.useEffect(() => {
    if (prevUploadState.current !== uploadState) {
      if (uploadState === "completed") {
        toast.success("Document Ingestion Complete", "Text chunks indexed and ready for retrieval.");
      } else if (uploadState === "error" && uploadError) {
        toast.error("Document Ingestion Failed", uploadError);
      }
      prevUploadState.current = uploadState;
    }
  }, [uploadState, uploadError, toast]);

  const isBusy = uploadState === "uploading" || uploadState === "processing";

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      onReset();
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isBusy) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      onReset();
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isBusy) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || isBusy) return;

    try {
      await onUpload(selectedFile);
    } catch {
      // Errors handled via state in hook
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onReset();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Provide contextual retry guidance based on error message
  const getRetryGuidance = (err: string | null): string => {
    if (!err) return "Please check your document and try again.";
    const lower = err.toLowerCase();
    if (lower.includes("ocr")) {
      return "OCR engine was unavailable. Upload a searchable PDF containing digital text or a plain TXT file.";
    }
    if (lower.includes("size") || lower.includes("25 mb")) {
      return "Document exceeds the 25 MB limit. Please compress or split the document before uploading.";
    }
    if (lower.includes("unsupported") || lower.includes("invalid text") || lower.includes("format")) {
      return "Only valid .pdf or .txt documents are supported. Ensure the file is not corrupted.";
    }
    return "Verify the file is not password-protected or corrupted, then try again.";
  };

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <UploadCloud className="h-4 w-4 text-primary" aria-hidden="true" />
            Upload Document
          </CardTitle>
          <div className="flex gap-1">
            <Badge variant="outline" className="text-[10px] font-normal">
              PDF, TXT
            </Badge>
            <Badge variant="outline" className="text-[10px] font-normal">
              Max 25 MB
            </Badge>
          </div>
        </div>
        <CardDescription>
          Extracts bounded chunks and generates 384d vector embeddings.
        </CardDescription>
      </CardHeader>

      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-3">
          {/* Drag & Drop Target Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition-all ${
              isBusy
                ? "border-muted bg-muted/20 opacity-60 cursor-not-allowed"
                : isDragOver
                ? "border-primary bg-primary/10 shadow-xs cursor-copy"
                : "border-border hover:border-primary/50 hover:bg-muted/30 cursor-pointer"
            }`}
          >
            <label htmlFor="document-file-input" className="sr-only">
              Upload PDF or TXT file
            </label>
            <input
              ref={fileInputRef}
              id="document-file-input"
              type="file"
              accept=".pdf,.txt"
              disabled={isBusy}
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              aria-label="Upload PDF or TXT file"
            />

            <UploadCloud
              className={`h-8 w-8 mb-2 transition-colors ${
                isDragOver ? "text-primary scale-110" : "text-muted-foreground"
              }`}
              aria-hidden="true"
            />

            <p className="text-xs font-semibold text-foreground">
              {isDragOver ? "Drop your document here" : "Click to select or drag and drop"}
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              Supports .pdf and .txt files up to 25 MB
            </p>
          </div>

          {/* Selected File Badge / Info */}
          {selectedFile && (
            <div className="flex items-center justify-between rounded-md border border-border bg-muted/30 p-2.5 text-xs">
              <div className="flex items-center space-x-2 truncate">
                {selectedFile.name.endsWith(".pdf") ? (
                  <FileText className="h-4 w-4 shrink-0 text-red-500" aria-hidden="true" />
                ) : (
                  <FileCode className="h-4 w-4 shrink-0 text-blue-500" aria-hidden="true" />
                )}
                <div className="truncate text-left">
                  <p className="font-medium text-foreground truncate">{selectedFile.name}</p>
                  <p className="text-[10px] text-muted-foreground">{formatFileSize(selectedFile.size)}</p>
                </div>
              </div>
              {!isBusy && (
                <button
                  type="button"
                  onClick={clearSelection}
                  className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground cursor-pointer"
                  aria-label="Remove selected file"
                >
                  <X className="h-3.5 w-3.5" aria-hidden="true" />
                </button>
              )}
            </div>
          )}

          {/* Upload & Ingestion Status Notifications */}
          {uploadState === "uploading" && (
            <div
              role="status"
              aria-live="polite"
              className="flex items-center gap-2 rounded-lg border border-primary/20 bg-primary/5 p-3 text-xs text-primary"
            >
              <Loader2 className="h-4 w-4 animate-spin shrink-0" aria-hidden="true" />
              <span>Uploading document to server...</span>
            </div>
          )}

          {uploadState === "processing" && (
            <div
              role="status"
              aria-live="polite"
              className="flex items-start gap-2.5 rounded-lg border border-blue-500/20 bg-blue-500/10 p-3 text-xs text-blue-700 dark:text-blue-300"
            >
              <Loader2 className="h-4 w-4 animate-spin shrink-0 mt-0.5" aria-hidden="true" />
              <div className="space-y-0.5">
                <p className="font-semibold">Processing &amp; Extracting Document...</p>
                <p className="text-[11px] opacity-90">
                  {activeJob && activeJob.pages_processed > 0
                    ? `Processed ${activeJob.pages_processed} pages...`
                    : "Extracting text chunks and generating dense pgvector embeddings..."}
                </p>
              </div>
            </div>
          )}

          {uploadState === "completed" && (
            <div
              role="status"
              aria-live="polite"
              className="flex items-center gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/10 p-3 text-xs text-emerald-700 dark:text-emerald-300"
            >
              <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />
              <div>
                <p className="font-semibold">Document Ready for Q&amp;A</p>
                <p className="text-[11px] opacity-90">
                  Extraction complete and indexed into PostgreSQL pgvector.
                </p>
              </div>
            </div>
          )}

          {uploadState === "error" && (
            <div
              role="alert"
              aria-live="assertive"
              className="rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-xs text-destructive space-y-1.5"
            >
              <div className="flex items-start gap-2">
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
                <div className="space-y-1">
                  <p className="font-semibold">Document Ingestion Failed</p>
                  <p className="text-[11px]">{uploadError || "An error occurred during extraction."}</p>
                </div>
              </div>
              <div className="rounded bg-background/50 p-2 text-[11px] text-muted-foreground border border-border/40">
                <span className="font-medium text-foreground">Guidance: </span>
                {getRetryGuidance(uploadError)}
              </div>
            </div>
          )}
        </CardContent>

        <CardFooter className="pt-0 flex gap-2">
          {uploadState === "error" ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleSubmit}
              className="w-full gap-1.5"
              disabled={!selectedFile || isBusy}
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
              Retry Ingestion
            </Button>
          ) : (
            <Button
              type="submit"
              size="sm"
              className="w-full"
              disabled={!selectedFile || isBusy}
            >
              {isBusy ? (
                <>
                  <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                  {uploadState === "uploading" ? "Uploading..." : "Processing..."}
                </>
              ) : (
                "Ingest & Generate Embeddings"
              )}
            </Button>
          )}
        </CardFooter>
      </form>
    </Card>
  );
}
