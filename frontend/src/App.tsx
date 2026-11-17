import * as React from "react";
import {
  Sparkles,
  LogOut,
  User as UserIcon,
  Loader2,
  FileText,
  MessageSquare,
  MessagesSquare,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useDocuments } from "@/hooks/useDocuments";
import { useChat } from "@/hooks/useChat";
import { DocumentItem } from "@/lib/api";
import { DocumentUpload } from "@/components/DocumentUpload";
import { DocumentList } from "@/components/DocumentList";
import { ChatSidebar } from "@/components/ChatSidebar";
import { ChatPanel } from "@/components/ChatPanel";
import { ThemeToggle } from "@/components/ThemeToggle";
import LoginPage from "@/pages/LoginPage";
import RegisterPage from "@/pages/RegisterPage";
import { Button } from "@/components/ui/button";

export default function App() {
  const { user, isLoading, logout } = useAuth();

  const {
    conversations,
    activeConversation,
    activeConversationId,
    messages,
    isLoadingConversations,
    isLoadingMessages,
    isStreaming,
    streamingDelta,
    error: chatError,
    selectConversation,
    createConversation,
    deleteConversation,
    sendMessage,
    stopStreaming,
    clearError: clearChatError,
  } = useChat();

  const {
    documents,
    allUserDocuments,
    selectedDocument,
    selectDocument,
    deleteDocument,
    attachDocument,
    isLoading: isDocsLoading,
    error: docsError,
    activeJob,
    uploadState,
    uploadError,
    uploadDocument,
    resetUploadState,
    refreshDocuments,
  } = useDocuments(activeConversationId, activeConversation?.selected_document_id);

  const [authView, setAuthView] = React.useState<"login" | "register">("login");
  const [isLoggingOut, setIsLoggingOut] = React.useState(false);
  const [mobileTab, setMobileTab] = React.useState<"chats" | "documents" | "chat">("chat");

  // 1. Initial authentication loading state
  if (isLoading) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="flex min-h-screen items-center justify-center bg-background text-foreground"
      >
        <div className="flex flex-col items-center space-y-3">
          <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
          <p className="text-sm text-muted-foreground">Initializing secure session...</p>
        </div>
      </div>
    );
  }

  // 2. Unauthenticated views: Login / Register
  if (!user) {
    if (authView === "register") {
      return <RegisterPage onNavigateToLogin={() => setAuthView("login")} />;
    }
    return <LoginPage onNavigateToRegister={() => setAuthView("register")} />;
  }

  // 3. Protected Main Application View
  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      await logout();
    } finally {
      setIsLoggingOut(false);
      setAuthView("login");
    }
  };

  const handleSelectConversation = (id: string) => {
    selectConversation(id);
    setMobileTab("chat");
  };

  const handleNewConversation = async () => {
    const newConv = await createConversation("New Conversation");
    if (newConv) {
      setMobileTab("chat");
    }
  };

  const handleDeleteConversation = async (id: string) => {
    await deleteConversation(id);
  };

  const handleSelectDocument = (doc: DocumentItem) => {
    selectDocument(doc);
    setMobileTab("chat");
  };

  const handleDeleteDocument = async (docId: string) => {
    await deleteDocument(docId);
  };

  const handleAttachDocument = async (docId: string) => {
    await attachDocument(docId);
  };

  const handleSendMessage = (content: string) => {
    if (selectedDocument) {
      sendMessage(selectedDocument.id, content);
    }
  };

  const handleTriggerUpload = () => {
    setMobileTab("documents");
    setTimeout(() => {
      document.getElementById("document-file-input")?.click();
    }, 50);
  };

  const displayName = user.full_name || user.email.split("@")[0];

  return (
    <div className="flex min-h-screen flex-col bg-background font-sans text-foreground">
      {/* Top Application Header with rebalanced spacing */}
      <header
        role="banner"
        className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
      >
        <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
          {/* Logo & Product Title */}
          <div className="flex items-center space-x-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow">
              <Sparkles className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <h1 className="text-lg font-bold leading-none tracking-tight">
                Vectorless RAG
              </h1>
              <p className="text-xs text-muted-foreground mt-0.5 hidden sm:block">
                Hybrid Lexical &amp; Vector Document Q&amp;A
              </p>
            </div>
          </div>

          {/* Right Header Navigation & Actions: ThemeToggle, Account info, Logout button */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Light / Dark Mode Theme Toggle */}
            <ThemeToggle />

            {/* Authenticated User Status */}
            <div className="flex items-center gap-2 rounded-lg border border-border bg-card/60 px-2.5 py-1.5 text-xs text-card-foreground shadow-2xs">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-primary">
                <UserIcon className="h-3.5 w-3.5" aria-hidden="true" />
              </div>
              <div className="hidden md:block text-left">
                <p className="font-semibold leading-none">{displayName}</p>
                <p className="text-[10px] text-muted-foreground">{user.email}</p>
              </div>
            </div>

            {/* Logout Action */}
            <Button
              variant="outline"
              size="sm"
              onClick={handleLogout}
              disabled={isLoggingOut}
              className="h-8 gap-1.5 text-xs cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
              aria-label="Log out of application"
            >
              {isLoggingOut ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              ) : (
                <LogOut className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
              )}
              <span className="hidden md:inline">Log out</span>
            </Button>
          </div>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <main
        role="main"
        className="flex-1 container mx-auto px-4 py-4 md:px-6 lg:py-6"
      >
        {/* Responsive Mobile Navigation Tab Bar (Visible on mobile screens < 1024px) */}
        <nav
          aria-label="Workspace mobile views"
          className="flex lg:hidden mb-4 p-1 rounded-xl bg-muted/70 border border-border"
        >
          <button
            type="button"
            onClick={() => setMobileTab("chats")}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors flex items-center justify-center gap-1.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 ${
              mobileTab === "chats"
                ? "bg-card text-foreground shadow-xs border border-border/80"
                : "text-muted-foreground hover:text-foreground"
            }`}
            aria-pressed={mobileTab === "chats"}
            aria-label={`View Chats list (${conversations.length} available)`}
          >
            <MessagesSquare className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Chats ({conversations.length})</span>
          </button>
          <button
            type="button"
            onClick={() => setMobileTab("documents")}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors flex items-center justify-center gap-1.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 ${
              mobileTab === "documents"
                ? "bg-card text-foreground shadow-xs border border-border/80"
                : "text-muted-foreground hover:text-foreground"
            }`}
            aria-pressed={mobileTab === "documents"}
            aria-label={`View Documents list (${documents.length} available)`}
          >
            <FileText className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Docs ({documents.length})</span>
          </button>
          <button
            type="button"
            onClick={() => setMobileTab("chat")}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-colors flex items-center justify-center gap-1.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 ${
              mobileTab === "chat"
                ? "bg-card text-foreground shadow-xs border border-border/80"
                : "text-muted-foreground hover:text-foreground"
            }`}
            aria-pressed={mobileTab === "chat"}
            aria-label="View Chat & Question Answering"
          >
            <MessageSquare className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Chat</span>
            {selectedDocument && (
              <span
                className="h-2 w-2 rounded-full bg-emerald-500 shrink-0"
                aria-label={`Active document: ${selectedDocument.filename}`}
              />
            )}
          </button>
        </nav>

        {/* Screen-reader live announcement of active tab */}
        <div className="sr-only" role="status" aria-live="polite">
          Currently displaying {mobileTab === "chats" ? "chat conversations list" : mobileTab === "documents" ? "documents repository" : "chat panel"}.
        </div>

        {/* Workspace Columns (3-column layout on desktop) */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Column 1: Conversations Sidebar (3 columns on lg) */}
          <aside
            aria-label="Conversations Sidebar"
            className={`lg:col-span-3 space-y-4 ${
              mobileTab === "chats" ? "block" : "hidden lg:block"
            }`}
          >
            <ChatSidebar
              conversations={conversations}
              activeConversationId={activeConversationId}
              isLoading={isLoadingConversations}
              onSelectConversation={handleSelectConversation}
              onNewChat={handleNewConversation}
              onDeleteConversation={handleDeleteConversation}
            />
          </aside>

          {/* Column 2: Document Ingestion & Repository (4 columns on lg) */}
          <aside
            aria-label="Document Management"
            className={`lg:col-span-4 space-y-6 ${
              mobileTab === "documents" ? "block" : "hidden lg:block"
            }`}
          >
            <DocumentUpload
              uploadState={uploadState}
              uploadError={uploadError}
              activeJob={activeJob}
              onUpload={uploadDocument}
              onReset={resetUploadState}
            />

            <DocumentList
              documents={documents}
              selectedDocument={selectedDocument}
              isLoading={isDocsLoading}
              error={docsError}
              onSelectDocument={handleSelectDocument}
              onDeleteDocument={handleDeleteDocument}
              onRefresh={refreshDocuments}
              onTriggerUpload={handleTriggerUpload}
              allUserDocuments={allUserDocuments}
              onAttachDocument={handleAttachDocument}
            />
          </aside>

          {/* Column 3: Chat & Conversation Session (5 columns on lg) */}
          <section
            aria-label="Conversation Thread and Question Answering"
            className={`lg:col-span-5 flex flex-col h-[calc(100vh-8.5rem)] min-h-[550px] ${
              mobileTab === "chat" ? "block" : "hidden lg:block"
            }`}
          >
            <ChatPanel
              selectedDocument={selectedDocument}
              activeConversation={activeConversation}
              messages={messages}
              isLoadingMessages={isLoadingMessages}
              isStreaming={isStreaming}
              streamingDelta={streamingDelta}
              error={chatError}
              onSendMessage={handleSendMessage}
              onStopStreaming={stopStreaming}
              onClearError={clearChatError}
              onNewChat={handleNewConversation}
              onSelectMobileTab={setMobileTab}
            />
          </section>
        </div>
      </main>
    </div>
  );
}
