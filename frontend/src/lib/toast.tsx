import * as React from "react";

export type ToastType = "success" | "error" | "info" | "warning";

export interface ToastItem {
  id: string;
  type: ToastType;
  title: string;
  description?: string;
  duration?: number;
}

interface ToastContextType {
  toasts: ToastItem[];
  toast: {
    (options: Omit<ToastItem, "id">): string;
    success: (title: string, description?: string) => string;
    error: (title: string, description?: string) => string;
    info: (title: string, description?: string) => string;
    warning: (title: string, description?: string) => string;
  };
  dismiss: (id: string) => void;
}

const ToastContext = React.createContext<ToastContextType | undefined>(undefined);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<ToastItem[]>([]);

  const dismiss = React.useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = React.useCallback(
    (options: Omit<ToastItem, "id">): string => {
      const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
      const duration = options.duration ?? (options.type === "error" ? 6000 : 4000);

      const newToast: ToastItem = {
        ...options,
        id,
        duration,
      };

      setToasts((prev) => [...prev, newToast]);

      if (duration > 0) {
        setTimeout(() => {
          dismiss(id);
        }, duration);
      }

      return id;
    },
    [dismiss]
  );

  const toastMethods = React.useMemo(() => {
    const fn = (options: Omit<ToastItem, "id">) => addToast(options);
    fn.success = (title: string, description?: string) =>
      addToast({ type: "success", title, description });
    fn.error = (title: string, description?: string) =>
      addToast({ type: "error", title, description });
    fn.info = (title: string, description?: string) =>
      addToast({ type: "info", title, description });
    fn.warning = (title: string, description?: string) =>
      addToast({ type: "warning", title, description });
    return fn;
  }, [addToast]);

  const value = React.useMemo(
    () => ({
      toasts,
      toast: toastMethods,
      dismiss,
    }),
    [toasts, toastMethods, dismiss]
  );

  return <ToastContext.Provider value={value}>{children}</ToastContext.Provider>;
}

export function useToast(): ToastContextType {
  const context = React.useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}
