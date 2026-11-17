import { CheckCircle2, AlertCircle, Info, AlertTriangle, X } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useToast, ToastType } from "@/lib/toast";

export function Toaster() {
  const { toasts, dismiss } = useToast();
  const shouldReduceMotion = useReducedMotion();

  const getIcon = (type: ToastType) => {
    switch (type) {
      case "success":
        return <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" aria-hidden="true" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" aria-hidden="true" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-amber-500 shrink-0 mt-0.5" aria-hidden="true" />;
      case "info":
      default:
        return <Info className="h-4 w-4 text-primary shrink-0 mt-0.5" aria-hidden="true" />;
    }
  };

  return (
    <div
      role="region"
      aria-label="Notifications"
      className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none px-4 sm:px-0"
    >
      <AnimatePresence mode="popLayout">
        {toasts.map((toast) => (
          <motion.div
            key={toast.id}
            layout={!shouldReduceMotion}
            initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 15, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: shouldReduceMotion ? 0 : 0.18, ease: "easeOut" }}
            role={toast.type === "error" ? "alert" : "status"}
            aria-live="polite"
            className="pointer-events-auto flex items-start justify-between gap-3 rounded-xl border border-border bg-card p-3.5 shadow-lg text-card-foreground text-xs"
          >
            <div className="flex items-start gap-2.5">
              {getIcon(toast.type)}
              <div className="space-y-0.5">
                <p className="font-semibold text-foreground leading-tight">{toast.title}</p>
                {toast.description && (
                  <p className="text-muted-foreground leading-normal">{toast.description}</p>
                )}
              </div>
            </div>

            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted cursor-pointer shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
              aria-label="Dismiss notification"
            >
              <X className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
