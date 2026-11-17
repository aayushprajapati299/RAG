import * as React from "react";

export function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      role="status"
      aria-label="Loading content..."
      className={`animate-pulse motion-reduce:animate-none rounded-md bg-muted/70 ${className || ""}`}
      {...props}
    />
  );
}
