import * as React from "react";
import { Sparkles, AlertCircle, ArrowRight, Loader2 } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

interface RegisterPageProps {
  onNavigateToLogin: () => void;
}

export default function RegisterPage({ onNavigateToLogin }: RegisterPageProps) {
  const { register, error: serverError, clearError } = useAuth();
  const [fullName, setFullName] = React.useState("");
  const [email, setEmail] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [confirmPassword, setConfirmPassword] = React.useState("");
  const [clientError, setClientError] = React.useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setClientError(null);
    clearError();

    const cleanEmail = email.trim();
    if (!cleanEmail) {
      setClientError("Please enter your email address.");
      return;
    }

    if (!cleanEmail.includes("@") || !cleanEmail.includes(".")) {
      setClientError("Please enter a valid email address.");
      return;
    }

    if (!password) {
      setClientError("Please enter a password.");
      return;
    }

    if (password.length < 8) {
      setClientError("Password must be at least 8 characters in length.");
      return;
    }

    if (password.length > 72) {
      setClientError("Password cannot exceed 72 characters.");
      return;
    }

    if (password !== confirmPassword) {
      setClientError("Passwords do not match. Please re-enter.");
      return;
    }

    setIsSubmitting(true);
    try {
      await register(cleanEmail, password, fullName.trim() || undefined);
    } catch {
      // Error handled by AuthContext
    } finally {
      setIsSubmitting(false);
    }
  };

  const displayError = clientError || serverError;

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-background px-4 py-12 sm:px-6 lg:px-8">
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>
      <Card className="w-full max-w-md shadow-md border-border">
        <CardHeader className="space-y-2 text-center pb-6">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow">
            <Sparkles className="h-6 w-6" aria-hidden="true" />
          </div>
          <CardTitle className="text-2xl font-bold tracking-tight">
            Create an Account
          </CardTitle>
          <CardDescription className="text-sm text-muted-foreground">
            Sign up to upload documents and query them with hybrid search
          </CardDescription>
        </CardHeader>

        <form onSubmit={handleSubmit} noValidate>
          <CardContent className="space-y-4">
            {displayError && (
              <div
                role="alert"
                aria-live="polite"
                className="flex items-start gap-2.5 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-xs text-destructive"
              >
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" aria-hidden="true" />
                <span>{displayError}</span>
              </div>
            )}

            <div className="space-y-1.5">
              <label
                htmlFor="reg-name"
                className="text-xs font-semibold leading-none text-foreground"
              >
                Full Name <span className="font-normal text-muted-foreground">(Optional)</span>
              </label>
              <Input
                id="reg-name"
                name="fullName"
                type="text"
                autoComplete="name"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Ada Lovelace"
                disabled={isSubmitting}
                className="h-10"
              />
            </div>

            <div className="space-y-1.5">
              <label
                htmlFor="reg-email"
                className="text-xs font-semibold leading-none text-foreground"
              >
                Email address
              </label>
              <Input
                id="reg-email"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  if (clientError) setClientError(null);
                }}
                placeholder="name@example.com"
                disabled={isSubmitting}
                className="h-10"
              />
            </div>

            <div className="space-y-1.5">
              <label
                htmlFor="reg-password"
                className="text-xs font-semibold leading-none text-foreground"
              >
                Password <span className="font-normal text-muted-foreground">(min 8 characters)</span>
              </label>
              <Input
                id="reg-password"
                name="password"
                type="password"
                autoComplete="new-password"
                required
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  if (clientError) setClientError(null);
                }}
                placeholder="••••••••"
                disabled={isSubmitting}
                className="h-10"
              />
            </div>

            <div className="space-y-1.5">
              <label
                htmlFor="reg-confirm-password"
                className="text-xs font-semibold leading-none text-foreground"
              >
                Confirm Password
              </label>
              <Input
                id="reg-confirm-password"
                name="confirmPassword"
                type="password"
                autoComplete="new-password"
                required
                value={confirmPassword}
                onChange={(e) => {
                  setConfirmPassword(e.target.value);
                  if (clientError) setClientError(null);
                }}
                placeholder="••••••••"
                disabled={isSubmitting}
                className="h-10"
              />
            </div>
          </CardContent>

          <CardFooter className="flex flex-col space-y-4 pt-2">
            <Button
              type="submit"
              disabled={isSubmitting}
              className="w-full h-10 font-medium"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                  Creating account...
                </>
              ) : (
                <>
                  Register
                  <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
                </>
              )}
            </Button>

            <div className="text-center text-xs text-muted-foreground">
              Already have an account?{" "}
              <button
                type="button"
                onClick={() => {
                  clearError();
                  onNavigateToLogin();
                }}
                className="font-semibold text-primary underline-offset-4 hover:underline cursor-pointer"
              >
                Sign in instead
              </button>
            </div>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
