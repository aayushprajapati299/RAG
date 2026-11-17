import * as React from "react";
import { User, authApi, ApiError } from "@/lib/api";

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  clearError: () => void;
}

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<User | null>(null);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [error, setError] = React.useState<string | null>(null);

  const refreshUser = React.useCallback(async () => {
    try {
      const currentUser = await authApi.getMe();
      setUser(currentUser);
    } catch (err) {
      setUser(null);
    }
  }, []);

  // Load current user from /api/auth/me on app startup
  React.useEffect(() => {
    let isMounted = true;

    async function initializeAuth() {
      try {
        const currentUser = await authApi.getMe();
        if (isMounted) {
          setUser(currentUser);
        }
      } catch {
        // 401 unauthenticated is normal when not logged in
        if (isMounted) {
          setUser(null);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    initializeAuth();

    return () => {
      isMounted = false;
    };
  }, []);

  const login = async (email: string, password: string) => {
    setError(null);
    try {
      const loggedInUser = await authApi.login({ email, password });
      setUser(loggedInUser);
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Failed to sign in. Please verify your credentials.";
      setError(message);
      throw err;
    }
  };

  const register = async (email: string, password: string, fullName?: string) => {
    setError(null);
    try {
      const newUser = await authApi.register({
        email,
        password,
        full_name: fullName?.trim() || undefined,
      });
      setUser(newUser);
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Registration failed. Please check the inputs.";
      setError(message);
      throw err;
    }
  };

  const logout = async () => {
    setError(null);
    try {
      await authApi.logout();
    } catch {
      // Clear user regardless of server response
    } finally {
      setUser(null);
    }
  };

  const clearError = () => setError(null);

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        error,
        login,
        register,
        logout,
        refreshUser,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = React.useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
