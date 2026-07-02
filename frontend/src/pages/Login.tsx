import { Sparkles } from "lucide-react";
import { useState } from "react";
import { Navigate } from "react-router-dom";
import { Spinner } from "../components/ui";
import { useAuth } from "../lib/auth";

export default function Login() {
  const { user, login } = useAuth();
  const [email, setEmail] = useState("admin@provenancerank.local");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await login(email, password);
    } catch (e) {
      setErr((e as Error).message || "login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-screen place-items-center p-6">
      <form onSubmit={onSubmit} className="card w-full max-w-sm">
        <div className="mb-6 flex items-center gap-2">
          <Sparkles className="text-brand" />
          <div className="text-lg font-semibold">ProvenanceRank</div>
        </div>
        <label className="label">Email</label>
        <input className="input mt-1 mb-3" value={email} onChange={(e) => setEmail(e.target.value)} />
        <label className="label">Password</label>
        <input
          className="input mt-1 mb-4"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {err && <div className="mb-3 text-sm text-bad">{err}</div>}
        <button className="btn w-full justify-center" disabled={busy}>
          {busy ? <Spinner /> : "Sign in"}
        </button>
        <p className="mt-3 text-center text-xs text-muted">
          Default admin is bootstrapped on first boot — see BOOTSTRAP_ADMIN_* env vars.
        </p>
      </form>
    </div>
  );
}
