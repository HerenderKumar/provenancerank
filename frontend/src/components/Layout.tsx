import { LayoutDashboard, ListOrdered, LogOut, Search, Settings, Sparkles } from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

const nav = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/search", label: "Evidence Search", icon: Search, end: false },
  { to: "/results", label: "Rankings", icon: ListOrdered, end: false },
  { to: "/settings", label: "Settings", icon: Settings, end: false },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen">
      <aside className="flex w-60 flex-col border-r border-edge bg-ink/50 p-4">
        <div className="mb-8 flex items-center gap-2 px-2">
          <Sparkles className="text-brand" size={22} />
          <div>
            <div className="font-semibold leading-tight">ProvenanceRank</div>
            <div className="text-[10px] uppercase tracking-widest text-muted">recruiter console</div>
          </div>
        </div>
        <nav className="flex flex-col gap-1">
          {nav.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  isActive ? "bg-brand/15 text-white" : "text-muted hover:text-white hover:bg-panel"
                }`
              }
            >
              <n.icon size={16} /> {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto">
          <div className="card mb-2 text-xs">
            <div className="text-white">{user?.email}</div>
            <div className="text-muted">{user?.role}</div>
          </div>
          <button
            className="btn-ghost w-full justify-center"
            onClick={() => {
              logout();
              navigate("/login");
            }}
          >
            <LogOut size={14} /> Sign out
          </button>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-6">{children}</main>
    </div>
  );
}
