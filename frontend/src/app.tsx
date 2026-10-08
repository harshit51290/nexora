import { HashRouter, NavLink, Route, Routes } from "react-router-dom";
import { useSettings } from "./stores/settings";
import Home from "./pages/Home";
import Discover from "./pages/Discover";
import Models from "./pages/Models";
import Workflows from "./pages/Workflows";
import Generate from "./pages/Generate";
import Downloads from "./pages/Downloads";
import Environments from "./pages/Environments";
import Hardware from "./pages/Hardware";
import Runtime from "./pages/Runtime";
import Settings from "./pages/Settings";

const NAV = [
  { to: "/", label: "Home" },
  { to: "/discover", label: "Discover" },
  { to: "/models", label: "Models" },
  { to: "/workflows", label: "Workflows" },
  { to: "/generate", label: "Generate" },
  { to: "/downloads", label: "Downloads" },
  { to: "/environments", label: "Environments" },
  { to: "/hardware", label: "Hardware" },
  { to: "/runtime", label: "Runtime" },
  { to: "/settings", label: "Settings" },
] as const;

function ModeSwitcher() {
  const { mode, setMode } = useSettings();
  return (
    <div className="flex gap-1 rounded-lg bg-neutral-800 p-1 text-xs">
      {(["beginner", "advanced", "developer"] as const).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className={
            mode === m
              ? "rounded-md bg-indigo-600 px-2 py-1 font-medium capitalize text-white"
              : "rounded-md px-2 py-1 capitalize text-neutral-400 hover:text-white"
          }
        >
          {m}
        </button>
      ))}
    </div>
  );
}

export default function App() {
  return (
    <HashRouter>
      <div className="flex min-h-screen">
        <aside className="flex w-52 shrink-0 flex-col gap-4 border-r border-neutral-800 bg-neutral-900 p-4">
          <div>
            <h1 className="text-lg font-bold">Nexora</h1>
            <p className="text-xs text-neutral-500">Local AI runtime manager</p>
          </div>
          <nav className="flex flex-col gap-1">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.to === "/"}
                className={({ isActive }) =>
                  isActive ? "nav-link-active" : "nav-link"
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="mt-auto">
            <ModeSwitcher />
          </div>
        </aside>
        <main className="min-w-0 flex-1 p-6">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/discover" element={<Discover />} />
            <Route path="/models" element={<Models />} />
            <Route path="/workflows" element={<Workflows />} />
            <Route path="/generate" element={<Generate />} />
            <Route path="/downloads" element={<Downloads />} />
            <Route path="/environments" element={<Environments />} />
            <Route path="/hardware" element={<Hardware />} />
            <Route path="/runtime" element={<Runtime />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}
