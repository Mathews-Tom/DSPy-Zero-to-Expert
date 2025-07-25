import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./components/Overview";
import LearningStages from "./components/LearningStages";
import Assessments from "./components/Assessments";
import Projects from "./components/Projects";
import Resources from "./components/Resources";
import DesignGuide from "./components/DesignGuide";

const tabs = [
  { to: "/", label: "Overview" },
  { to: "/stages", label: "Learning Stages" },
  { to: "/assessments", label: "Assessments" },
  { to: "/projects", label: "Projects" },
  { to: "/resources", label: "Resources" }
];

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-primary-600 text-white">
        <div className="max-w-6xl mx-auto px-4 py-3 text-xl font-semibold">
          DSPy Zero-to-Expert
        </div>
      </header>

      <nav className="bg-primary-50 shadow-inner">
        <div className="max-w-6xl mx-auto flex gap-2 px-4">
          {tabs.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `px-3 py-2 text-sm font-medium rounded-t ${
                  isActive
                    ? "bg-white text-primary-700 border-b-2 border-primary-600"
                    : "text-gray-600 hover:text-primary-700"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </div>
      </nav>

      <main className="flex-1 max-w-6xl mx-auto p-4 bg-white">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/stages" element={<LearningStages />} />
          <Route path="/assessments" element={<Assessments />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/resources" element={<Resources />} />
          <Route path="/design-guide" element={<DesignGuide />} />
        </Routes>
      </main>

      <footer className="text-center text-xs py-4 text-gray-500">
        © {new Date().getFullYear()} DSPy Learning Platform Scaffold
      </footer>
    </div>
  );
}
