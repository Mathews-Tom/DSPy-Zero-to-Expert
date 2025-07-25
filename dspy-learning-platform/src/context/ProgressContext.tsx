import React, { createContext, useContext, useState } from "react";
import type { Stage } from "../types";

interface ProgressCtx {
  stages: Stage[];
  toggleObjective: (stageId: number, objId: number) => void;
  setNotes: (stageId: number, text: string) => void;
  markStageComplete: (stageId: number, checked: boolean) => void;
}

const defaultStages: Stage[] = Array.from({ length: 7 }, (_, idx) => {
  const id = idx + 1;
  return {
    id,
    title: `Stage ${id}`,
    level: "TBD",
    duration: "TBD",
    description: "",
    objectives: [
      { id: 1, text: "Objective 1", done: false },
      { id: 2, text: "Objective 2", done: false },
      { id: 3, text: "Objective 3", done: false }
    ],
    notes: "",
    complete: false
  };
});

const ProgressContext = createContext<ProgressCtx | null>(null);

export const ProgressProvider: React.FC<{ children: React.ReactNode }> = ({
  children
}) => {
  const [stages, setStages] = useState<Stage[]>(defaultStages);

  const toggleObjective = (stageId: number, objId: number) => {
    setStages((prev) =>
      prev.map((s) =>
        s.id === stageId
          ? {
              ...s,
              objectives: s.objectives.map((o) =>
                o.id === objId ? { ...o, done: !o.done } : o
              )
            }
          : s
      )
    );
  };

  const setNotes = (stageId: number, text: string) => {
    setStages((prev) =>
      prev.map((s) => (s.id === stageId ? { ...s, notes: text } : s))
    );
  };

  const markStageComplete = (stageId: number, checked: boolean) => {
    setStages((prev) =>
      prev.map((s) => (s.id === stageId ? { ...s, complete: checked } : s))
    );
  };

  return (
    <ProgressContext.Provider
      value={{ stages, toggleObjective, setNotes, markStageComplete }}
    >
      {children}
    </ProgressContext.Provider>
  );
};

export const useProgress = () => {
  const ctx = useContext(ProgressContext);
  if (!ctx) throw new Error("useProgress must be inside ProgressProvider");
  return ctx;
};
