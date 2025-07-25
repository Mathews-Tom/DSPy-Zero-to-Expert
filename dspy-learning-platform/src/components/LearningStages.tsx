import { useProgress } from "../context/ProgressContext";

export default function LearningStages() {
  const { stages, toggleObjective, setNotes, markStageComplete } =
    useProgress();

  return (
    <div className="grid gap-6">
      {stages.map((stage) => (
        <article
          key={stage.id}
          className="border rounded shadow-sm p-4 space-y-4"
        >
          <header className="flex items-center justify-between">
            <h3 className="text-xl font-semibold">
              {stage.title}: Objective Placeholder
            </h3>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={stage.complete}
                onChange={(e) =>
                  markStageComplete(stage.id, e.currentTarget.checked)
                }
              />
              <span className="text-sm">Mark as complete</span>
            </label>
          </header>

          {/* Objectives */}
          <ul className="grid gap-2">
            {stage.objectives.map((obj) => (
              <li key={obj.id} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={obj.done}
                  onChange={() => toggleObjective(stage.id, obj.id)}
                />
                <span>{obj.text}</span>
              </li>
            ))}
          </ul>

          {/* Placeholder content blocks */}
          <div className="grid md:grid-cols-2 gap-4">
            <Placeholder title="Lesson explanation" />
            <Placeholder title="Python code example" code />
            <Placeholder title="Practice exercise" />
            <Placeholder title="Multiple-choice quiz" />
          </div>

          {/* Notes */}
          <div>
            <label className="block mb-1 font-medium text-sm">
              Personal notes
            </label>
            <textarea
              className="w-full border rounded p-2 min-h-[100px]"
              value={stage.notes}
              onChange={(e) => setNotes(stage.id, e.target.value)}
              placeholder="Write your notes here…"
            />
          </div>
        </article>
      ))}
    </div>
  );
}

function Placeholder({
  title,
  code
}: {
  title: string;
  code?: boolean;
}) {
  return (
    <div className="border rounded p-3 bg-gray-50 text-gray-600 text-sm">
      <p className="font-medium mb-1">{title}</p>
      <div
        className={
          code
            ? "bg-gray-900/90 text-green-300 font-mono text-xs p-2 rounded"
            : "italic"
        }
      >
        {code ? "# code block placeholder" : "Content coming soon…"}
      </div>
    </div>
  );
}
