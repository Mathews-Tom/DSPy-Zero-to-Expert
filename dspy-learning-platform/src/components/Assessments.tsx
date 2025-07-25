import { useProgress } from "../context/ProgressContext";

export default function Assessments() {
  const { stages } = useProgress();
  const completed = stages.filter((s) => s.complete).length;
  const pct = Math.round((completed / stages.length) * 100);

  return (
    <section className="space-y-6">
      <h2 className="text-2xl font-semibold">Assessments Overview</h2>

      <div className="flex flex-col gap-4">
        <p>
          This page will ultimately host stage-level quizzes and project
          rubrics. For now it shows aggregate progress based on the checkboxes
          you tick in the Learning Stages tab.
        </p>

        <div>
          <p className="font-medium mb-1">
            Overall progress: {completed}/{stages.length} stages (
            {pct}%)
          </p>
          <div className="h-3 w-full bg-gray-200 rounded">
            <div
              style={{ width: `${pct}%` }}
              className="h-full bg-primary-600 rounded"
            />
          </div>
        </div>
      </div>
    </section>
  );
}
