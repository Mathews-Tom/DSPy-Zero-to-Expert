export default function DesignGuide() {
  return (
    <article className="prose lg:prose-lg max-w-none">
      <h1>Internal Design Guide</h1>

      <h2>Tabs &amp; Navigation</h2>
      <p>
        Use <code>.bg-primary-50</code> for inactive backgrounds and{" "}
        <code>.bg-white .border-b-2 .border-primary-600</code> to indicate the
        active tab.
      </p>

      <h2>Stage Cards</h2>
      <pre className="font-mono bg-gray-100 p-3 rounded">
{`<article class="border rounded p-4 shadow-sm">...</article>`}
      </pre>

      <h2>Buttons</h2>
      <div className="flex gap-3">
        <button className="btn-primary">Primary</button>
        <button className="btn-outline">Outline</button>
      </div>

      <h2>Typography Examples</h2>
      <p className="text-xl font-semibold">.text-xl .font-semibold</p>
      <p className="text-sm text-gray-500">.text-sm .text-gray-500</p>

      <h2>Code Block Styling</h2>
      <pre className="bg-gray-900 text-green-300 p-2 rounded text-xs">
{`# Sample code\nprint("Hello, DSPy")`}
      </pre>

      <h2>Inputs</h2>
      <input
        className="border p-2 rounded w-full mb-2"
        placeholder="Input example"
      />
      <textarea
        className="border p-2 rounded w-full"
        rows={3}
        placeholder="Textarea example"
      />
    </article>
  );
}
