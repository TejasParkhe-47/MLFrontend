import React, { useState, useMemo } from "react";
import {
  Loader2,
  Upload,
  Zap,
  BarChart3,
  TrendingUp,
  Cpu,
  CheckCircle,
  FileText,
} from "lucide-react";
import ReactFlow, { MiniMap, Controls, Background } from "reactflow";
import "reactflow/dist/style.css";

// --- API URLs ---
const TRAIN_API_URL = "http://localhost:5000/train";
const PREDICTION_API_URL = "http://localhost:5000/predict";
const STRUCTURE_API_URL = "http://localhost:5000/structure";
const FORMATTED_STRUCTURE_API_URL = "http://localhost:5000/formatted-structure";

// --- Loader Component ---
const Loader = ({ message, icon: Icon = Loader2 }) => (
  <div className="flex flex-col items-center justify-center p-8 bg-indigo-50 rounded-xl shadow-inner border-2 border-indigo-200">
    <Icon className="animate-spin w-10 h-10 text-indigo-600 mb-3" />
    <p className="text-xl font-bold text-gray-700">{message}</p>
    <p className="text-sm text-gray-500 mt-1">
      Please wait while the server processes the request...
    </p>
  </div>
);

// --- Classification Report Component ---
const ClassificationReport = ({ report }) => {
  if (!report) return null;
  const classKeys = Object.keys(report).filter((k) => !isNaN(parseInt(k)));
  const avgKeys = ["macro avg", "weighted avg"];
  const formatValue = (v) =>
    typeof v === "number" ? (v * 100).toFixed(2) + "%" : v;

  const renderRow = (keys) =>
    keys.map((key) => {
      const m = report[key];
      return (
        <tr
          key={key}
          className={
            key.includes("avg")
              ? "bg-indigo-100 font-semibold text-indigo-800"
              : "hover:bg-green-50"
          }
        >
          <td className="p-3 border border-gray-200">
            {key.toUpperCase().replace(" AVG", " Average")}
          </td>
          <td className="p-3 border border-gray-200">{formatValue(m.precision)}</td>
          <td className="p-3 border border-gray-200">{formatValue(m.recall)}</td>
          <td className="p-3 border border-gray-200">{formatValue(m["f1-score"])}</td>
          <td className="p-3 border border-gray-200">{m.support}</td>
        </tr>
      );
    });

  return (
    <div className="mt-4">
      <div className="mb-6 bg-indigo-50 p-5 rounded-xl shadow-md flex justify-between items-center border border-indigo-200">
        <div>
          <p className="text-md font-medium text-indigo-800 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" /> Overall Model Accuracy
          </p>
          <p className="text-5xl font-extrabold text-indigo-700 mt-1">
            {(report.accuracy * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      <h3 className="text-xl font-semibold mb-3 text-gray-700">Detailed Metrics</h3>
      <div className="overflow-x-auto border rounded-xl shadow-sm">
        <table className="min-w-full border-collapse text-sm">
          <thead>
            <tr className="bg-gray-100 text-left text-gray-600 font-bold">
              <th className="p-3 border border-gray-200">Class/Avg</th>
              <th className="p-3 border border-gray-200">Precision</th>
              <th className="p-3 border border-gray-200">Recall</th>
              <th className="p-3 border border-gray-200">F1-Score</th>
              <th className="p-3 border border-gray-200">Support</th>
            </tr>
          </thead>
          <tbody>
            {renderRow(classKeys)}
            {renderRow(avgKeys)}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// --- Structure Viewer (Formatted Python-style Text) ---
const StructureViewer = ({ data }) => {
  if (!data) return null;

  return (
    <div className="mt-10 bg-white p-6 rounded-2xl border border-indigo-200 shadow-md">
      <h2 className="text-2xl font-extrabold text-indigo-700 mb-4 flex items-center gap-2">
        <FileText className="w-6 h-6 text-indigo-500" /> Decision Tree Structure
      </h2>
      <p className="text-sm text-gray-600 mb-6">
        Below is the detailed breakdown of your trained decision trees,
        including thresholds, Gini impurities, and information gain calculations.
      </p>
      <pre className="whitespace-pre-wrap font-mono text-gray-900 bg-gray-50 p-5 rounded-xl border border-gray-200 shadow-inner max-h-[700px] overflow-y-auto text-sm leading-relaxed">
        {data}
      </pre>
    </div>
  );
};

// --- Decision Tree Flow Visualization ---
const DecisionTreeFlow = ({ treeId, flow }) => {
  const yStep = 220;
  const xStep = 500;
  const totalHeight = Math.max(800, flow.length * yStep + 300);
  const { nodes, edges } = useMemo(() => {
    const nodes = [];
    const edges = [];
    flow.forEach((step, index) => {
      const [feature, operator, threshold, next] = step;
      const isLeaf = typeof next === "number";
      const id = `n-${treeId}-${index}`;
      nodes.push({
        id,
        data: {
          label: isLeaf
            ? `🌿 Leaf → Class ${next}`
            : `${feature} ${operator} ${threshold}`,
        },
        position: { x: (index % 2) * xStep - xStep / 2, y: index * yStep },
        style: {
          background: isLeaf ? `${next? "#00ff00" : "#ff0000"}` : "#eef2ff",
          border: "3px solid #6366f1",
          borderRadius: 12,
          padding: 10,
          width: 300,
          fontWeight: 600,
          textAlign: "center",
          boxShadow: "0 6px 14px rgba(0,0,0,0.15)",
        },
      });
      if (index > 0) {
        edges.push({
          id: `e-${treeId}-${index}`,
          source: `n-${treeId}-${index - 1}`,
          target: id,
          animated: true,
          style: { stroke: "#4f46e5", strokeWidth: 3 },
        });
      }
    });
    return { nodes, edges };
  }, [treeId, flow]);

  return (
    <div
      className="border border-indigo-200 rounded-xl overflow-hidden shadow-xl bg-slate-50"
      style={{ height: totalHeight }}
    >
      <h4 className="font-bold text-lg text-indigo-700 px-4 pt-3">
        🌳 Decision Tree #{treeId}
      </h4>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.9 }}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        panOnScroll
        zoomOnScroll
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnPinch
        minZoom={0.2}
        maxZoom={2}
      >
        <MiniMap />
        <Controls showInteractive={false} />
        <Background color="#e5e7eb" gap={20} />
      </ReactFlow>
    </div>
  );
};

// --- Main App Component ---
const App = () => {
  const [file, setFile] = useState(null);
  const [features, setFeatures] = useState([]);
  const [formData, setFormData] = useState({});
  const [trainingReport, setTrainingReport] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [structureData, setStructureData] = useState(null);
  const [loadingState, setLoadingState] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setTrainingReport(null);
    setPredictionResult(null);
    setStructureData(null);
  };

  const handleTrain = async () => {
    if (!file) return alert("Please upload a CSV file!");
    setLoadingState("training");
    setPredictionResult(null);
    setTrainingReport(null);

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(TRAIN_API_URL, { method: "POST", body: form });
      const data = await res.json();
      if (data.error) return alert(data.error);

      const report = data.classification_report || data;
      setTrainingReport(report);
      setFeatures(data.features || []);
      const init = {};
      (data.features || []).forEach((f) => (init[f] = ""));
      setFormData(init);

      // --- Fetch formatted structure from backend ---
      const formattedRes = await fetch(FORMATTED_STRUCTURE_API_URL);
      const formattedJson = await formattedRes.json();
      if (formattedJson.formatted_structure)
        setStructureData(formattedJson.formatted_structure);
    } catch (err) {
      console.error(err);
      alert("Error during training.");
    } finally {
      setLoadingState(null);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!trainingReport) return alert("Train the model first!");
    setLoadingState("predicting");
    setPredictionResult(null);

    try {
      const payload = Object.fromEntries(
        Object.entries(formData).map(([k, v]) => [k, parseFloat(v) || 0])
      );
      const res = await fetch(PREDICTION_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setPredictionResult({
        prediction: data.prediction,
        probabilities: Array.isArray(data.probabilities)
          ? data.probabilities.map((p) => parseFloat(p))
          : [parseFloat(data.probabilities), 1 - parseFloat(data.probabilities)],
        flow: data.flow,
      });
    } catch (err) {
      console.error(err);
      alert("Error during prediction.");
    } finally {
      setLoadingState(null);
    }
  };

  return (
    <div className="p-8 font-sans max-w-7xl mx-auto bg-gray-50 min-h-screen">
      <header className="mb-10 border-b-4 border-indigo-200 pb-4">
        <h1 className="text-4xl font-extrabold text-indigo-700 flex items-center gap-3">
          <Zap className="w-8 h-8 text-yellow-500" /> MarketMind - Marketing Success Predictor
        </h1>
        <p className="text-gray-600 mt-2">
          Upload a CSV to train your model, visualize every decision tree, and view the learned structure.
        </p>
      </header>

      {/* Training Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <section className="p-6 rounded-xl shadow-lg bg-white border-2 border-indigo-200">
          <h2 className="text-2xl font-semibold mb-4 text-indigo-600 flex items-center gap-2 border-b pb-2">
            <Cpu className="text-indigo-500 w-6 h-6" /> 1. Train Model
          </h2>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
          />
          {file && (
            <p className="text-xs text-gray-400 mb-4 truncate">File: {file.name}</p>
          )}
          <button
            onClick={handleTrain}
            disabled={!file || loadingState === "training"}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700"
          >
            {loadingState === "training" ? (
              <Loader2 className="animate-spin w-5 h-5" />
            ) : (
              <Upload className="w-5 h-5" />
            )}
            {loadingState === "training" ? "Training..." : "Start Training"}
          </button>
        </section>

        {/* Prediction Section */}
        <section className="lg:col-span-2 p-6 rounded-xl shadow-lg bg-white">
          <h2 className="text-2xl font-semibold mb-4 text-indigo-600 flex items-center gap-2 border-b pb-2">
            <TrendingUp className="text-indigo-500 w-6 h-6" /> 2. Input & Predict
          </h2>
          <form onSubmit={handlePredict}>
            {features.length > 0 && (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mb-4 p-4 bg-gray-50 rounded-lg border">
                {features.map((f) => (
                  <div key={f} className="flex flex-col">
                    <label
                      htmlFor={f}
                      className="text-xs font-medium text-gray-500 truncate mb-1"
                    >
                      {f}
                    </label>
                    <input
                      id={f}
                      type="number"
                      step="any"
                      value={formData[f]}
                      onChange={(e) =>
                        setFormData({ ...formData, [f]: e.target.value })
                      }
                      required
                      className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                ))}
              </div>
            )}
            <button
              type="submit"
              disabled={!trainingReport || loadingState === "predicting"}
              className="w-full flex items-center justify-center gap-2 p-3 text-lg font-bold rounded-lg bg-green-600 text-white hover:bg-green-700"
            >
              {loadingState === "predicting" ? (
                <Loader2 className="animate-spin w-6 h-6" />
              ) : (
                <Zap className="w-6 h-6" />
              )}
              {loadingState === "predicting" ? "Predicting..." : "Get Prediction"}
            </button>
          </form>
        </section>
      </div>

      {/* --- Training Report and Structure --- */}
      {trainingReport && (
        <section className="p-8 bg-white rounded-xl shadow-2xl border-t-4 border-green-500 mt-10">
          <h2 className="text-3xl font-extrabold text-green-700 mb-4 flex items-center gap-3">
            <CheckCircle className="w-7 h-7" /> Model Training Complete
          </h2>
          <ClassificationReport report={trainingReport} />
          <StructureViewer data={structureData} />
        </section>
      )}

      {/* --- Prediction Visualization --- */}
      {predictionResult && (
        <section className="p-8 bg-white rounded-xl shadow-2xl border-t-4 border-yellow-500 mt-10">
          <h2 className="text-3xl font-extrabold text-yellow-700 mb-4 flex items-center gap-3">
            Prediction & Full Tree Visualization
          </h2>
          <div
            className={`mb-6 p-5 rounded-xl flex items-center justify-between shadow-lg ${
              predictionResult.prediction === 1
                ? "bg-blue-100 border-l-4 border-blue-500"
                : "bg-blue-100 border-l-4 border-blue-500"
            }`}
          >
            <p className="text-xl font-bold">
              Predicted Outcome:{" "}
              <span
                className={`text-3xl ${
                  predictionResult.prediction === 1
                    ? "text-blue-500"
                    : "text-blue-500"
                }`}
              >
                {predictionResult.prediction === 1? "Marketing Success" : "Marketing Failed"}
              </span>
            </p>
            
          </div>

          <div className="space-y-16">
            {Object.entries(predictionResult.flow).map(([treeId, flow]) => (
              <DecisionTreeFlow key={treeId} treeId={treeId} flow={flow} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
};

export default App;
