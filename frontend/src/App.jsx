import React, { useState, useMemo,useRef, use } from "react";
import Papa from "papaparse";
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
import ReactFlow, { MiniMap, Controls, Background,MarkerType } from "reactflow";
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



// --- Structure Viewer ---
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
          background: isLeaf ? "#eef2ff" : "#eef2ff",
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

// --- Generate Proper Hierarchical Tree Nodes & Edges ---
// --- Generate Proper Hierarchical Tree Nodes & Edges (Fixed) ---
// --- Safe Tree Builder for ReactFlow ---




const TREE_HIERARCHY_API_URL = "http://localhost:5000/tree-hierarchy";

const TreeNode = ({ node }) => {
  if (!node) return null;

  if (node.leaf !== undefined) {
    return (
      <div className="bg-green-100 text-green-800 px-4 py-2 rounded-lg shadow-md border border-green-300 text-sm font-semibold">
        🌿 Leaf: {node.leaf}
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center">
      {/* Root node */}
      <div className="bg-indigo-100 text-indigo-800 px-4 py-2 rounded-lg shadow-md border border-indigo-300 font-semibold text-sm">
        {node.feature} ≤ {node.threshold.toFixed(2)}
      </div>

      {/* Branch connections */}
      <div className="flex justify-between w-full mt-4">
        <div className="flex flex-col items-center w-1/2">
          <div className="text-xs text-gray-500 mb-1">Left</div>
          <div className="border-t-2 border-gray-300 w-10 mb-1"></div>
          <TreeNode node={node.left} />
        </div>

        <div className="flex flex-col items-center w-1/2">
          <div className="text-xs text-gray-500 mb-1">Right</div>
          <div className="border-t-2 border-gray-300 w-10 mb-1"></div>
          <TreeNode node={node.right} />
        </div>
      </div>
    </div>
  );
};












// --- Main App Component ---
const App = () => {
  const predictRef = useRef(null);
  const [file, setFile] = useState(null);
  const [features, setFeatures] = useState([]);
  const [formData, setFormData] = useState({});
  const [trainingReport, setTrainingReport] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [structureData, setStructureData] = useState(null);
  const [loadingState, setLoadingState] = useState(null);
  const [datasetPreview, setDatasetPreview] = useState(null); // 👈 new state
  const [decisionTrees,setDecisionTrees] = useState(null)
  const [trainData, setTrainData] = useState(null);
const [testData, setTestData] = useState(null);
const [trees, setTrees] = useState(null);
const [selectedTree, setSelectedTree] = useState(null);
const [treeData, setTreeData] = useState(null);
const [forestTrees, setForestTrees] = useState(null);
const [decisionPaths, setDecisionPaths] = useState({});

const [treeHierarchy,setTreeHierarchy] = useState(null)

const [testResults, setTestResults] = useState([]);
const [metrics, setMetrics] = useState(null);
const [isTesting, setIsTesting] = useState(false);


const handleTestData = async () => {
  if (!testData || testData.length === 0) {
    alert("No test data available. Train the model first!");
    return;
  }

  setIsTesting(true);
  let results = [];
  let yTrue = [];
  let yPred = [];

  for (let i = 0; i < testData.length; i++) {
    const row = testData[i];
    const actual = row.Response ?? null;

    try {
      const res = await fetch("http://127.0.0.1:5000/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(row),
      });
      const data = await res.json();

      if (data.prediction !== undefined) {
        const predicted = data.prediction;
        yTrue.push(actual);
        yPred.push(predicted);
        results.push({ ...row, Predicted: predicted, Actual: actual });
      }
    } catch (error) {
      console.error("Prediction error:", error);
    }
  }

  // Calculate metrics
  const TP = yTrue.filter((y, i) => y === 1 && yPred[i] === 1).length;
  const TN = yTrue.filter((y, i) => y === 0 && yPred[i] === 0).length;
  const FP = yTrue.filter((y, i) => y === 0 && yPred[i] === 1).length;
  const FN = yTrue.filter((y, i) => y === 1 && yPred[i] === 0).length;

  const accuracy = ((TP + TN) / yTrue.length) * 100;
  const precision = TP + FP === 0 ? 0 : (TP / (TP + FP)) * 100;
  const recall = TP + FN === 0 ? 0 : (TP / (TP + FN)) * 100;
  const f1 =
    precision + recall === 0
      ? 0
      : (2 * precision * recall) / (precision + recall);

  const metricsData = {
    Accuracy: accuracy.toFixed(2),
    Precision: precision.toFixed(2),
    Recall: recall.toFixed(2),
    "F1-Score": f1.toFixed(2),
    "True Positives": TP,
    "True Negatives": TN,
    "False Positives": FP,
    "False Negatives": FN,
  };

  setTestResults(results);
  setMetrics(metricsData);
  setIsTesting(false);
};





  

  // --- Handle CSV Upload + Parse ---
  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    setFile(uploadedFile);
    setTrainingReport(null);
    setPredictionResult(null);
    setStructureData(null);

    if (uploadedFile) {
      Papa.parse(uploadedFile, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          console.log("Parsed dataset:", results.data);
          setDatasetPreview(results.data.slice(0, 3000)); // preview first 3000 rows
        },
      });
    }
  };

  // --- Train Model ---
 // --- Train Model ---
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
    setFeatures(data.features || []);
    setTrainData(data.train_data || []);
    setTestData(data.test_data || []);

    const structureRes1 = await fetch("http://localhost:5000/forest-structure");
  const structureJson2 = await structureRes1.json();
  setForestTrees(structureJson2);


    // ⬇️ Fetch tree structure separately after training
    try {
      const structureRes = await fetch(STRUCTURE_API_URL);
const structureJson = await structureRes.json();
setTrees(structureJson); // keep JSON for later
setSelectedTree(1); // directly show one tree

    } catch (treeErr) {
      console.error("Error loading structure.json:", treeErr);
    }

    // Initialize prediction form fields
    const init = {};
    (data.features || []).forEach((f) => (init[f] = ""));
    setFormData(init);

    // Optional: fetch formatted structure (if supported)
    try {
      const formattedRes = await fetch(FORMATTED_STRUCTURE_API_URL);
      const formattedJson = await formattedRes.json();
      if (formattedJson.formatted_structure)
        setStructureData(formattedJson.formatted_structure);
    } catch (e) {
      console.warn("Formatted structure not available.");
    }
  } catch (err) {
    console.error(err);
    alert("Error during training.");
  } finally {
    setLoadingState(null);
  }
};

const handleRowToForm = (row) => {
  // Remove non-feature columns like 'Response', 'Predicted', 'Actual'
  const cleaned = Object.fromEntries(
    Object.entries(row).filter(
      ([key]) => !["Response", "Predicted", "Actual"].includes(key)
    )
  );

  // Convert values to numbers (optional)
  const formatted = Object.fromEntries(
    Object.entries(cleaned).map(([k, v]) => [k, parseFloat(v) || 0])
  );

  setFormData(formatted);
  predictRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
};


  // --- Predict ---
  const handlePredict = async (e) => {
    e.preventDefault();
  if (!features.length) return alert("Train the model first!"); //
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
      const treesResponse = await fetch(`http://localhost:5000/structure`);
const treesJson = await treesResponse.json();
setDecisionTrees(treesJson);
      const data = await res.json();
      setDecisionPaths(data.decision_paths || {});
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
          <Zap className="w-8 h-8 text-yellow-500" /> MarketMind - Traceable Ensemble Learning for Marketing Decisions
        </h1>
        <p className="text-gray-600 mt-2">
          Upload a CSV to train your model, visualize every decision tree, and view the learned structure.
        </p>
      </header>

      {/* --- Training Section --- */}
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
            className="w-full flex items-center justify-center gap-2 px-4 py-3 font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 mt-6"
          >
            {loadingState === "training" ? (
              <Loader2 className="animate-spin w-5 h-5" />
            ) : (
              <Upload className="w-5 h-5" />
            )}
            {loadingState === "training" ? "Training..." : "Start Training"}
          </button>
        </section>

        
      </div>

      
          {/* --- Dataset Preview --- */}
          {datasetPreview && (
        <section className="mt-8 p-6 rounded-xl shadow-lg bg-white border-2 border-indigo-200">
          <h2 className="text-2xl font-semibold mb-4 text-indigo-600 flex items-center gap-2 border-b pb-2">
            📊 Uploaded Dataset (Preview)
          </h2>
          <div className="overflow-x-auto border border-indigo-200 rounded-lg shadow-sm max-h-[600px] overflow-y-auto">
            <table className="min-w-full border-collapse text-sm">
              <thead className="bg-indigo-50 sticky top-0 z-10">
                <tr>
                  {/* --- NEW: Index Header Column --- */}
                  <th
                    key="Number"
                    className="p-3 border border-indigo-100 text-left font-semibold text-gray-600 whitespace-nowrap w-1/12"
                  >
                    Number
                  </th>
                  {/* ---------------------------------- */}
                  {Object.keys(datasetPreview[0]).map((col) => (
                    <th
                      key={col}
                      className="p-3 border border-indigo-100 text-left font-semibold text-gray-600 whitespace-nowrap"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {datasetPreview.map((row, i) => (
                  <tr key={i} className="hover:bg-indigo-50">
                    {/* --- NEW: Index Data Cell (i + 1) --- */}
                    <td className="p-3 border border-indigo-100 text-indigo-500 font-bold whitespace-nowrap">
                      {i + 1}
                    </td>
                    {/* --------------------------------------- */}
                    {Object.values(row).map((val, j) => (
                      <td
                        key={j}
                        className="p-3 border border-indigo-100 text-gray-700 whitespace-nowrap"
                      >
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}


      {/* --- Training Report and Structure --- */}
      {trainingReport && (
        <section className="p-8 bg-white rounded-xl shadow-2xl border-t-4 border-green-500 mt-10">
          

          <StructureViewer data={structureData} />
        </section>
      )}


                {/* --- Train/Test Data Visualization --- */}
{trainData && testData && (
  <section className="mt-10 bg-indigo-50 p-6 rounded-xl shadow-lg border border-indigo-200">
    <h2 className="text-3xl font-extrabold text-green-700 mb-4 flex items-center gap-3">
            <CheckCircle className="w-7 h-7" /> Model Training completed successfully!
          </h2>
    <h3 className="text-2xl font-bold text-indigo-700 mb-4">
      📊 Data Split Visualization (60%-40%)
    </h3>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      
      {/* Training Data Table */}
      <div className="bg-white rounded-xl shadow-md border border-indigo-100 p-4">
        <h4 className="text-lg font-semibold text-indigo-600 mb-3 text-center">
          🧠 Training Data
        </h4>
        <div className="overflow-x-auto max-h-96 overflow-y-auto border rounded-lg">
          <table className="min-w-full border-collapse text-sm">
            <thead className="bg-indigo-100 sticky top-0 z-10">
              <tr>
                {Object.keys(trainData[0] || {}).map((col) => (
                  <th
                    key={col}
                    className="p-2 border border-indigo-200 text-left font-semibold text-gray-700"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trainData.map((row, i) => (
                <tr key={i} className="hover:bg-indigo-50">
                  {Object.values(row).map((val, j) => (
                    <td
                      key={j}
                      className="p-2 border border-indigo-100 text-gray-800 whitespace-nowrap"
                    >
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Testing Data Table */}
      <div className="bg-white rounded-xl shadow-md border border-indigo-100 p-4">
        <h4 className="text-lg font-semibold text-indigo-600 mb-3 text-center">
          🧪 Testing Data
        </h4>
        <div className="overflow-x-auto max-h-96 overflow-y-auto border rounded-lg">
          <table className="min-w-full border-collapse text-sm">
            <thead className="bg-indigo-100 sticky top-0 z-10">
              <tr>
                {Object.keys(testData[0] || {}).map((col) => (
                  <th
                    key={col}
                    className="p-2 border border-indigo-200 text-left font-semibold text-gray-700"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {testData.map((row, i) => (
                <tr key={i} className="hover:bg-indigo-50">
                  {Object.values(row).map((val, j) => (
                    <td
                      key={j}
                      className="p-2 border border-indigo-100 text-gray-800 whitespace-nowrap"
                    >
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>
)}


{forestTrees && (
  <section className="mt-10 bg-purple-50 p-6 rounded-2xl shadow-lg border border-purple-200">
    <h2 className="text-2xl font-bold text-purple-700 mb-4 text-center">
      🌳 Random Forest - Tree Visualization
    </h2>
    <p className="text-sm text-gray-600 text-center mb-8">
      Explore the internal structure of each decision tree. You can zoom, pan, and inspect branches.
    </p>

    {/* Render each tree */}
    <div className="space-y-12">
      {Object.entries(forestTrees).map(([treeId, tree]) => (
        <div key={treeId} className="border-t pt-6">
          <h3 className="text-xl font-semibold text-purple-700 mb-4 text-center">
            Tree #{treeId}
          </h3>
          <div className="h-[600px] bg-white rounded-xl border border-purple-200 shadow-inner p-4">
            <ReactFlow
              nodes={tree.nodes}
              edges={tree.edges.map(e => ({
                ...e,
                animated: true,
                style: { stroke: "#7c3aed", strokeWidth: 2 },
              }))}
              fitView
              fitViewOptions={{ padding: 0.4 }}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable={false}
              zoomOnScroll
              panOnScroll
              minZoom={0.2}
              maxZoom={2}
            >
              <MiniMap />
              <Controls />
              <Background />
            </ReactFlow>
          </div>
        </div>
      ))}
    </div>
  </section>
)}







{
  trainData && (<div className="mt-10 flex flex-col items-center">
  <button
    onClick={handleTestData}
    disabled={isTesting}
    className={`${
      isTesting ? "bg-gray-400" : "bg-blue-600 hover:bg-blue-700"
    } text-white px-6 py-3 rounded-lg shadow-md transition-all`}
  >
    {isTesting ? "Testing..." : "Run Test Data"}
  </button>
</div>)
}

{testResults.length > 0 && (
  <section className="mt-10 bg-white p-8 rounded-2xl shadow-2xl border-t-4 border-blue-500">
    <h2 className="text-3xl font-extrabold text-indigo-700 mb-4 text-center">
      🧾 Test Data Evaluation Results
    </h2>

    {/* Results Table */}
    <div className="overflow-x-auto border border-indigo-100 rounded-xl shadow-inner max-h-[600px] overflow-y-auto">
      <table className="min-w-full border-collapse text-sm">
        <thead className="bg-indigo-100 sticky top-0 z-10">
          <tr>
            {Object.keys(testResults[0]).map((key) => (
              <th
                key={key}
                className="p-3 border border-indigo-200 text-left font-semibold text-gray-700"
              >
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {testResults.map((row, idx) => (
            <tr
  key={idx}
  onClick={() => handleRowToForm(row)}
  className={`cursor-pointer ${
    row.Predicted === row.Actual
      ? "bg-green-300 hover:bg-green-200"
      : "bg-red-300 hover:bg-red-200"
  } transition`}
>

              {Object.values(row).map((value, i) => (
                <td
                  key={i}
                  className="p-3 border border-indigo-50 text-gray-800 whitespace-nowrap"
                >
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>

    {/* Metrics Summary */}
{metrics && (
  <div className="mt-10 p-6 bg-indigo-50 border border-indigo-200 rounded-xl shadow-inner">
    <h3 className="text-2xl font-bold text-indigo-700 mb-4 text-center">
      📊 Classification Metrics Summary
    </h3>

    {/* Metrics Explanation Table */}
    <div className="overflow-x-auto rounded-xl border border-indigo-200 shadow-sm bg-white">
      <table className="min-w-full border-collapse text-sm">
        <thead className="bg-indigo-100">
          <tr>
            <th className="p-3 border border-indigo-200 text-left font-semibold text-gray-700">
              Metric
            </th>
            <th className="p-3 border border-indigo-200 text-left font-semibold text-gray-700">
              Formula
            </th>
            <th className="p-3 border border-indigo-200 text-left font-semibold text-gray-700">
              Calculation
            </th>
            <th className="p-3 border border-indigo-200 text-left font-semibold text-gray-700">
              Result
            </th>
          </tr>
        </thead>
        <tbody>
          {/* Accuracy */}
          <tr className="hover:bg-indigo-50">
            <td className="p-3 border border-indigo-100 font-semibold text-gray-800">
              Accuracy
            </td>
            <td className="p-3 border border-indigo-100 text-gray-700">
              (TP + TN) / (TP + TN + FP + FN)
            </td>
            <td className="p-3 border border-indigo-100 text-gray-600">
              ({metrics["True Positives"]} + {metrics["True Negatives"]}) / (
              {metrics["True Positives"]} + {metrics["True Negatives"]} +{" "}
              {metrics["False Positives"]} + {metrics["False Negatives"]})
            </td>
            <td className="p-3 border border-indigo-100 font-bold text-indigo-700">
              {metrics.Accuracy}%
            </td>
          </tr>

          {/* Precision */}
          <tr className="hover:bg-indigo-50">
            <td className="p-3 border border-indigo-100 font-semibold text-gray-800">
              Precision
            </td>
            <td className="p-3 border border-indigo-100 text-gray-700">
              TP / (TP + FP)
            </td>
            <td className="p-3 border border-indigo-100 text-gray-600">
              {metrics["True Positives"]} / ({metrics["True Positives"]} +{" "}
              {metrics["False Positives"]})
            </td>
            <td className="p-3 border border-indigo-100 font-bold text-indigo-700">
              {metrics.Precision}%
            </td>
          </tr>

          {/* Recall */}
          <tr className="hover:bg-indigo-50">
            <td className="p-3 border border-indigo-100 font-semibold text-gray-800">
              Recall
            </td>
            <td className="p-3 border border-indigo-100 text-gray-700">
              TP / (TP + FN)
            </td>
            <td className="p-3 border border-indigo-100 text-gray-600">
              {metrics["True Positives"]} / ({metrics["True Positives"]} +{" "}
              {metrics["False Negatives"]})
            </td>
            <td className="p-3 border border-indigo-100 font-bold text-indigo-700">
              {metrics.Recall}%
            </td>
          </tr>

          {/* F1 Score */}
          <tr className="hover:bg-indigo-50">
            <td className="p-3 border border-indigo-100 font-semibold text-gray-800">
              F1-Score
            </td>
            <td className="p-3 border border-indigo-100 text-gray-700">
              2 × (Precision × Recall) / (Precision + Recall)
            </td>
            <td className="p-3 border border-indigo-100 text-gray-600">
              2 × ({metrics.Precision}% × {metrics.Recall}%) / (
              {metrics.Precision}% + {metrics.Recall}%)
            </td>
            <td className="p-3 border border-indigo-100 font-bold text-indigo-700">
              {metrics["F1-Score"]}%
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    {/* Confusion Matrix Below */}
    <div className="mt-8 bg-white border border-indigo-200 rounded-xl shadow-sm p-4">
      <h4 className="text-lg font-semibold text-gray-700 mb-3 text-center">
        Confusion Matrix
      </h4>
      <div className="flex justify-center">
        <table className="border border-gray-300 text-sm">
          <thead className="bg-indigo-100">
            <tr>
              <th className="p-2 border border-gray-300"></th>
              <th className="p-2 border border-gray-300">Pred 0</th>
              <th className="p-2 border border-gray-300">Pred 1</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th className="p-2 border border-gray-300 text-gray-600">
                Actual 0
              </th>
              <td className="p-2 border border-gray-300 font-semibold text-green-700">
                {metrics["True Negatives"]}
              </td>
              <td className="p-2 border border-gray-300 font-semibold text-red-700">
                {metrics["False Positives"]}
              </td>
            </tr>
            <tr>
              <th className="p-2 border border-gray-300 text-gray-600">
                Actual 1
              </th>
              <td className="p-2 border border-gray-300 font-semibold text-red-700">
                {metrics["False Negatives"]}
              </td>
              <td className="p-2 border border-gray-300 font-semibold text-green-700">
                {metrics["True Positives"]}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
)}

  </section>
)}

{/* --- Prediction Section --- */}
        <section ref={predictRef}    className="lg:col-span-2 p-6 rounded-xl shadow-lg bg-white mt-4">
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
                      value={formData[f]??""}
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
              disabled={loadingState === "predicting"}
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




 {decisionPaths && Object.keys(decisionPaths).length > 0 && predictionResult && (
  <section className="mt-10 bg-green-50 p-8 rounded-2xl shadow-lg border border-green-200">
    <h2 className="text-3xl font-extrabold text-green-700 mb-6 text-center">
      🧠 Final Model Prediction: Class {predictionResult.prediction}
    </h2>

    

    <div className="space-y-12">
      {Object.entries(decisionPaths).map(([treeIndex, treeData]) => {
        const { nodes, edges } = treeData;
        
        return (
          <div key={treeIndex} className="border-t pt-6">
            <h3 className="text-xl font-semibold text-green-700 mb-4 text-center">
              Decision Tree #{treeIndex} - Prediction Path
            </h3>
            <div className="h-[700px] bg-white rounded-xl border border-green-200 shadow-inner p-4">
              <ReactFlow
  nodes={nodes.map(n => ({
    ...n,
    position: {
      x: n.position.x * 1.5,  // Increase horizontal spacing by 50%
      y: n.position.y * 1.2   // Slightly increase vertical spacing
    },
    style: {
      ...n.style,
      transition: "all 0.3s ease-in-out",
    },
  }))}
  edges={edges}
  fitView
  fitViewOptions={{ padding: 0.4 }}  // Increased padding for better view
  nodesDraggable={false}
  nodesConnectable={false}
  zoomOnScroll
  panOnScroll
  minZoom={0.15}  // Allow more zoom out to see the wider tree
  maxZoom={2}
>
  <MiniMap />
  <Controls />
  <Background />
</ReactFlow>
            </div>
          </div>
        );
      })}
    </div>
  </section>
)}


    </div>
  );
};

export default App;
