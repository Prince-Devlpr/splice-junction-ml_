import { useState } from "react";
import DataView from "./pages/DataView";
import Analytics from "./pages/Analytics";
import Preprocess from "./pages/Preprocess";
import PreprocessedData from "./pages/PreprocessedData";
import KBANNView from "./pages/KBANNView";
import Metrics from "./pages/Metrics";
import Comparison from "./pages/Comparison"; // Imported here
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("overview");

  // Logical Flow Grouping
  const renderContent = () => {
    switch (activeTab) {
      case "overview":
        return (
          <div className="view-column">
            <Analytics />
            <Metrics />
          </div>
        );
      case "dataset":
        return (
          <div className="view-column">
            <DataView />
            <PreprocessedData />
          </div>
        );
      case "experiment":
        return (
          <div className="view-grid">
            <Preprocess />
            <KBANNView />
          </div>
        );
      case "compare": // Added new case
        return (
          <div className="view-column">
            <Comparison />
          </div>
        );
      default:
        return <Analytics />;
    }
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <h2 className="logo">Splice Junction Dashboard</h2>
        <nav className="nav-menu">
          <button 
            className={activeTab === "overview" ? "nav-btn active" : "nav-btn"} 
            onClick={() => setActiveTab("overview")}
          >
            📊 Overview & Metrics
          </button>
          <button 
            className={activeTab === "dataset" ? "nav-btn active" : "nav-btn"} 
            onClick={() => setActiveTab("dataset")}
          >
            🗄️ Datasets
          </button>
          <button 
            className={activeTab === "experiment" ? "nav-btn active" : "nav-btn"} 
            onClick={() => setActiveTab("experiment")}
          >
            🔬 Test & Analyze
          </button>
          {/* New Tab Button */}
          <button 
            className={activeTab === "compare" ? "nav-btn active" : "nav-btn"} 
            onClick={() => setActiveTab("compare")}
          >
            ⚖️ Compare Models
          </button>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="top-header">
          <h1>{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Dashboard</h1>
        </header>
        <div className="content-area">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
