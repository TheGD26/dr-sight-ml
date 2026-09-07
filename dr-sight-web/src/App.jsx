import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Home from "./pages/Home.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Screening from "./pages/Screening.jsx";
import Results from "./pages/Results.jsx";
import HistoryPage from "./pages/History.jsx";
import Learn from "./pages/Learn.jsx";
import NotFound from "./pages/NotFound.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/screening" element={<Screening />} />
        <Route path="/results/:id" element={<Results />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/learn" element={<Learn />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
