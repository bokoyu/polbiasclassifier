// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import EvaluatePage from './pages/EvaluatePage';
import TrainPage from './pages/TrainPage';
import NavBar from './components/NavBar';

function App() {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/evaluate" element={<EvaluatePage />} />
        <Route path="/train" element={<TrainPage />} />
      </Routes>
    </Router>
  );
}

export default App;
