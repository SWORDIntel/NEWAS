import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Overview from './pages/Overview';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Overview />} />
      </Routes>
    </Router>
  );
}

export default App;
