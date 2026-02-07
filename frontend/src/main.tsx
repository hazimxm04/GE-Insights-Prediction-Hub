import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // You don't need the extension here
import './index.css';

// The '!' tells TS: "I know the root div exists in index.html, don't worry."
const rootElement = document.getElementById('root')!;
const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);