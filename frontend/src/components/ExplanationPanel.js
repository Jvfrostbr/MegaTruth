import React from 'react';
import ReactMarkdown from 'react-markdown';
import './ExplanationPanel.css';

function ExplanationPanel({ explanation }) {
  return (
    <div className="explanation-panel">
      <h3>3. Análise Multimodal (Laudo Explicativo)</h3>
      
      <div className="explanation-content">
        <div className="model-badge">
          🤖 <strong>Modelo Utilizado:</strong> {explanation.model_used}
        </div>
        
        <div className="markdown-content">
          <ReactMarkdown>{explanation.explanation}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

export default ExplanationPanel;