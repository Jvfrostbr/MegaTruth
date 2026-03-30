import React from 'react';
import './AnalysisResults.css';

function AnalysisResults({ result }) {
  const { label, probability, conceitos, overlay_base64 } = result;

  const formatConceitos = () => {
    if (!conceitos || Object.keys(conceitos).length === 0) {
      return 'Nenhum defeito específico detectado';
    }
    
    return Object.entries(conceitos)
      .slice(0, 5)
      .map(([key, value]) => `• ${key}: ${(value * 100).toFixed(1)}%`)
      .join('\n');
  };

  return (
    <div className="analysis-results">
      <h3>2. Resultados Técnicos (Detector)</h3>
      
      <div className="results-grid">
        <div className="result-item">
          <div className="result-label">Classificação</div>
          <div className="result-value">{label}</div>
        </div>
        
        <div className="result-item">
          <div className="result-label">Grau de Certeza</div>
          <div className="result-value probability">
            {(probability * 100).toFixed(2)}%
          </div>
        </div>
        
        <div className="result-item full-width">
          <div className="result-label">Defeitos Específicos Detectados</div>
          <div className="result-value conceitos">
            <pre>{formatConceitos()}</pre>
          </div>
        </div>
        
        {overlay_base64 && (
          <div className="result-item full-width">
            <div className="result-label">Indicador dos defeitos Visuais</div>
            <img 
              src={`data:image/png;base64,${overlay_base64}`}
              alt="Defect Map"
              className="defect-map"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalysisResults;