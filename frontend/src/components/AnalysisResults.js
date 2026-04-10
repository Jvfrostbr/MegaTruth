import React, { useState } from 'react';
import './AnalysisResults.css';

function AnalysisResults({ result }) {
  const { label, probability, conceitos, defect_maps } = result;
  const [currentDefectMapIndex, setCurrentDefectMapIndex] = useState(0);

  const formatConceitos = () => {
    if (!conceitos || Object.keys(conceitos).length === 0) {
      return 'Nenhum defeito específico detectado';
    }
    
    return Object.entries(conceitos)
      .slice(0, 5)
      .map(([key, value]) => `• ${key}: ${(value * 100).toFixed(1)}%`)
      .join('\n');
  };

  // Cria lista de conceitos ordenados por probabilidade
  const sortedConceitos = defect_maps && defect_maps.length > 0
    ? defect_maps
        .map((d, idx) => ({ ...d, index: idx }))
        .sort((a, b) => b.probabilidade - a.probabilidade)
    : [];

  const currentDefectMap = defect_maps && defect_maps.length > 0 ? defect_maps[currentDefectMapIndex] : null;

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

        {sortedConceitos.length > 0 && (
          <div className="result-item full-width">
            <div className="result-label">Selecione um Defeito</div>
            <div className="defect-buttons-container">
              {sortedConceitos.map((defect, idx) => (
                <button
                  key={idx}
                  className={`defect-button ${currentDefectMapIndex === defect.index ? 'active' : ''}`}
                  onClick={() => setCurrentDefectMapIndex(defect.index)}
                >
                  <span className="button-concept">{defect.conceito}</span>
                  <span className="button-probability">
                    {(defect.probabilidade * 100).toFixed(1)}%
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
        
        {currentDefectMap && (
          <div className="result-item full-width">
            <div className="defect-map-header-full">
              <div className="defect-info">
                <span className="defect-concept">{currentDefectMap.conceito}</span>
                <span className="defect-probability">
                  {(currentDefectMap.probabilidade * 100).toFixed(1)}%
                </span>
              </div>
              {defect_maps.length > 1 && (
                <div className="defect-counter">
                  {currentDefectMapIndex + 1} de {defect_maps.length}
                </div>
              )}
            </div>
            <img 
              src={`data:image/png;base64,${currentDefectMap.image_base64}`}
              alt={`Defect Map - ${currentDefectMap.conceito}`}
              className="defect-map"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalysisResults;