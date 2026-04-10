import React, { useState } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import ColorSelector from './components/ColorSelector';
import AnalysisResults from './components/AnalysisResults';
import ExplanationPanel from './components/ExplanationPanel';
import LoadingSpinner from './components/LoadingSpinner';
import { analyzeImage, generateExplanation } from './services/api';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [overlayColor, setOverlayColor] = useState('🔴 Vermelho (Padrão)');
  const [modelChoice, setModelChoice] = useState('🔧 Nemotron (Reduzir imagem e usar Nemotron)');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExplaining, setIsExplaining] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (uploadedImage) => {
    setImage(uploadedImage);
    setAnalysisResult(null);
    setExplanation(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!image) {
      setError('Por favor, selecione uma imagem primeiro.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await analyzeImage(image, overlayColor);
      setAnalysisResult(result);
    } catch (err) {
      setError(err.message || 'Erro ao analisar imagem');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExplain = async () => {
    if (!analysisResult) {
      setError('Por favor, execute a análise visual primeiro.');
      return;
    }

    setIsExplaining(true);
    setError(null);

    try {
      const result = await generateExplanation(
        analysisResult.image_path,
        analysisResult.defect_maps,
        analysisResult.label,
        analysisResult.probability,
        analysisResult.conceitos,
        overlayColor,
        modelChoice.includes('Nemotron')
      );
      setExplanation(result);
    } catch (err) {
      setError(err.message || 'Erro ao gerar explicação');
    } finally {
      setIsExplaining(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      
      <div className="content-container">
        <div className="input-section">
          <ImageUploader onImageUpload={handleImageUpload} />
          
          {image && (
            <>
              <ColorSelector 
                value={overlayColor}
                onChange={setOverlayColor}
              />
              
              <button 
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? <LoadingSpinner size="small" /> : 'Analisar Imagem'}
              </button>
            </>
          )}
          
          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </div>
        
        {analysisResult && (
          <>
            <AnalysisResults result={analysisResult} />
            
            <div className="explanation-section">
              <div className="model-selector">
                <label>Modo de Geração do Laudo:</label>
                <select 
                  value={modelChoice}
                  onChange={(e) => setModelChoice(e.target.value)}
                >
                  <option value="🔧 Nemotron (Reduzir imagem e usar Nemotron)">
                    🔧 Nemotron (Reduzir imagem e usar Nemotron)
                  </option>
                  <option value="🛡️ LLaVA (Manter tamanho e usar LLaVA)">
                    🛡️ LLaVA (Manter tamanho e usar LLaVA)
                  </option>
                </select>
              </div>
              
              <button 
                className="explain-button"
                onClick={handleExplain}
                disabled={isExplaining}
              >
                {isExplaining ? <LoadingSpinner size="small" /> : 'Gerar Laudo Explicativo'}
              </button>
              
              {explanation && (
                <ExplanationPanel explanation={explanation} />
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;