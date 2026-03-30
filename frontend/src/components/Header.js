import React from 'react';
import './Header.css';

function Header() {
  return (
    <div className="header">
      <div className="header-content">
        <img 
          src="https://raw.githubusercontent.com/Jvfrostbr/MegaTruth/main/images/logo/logo_mega_truth.png" 
          alt="MegaTruth Logo" 
          className="logo"
        />
        <div className="header-text">
          <h1>MegaTruth</h1>
          <h3>Sistema Forense de Detecção de IA</h3>
          <p>
            <strong>1. Análise Visual:</strong> O modelo <strong>CLIP (Fine-Tuned)</strong> detecta anomalias e gera um <em>defect_map</em>.<br />
            <strong>2. Análise Semântica:</strong> O sistema verifica <strong>Conceitos Específicos</strong> (anatomia, física).<br />
            <strong>3. Análise Multimodal:</strong> Uma <strong>IA Multimodal</strong> gera o laudo final.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Header;