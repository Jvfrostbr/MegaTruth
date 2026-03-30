import React from 'react';
import './ColorSelector.css';

function ColorSelector({ value, onChange }) {
  const colors = [
    { value: '🔴 Vermelho (Padrão)', label: 'Vermelho (Padrão)', color: '#ff0000' },
    { value: '🟢 Verde (Para fundos avermelhados)', label: 'Verde (Para fundos avermelhados)', color: '#00ff00' },
    { value: '🔵 Azul (Para fundos quentes)', label: 'Azul (Para fundos quentes)', color: '#0000ff' }
  ];

  return (
    <div className="color-selector">
      <label>Cor do Overlay (defect_map):</label>
      <div className="color-options">
        {colors.map(option => (
          <div 
            key={option.value}
            className={`color-option ${value === option.value ? 'selected' : ''}`}
            onClick={() => onChange(option.value)}
          >
            <div 
              className="color-swatch" 
              style={{ backgroundColor: option.color }}
            />
            <span>{option.label}</span>
          </div>
        ))}
      </div>
      <p className="color-hint">Escolha uma cor que contraste com a imagem original.</p>
    </div>
  );
}

export default ColorSelector;