import React from 'react';
import { ThreeDots } from 'react-loader-spinner';
import './LoadingSpinner.css';

function LoadingSpinner({ size = 'medium' }) {
  const sizes = {
    small: 30,
    medium: 50,
    large: 80
  };

  return (
    <div className="spinner-container">
      <ThreeDots
        height={sizes[size]}
        width={sizes[size]}
        radius="9"
        color="#667eea"
        ariaLabel="three-dots-loading"
        visible={true}
      />
    </div>
  );
}

export default LoadingSpinner;