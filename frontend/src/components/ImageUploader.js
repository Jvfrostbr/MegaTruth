import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './ImageUploader.css';

function ImageUploader({ onImageUpload }) {
  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageUpload(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  }, [onImageUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp']
    },
    maxFiles: 1
  });

  return (
    <div className="image-uploader">
      <div 
        {...getRootProps()} 
        className={`dropzone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Solte a imagem aqui...</p>
        ) : (
          <div className="upload-prompt">
            <p>Arraste e solte uma imagem aqui, ou clique para selecionar</p>
            <p className="upload-hint">Suporta: JPG, PNG, GIF, BMP</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageUploader;