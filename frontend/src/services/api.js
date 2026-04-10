import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const analyzeImage = async (imageData, overlayColor) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/analyze`, {
      image: imageData,
      overlay_color: overlayColor
    });
    
    if (response.data.status === 'error') {
      throw new Error(response.data.error);
    }
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw new Error(error.response?.data?.error || 'Failed to analyze image');
  }
};

export const generateExplanation = async (
  imagePath,
  defectMaps,
  clipLabel,
  clipProbability,
  conceitos,
  overlayColor,
  preferNemotron
) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/explain`, {
      image_path: imagePath,
      defect_maps: defectMaps,
      clip_label: clipLabel,
      clip_probability: clipProbability,
      conceitos: conceitos,
      overlay_color: overlayColor,
      prefer_nemotron: preferNemotron,
      resize_images: preferNemotron
    });
    
    if (response.data.status === 'error') {
      throw new Error(response.data.error);
    }
    
    return response.data;
  } catch (error) {
    console.error('Error generating explanation:', error);
    throw new Error(error.response?.data?.error || 'Failed to generate explanation');
  }
};

export const healthCheck = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};