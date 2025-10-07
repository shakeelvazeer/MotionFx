import React from 'react';

const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
  </svg>
);

const ImageUpload = ({ previewUrl, onImageChange }) => {
  return (
    <div className="input-section">
      <h2>1. Upload an Image</h2>
      <label htmlFor="image-input" className="upload-box">
        {previewUrl ? (
          <img src={previewUrl} alt="Preview" className="preview-image" />
        ) : (
          <div className="upload-prompt">
            <UploadIcon />
            <span>Drag & Drop or Click to Upload</span>
          </div>
        )}
      </label>
      <input type="file" id="image-input" accept="image/png, image/jpeg" hidden onChange={onImageChange} />
      <div className="upload-note">Front facing photos work best</div>
    </div>
  );
};

export default ImageUpload;