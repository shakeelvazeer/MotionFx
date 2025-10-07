import React from 'react';

const BACKEND_URL = "http://127.0.0.1:8000";

const StatusDisplay = ({ status, statusText, error, resultUrl, onReset }) => {
  
  const fullResultUrl = resultUrl ? `${BACKEND_URL}${resultUrl}` : "#";

  return (
    <div id="status-area">
      {status === 'processing' && (
        <>
          <div className="spinner"></div>
          <p id="status-text">{statusText}</p>
        </>
      )}
      {status === 'error' && (
        <>
          <h3 id="error-message">{error}</h3>
          <button className="reset-button" onClick={onReset} style={{marginTop: '1rem'}}>Try Again</button>
        </>
      )}
      {status === 'completed' && (
        <div id="result-view">
          <video id="result-video" src={resultUrl} controls autoPlay loop></video>
          <div className="result-actions">
            
            <a href={fullResultUrl} download="promotion-fx.mp4" className="button download-button">
              Download Video
            </a>
            
            <button className="button reset-button" onClick={onReset}>
              Create Another
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default StatusDisplay;