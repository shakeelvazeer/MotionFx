import React from 'react';
import ImageUpload from './ImageUpload';
import MotionOption from './MotionOption';

const InputForm = ({
  MOTIONS,
  previewUrl,
  selectedMotionId,
  isFormComplete,
  onImageChange,
  onMotionSelect,
  onAnimate,
}) => {
  return (
    <>
      <div id="input-area">
        <ImageUpload previewUrl={previewUrl} onImageChange={onImageChange} />
        <div className="input-section">
          <h2>2. Choose a Motion</h2>
          <div className="motion-gallery">
            {MOTIONS.map((motion) => (
              <MotionOption
                key={motion.id}
                motion={motion}
                isSelected={selectedMotionId === motion.id}
                onSelect={onMotionSelect}
              />
            ))}
          </div>
        </div>
      </div>
      <button className="animate-button" disabled={!isFormComplete} onClick={onAnimate}>
        Animate Now
      </button>
    </>
  );
};

export default InputForm;