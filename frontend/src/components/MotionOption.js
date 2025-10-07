import React from 'react';

const MotionOption = ({ motion, isSelected, onSelect }) => {
  return (
    <div
      className={`motion-option ${isSelected ? 'selected' : ''}`}
      onClick={() => onSelect(motion.id)}
    >
      <video src={motion.src} muted autoPlay loop playsInline />
      <span>{motion.name}</span>
    </div>
  );
};

export default MotionOption;