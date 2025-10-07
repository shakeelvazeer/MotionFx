import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import InputForm from './components/InputForm';
import StatusDisplay from './components/StatusDisplay';

const MOTIONS = [
    { id: "Taichi 1", src: "motions/taichi5.mp4", name: "Taichi 1" },
    { id: "Taichi 2", src: "motions/taichi6.mp4", name: "Taichi 2" },
    { id: "Taichi 3", src: "motions/taichi7.mp4", name: "Taichi 3" },
    { id: "Taichi 4", src: "motions/taichi8.mp4", name: "Taichi 4" },
];

function App() {
    const [selectedImage, setSelectedImage] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [selectedMotionId, setSelectedMotionId] = useState(null);
    const [jobId, setJobId] = useState(null);
    const [status, setStatus] = useState('idle');
    const [statusText, setStatusText] = useState('');
    const [error, setError] = useState(null);
    const [resultUrl, setResultUrl] = useState(null);

    const handleImageChange = (event) => {
        const file = event.target.files[0];
        if (!file) return;
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
        setSelectedImage(file);
        setPreviewUrl(URL.createObjectURL(file));
    };

    const startAnimationProcess = async () => {
        if (!selectedImage || !selectedMotionId) return;

        setStatus('processing');
        setStatusText('Uploading and creating job...');
        setError(null);

        const formData = new FormData();
        formData.append('image', selectedImage);
        formData.append('motion_id', selectedMotionId);

        try {
            const response = await fetch('/api/jobs', { method: 'POST', body: formData });
            if (!response.ok) {
                const errData = await response.json().catch(() => null);
                throw new Error(errData?.message || 'Server failed to start the job.');
            }
            const data = await response.json();
            setJobId(data.job_id);
        } catch (err) {
            setStatus('error');
            setError(err.message || 'Failed to connect to the server.');
        }
    };

    const resetState = useCallback(() => {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setSelectedImage(null);
        setPreviewUrl(null);
        setSelectedMotionId(null);
        setJobId(null);
        setStatus('idle');
        setError(null);
        setResultUrl(null);
        if (document.getElementById('image-input')) {
            document.getElementById('image-input').value = null;
        }
    }, [previewUrl]);

    useEffect(() => {
        if (status !== 'processing' || !jobId) return;

        const intervalId = setInterval(async () => {
            try {
                const response = await fetch(`/api/jobs/${jobId}/status`);
                if (!response.ok) throw new Error('Could not get job status.');
                const data = await response.json();

                if (data.status === 'processing') {
                    setStatusText(data.step || 'Processing...');
                } else if (data.status === 'completed') {
                    setStatus('completed');
                    setResultUrl(data.result_url);
                    clearInterval(intervalId);
                } else if (data.status === 'error') {
                    setStatus('error');
                    setError(data.message || 'An error occurred during processing.');
                    clearInterval(intervalId);
                }
            } catch (err) {
                setStatus('error');
                setError('Connection to server lost.');
                clearInterval(intervalId);
            }
        }, 3000);

        return () => clearInterval(intervalId);

    }, [jobId, status]);

    return (
        <div className="container">
            <header>
                <h1>MotionFx</h1>
                <p>Bring Your Photos to Life</p>
            </header>

            <main>
                {status === 'idle' ? (
                    <InputForm
                        MOTIONS={MOTIONS}
                        previewUrl={previewUrl}
                        selectedMotionId={selectedMotionId}
                        isFormComplete={!!selectedImage && !!selectedMotionId}
                        onImageChange={handleImageChange}
                        onMotionSelect={setSelectedMotionId}
                        onAnimate={startAnimationProcess}
                    />
                ) : (
                    <StatusDisplay
                        status={status}
                        statusText={statusText}
                        error={error}
                        resultUrl={resultUrl}
                        onReset={resetState}
                    />
                )}
            </main>
            
            <footer>
                <p>A project by <a href="https://www.linkedin.com/in/shakeel-vazeer/" target="_blank" rel="noopener noreferrer">Shakeel Vazeer</a></p>
            </footer>
        </div>
    );
}

export default App;