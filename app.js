// Global variables
let mediaRecorder;
let audioChunks = [];
let recordingStream;
const Api = "http://localhost:8800"

document.addEventListener('DOMContentLoaded', function() {
    // Form submission for file upload
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }

    // Recording functionality
    const recordButton = document.getElementById('recordButton');
    if (recordButton) {
        recordButton.addEventListener('click', toggleRecording);
    }

    // New analysis button
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', resetAnalysis);
    }
});

// Handle file upload submission
function handleFileUpload(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an audio file first');
        return;
    }
    
    // Create FormData and append file
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    const uploadBtn = document.getElementById('uploadBtn');
    const originalBtnText = uploadBtn.innerHTML;
    uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
    uploadBtn.disabled = true;
    
    // Use AJAX to send the file
    $.ajax({
        url: `${Api}/upload`,
        type: 'POST',
        data: formData,
        processData: false,  
        contentType: false,  
        headers: {
            'Accept': 'application/json'
        },
        success: function(data) {
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error: ' + (data.error || 'Unknown error occurred'));
            }
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
            alert('An error occurred while uploading the file');
        },
        complete: function() {
            // Reset button state
            uploadBtn.innerHTML = originalBtnText;
            uploadBtn.disabled = false;
        }
    });
}

// Toggle recording state
function toggleRecording() {
    const recordButton = document.getElementById('recordButton');
    const recordingStatus = document.getElementById('recordingStatus');
    
    // If currently recording, stop recording
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
        recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordButton.classList.replace('btn-danger', 'btn-outline-primary');
        recordingStatus.classList.add('d-none');
        return;
    }
    
    // Start recording
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            recordingStream = stream;
            
            // Show recording state
            recordButton.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
            recordButton.classList.replace('btn-outline-primary', 'btn-danger');
            recordingStatus.classList.remove('d-none');
            
            // Create media recorder
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            // Handle data available event
            mediaRecorder.ondataavailable = function(e) {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };
            
            // Handle recording stop event
            mediaRecorder.onstop = function() {
                // Process and send audio data
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                submitAudioForAnalysis(audioBlob);
                
                // Clean up
                stopMediaTracks(recordingStream);
            };
            
            // Start recording
            mediaRecorder.start();
        })
        .catch(function(error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access your microphone. Please ensure it is properly connected and you have granted permission.');
        });
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }
}

// Clean up media tracks
function stopMediaTracks(stream) {
    if (!stream) return;
    stream.getTracks().forEach(function(track) {
        track.stop();
    });
}

// Submit recorded audio for analysis
function submitAudioForAnalysis(audioBlob) {
    // Show loading state
    const recordButton = document.getElementById('recordButton');
    recordButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
    recordButton.disabled = true;
    
    // Create form data with audio blob
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    // Send to server using AJAX
    $.ajax({
        url: `${Api}/analyze-live`,
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(data) {
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error analyzing audio: ' + (data.error || 'Unknown error'));
            }
        },
        error: function(xhr, status, error) {
            console.error('Error submitting audio:', error);
            alert('An error occurred while analyzing your recording');
        },
        complete: function() {
            // Reset button state
            recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            recordButton.classList.replace('btn-danger', 'btn-outline-primary');
            recordButton.disabled = false;
        }
    });
}

// Display analysis results
function displayResults(data) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('d-none');
    
    // Update prediction text and icon
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const resultIcon = document.getElementById('resultIcon');
    
    predictionResult.textContent = data.prediction;
    confidenceResult.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    
    // Set appropriate icon based on prediction
    setResultIcon(resultIcon, data.prediction);
    
    // Set visualization image
    const audioVisualization = document.getElementById('audioVisualization');
    audioVisualization.src = `data:image/png;base64,${data.visualization}`;
    
    // Create probability chart
    createProbabilityChart(data.all_probabilities);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Set the appropriate icon based on prediction
function setResultIcon(iconElement, prediction) {
    // Define icon mapping
    const iconMap = {
        'hungry': 'fa-utensils',
        'tired': 'fa-bed',
        'belly_pain': 'fa-stomach',
        'burping': 'fa-wind',
        'discomfort': 'fa-frown',
        // Add more mappings as needed
    };
    
    // Set default if prediction not in map
    const iconClass = iconMap[prediction.toLowerCase()] || 'fa-question-circle';
    
    // Clear existing classes and add new ones
    iconElement.className = '';
    iconElement.classList.add('fas', iconClass, 'text-primary', 'fa-3x', 'mb-3');
}

// Create probability chart
function createProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    
    // Convert probabilities object to arrays for Chart.js
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities).map(function(p) { 
        return p * 100; // Convert to percentages
    });
    
    // Destroy existing chart if exists
    if (window.probabilityChart instanceof Chart) {
        window.probabilityChart.destroy();
    }
    
    // Create new chart
    window.probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: data,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Classification Probabilities'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Cry Type'
                    }
                }
            }
        }
    });
}

// Reset analysis and form
function resetAnalysis() {
    // Hide results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.add('d-none');
    
    // Reset file input
    const fileInput = document.getElementById('audioFile');
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Reset recording button if needed
    const recordButton = document.getElementById('recordButton');
    if (recordButton) {
        recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordButton.classList.replace('btn-danger', 'btn-outline-primary');
        recordButton.disabled = false;
    }
    
    // Hide recording status
    const recordingStatus = document.getElementById('recordingStatus');
    if (recordingStatus) {
        recordingStatus.classList.add('d-none');
    }
    
    // Stop any ongoing recording
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    }
}