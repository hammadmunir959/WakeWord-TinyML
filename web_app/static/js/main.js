/**
 * WakeWord AI - Frontend Logic
 * Handles audio capture, resampling, WebSocket streaming, and visualization.
 */

let ws = null;
let audioContext = null;
let scriptProcessor = null;
let source = null;

const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusBadge = document.getElementById('ws-status');
const detectionResult = document.getElementById('detection-result');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceVal = document.getElementById('confidence-val');
const latencyVal = document.getElementById('latency-val');
const historyLog = document.getElementById('history-log');
const canvas = document.getElementById('waveform');
const ctx = canvas.getContext('2d');

// Audio Visualizer data
let audioData = new Float32Array(256);

function initWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        statusBadge.innerText = 'Connected';
        statusBadge.classList.add('connected');
        console.log('WebSocket Connected');
    };

    ws.onclose = () => {
        statusBadge.innerText = 'Disconnected';
        statusBadge.classList.remove('connected');
        console.log('WebSocket Closed');
        stopAudio();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
    };
}

let lastDetection = "";
let lastDetectionTime = 0;

function updateUI(data) {
    // Latency and Energy
    if (data.latency) latencyVal.innerText = `${Math.round(data.latency)}ms`;

    // Update visualizer energy (handled in draw loop now)

    if (data.label && data.label !== "listening...") {
        detectionResult.innerText = data.label;
        detectionResult.style.transform = 'scale(1.1)';
        setTimeout(() => detectionResult.style.transform = 'scale(1)', 200);

        const confidence = Math.round(data.confidence * 100);
        confidenceFill.style.width = `${confidence}%`;
        confidenceVal.innerText = `${confidence}%`;

        // Highlight active tag
        document.querySelectorAll('.class-tag').forEach(t => t.classList.remove('active'));
        const activeTag = document.getElementById(`tag-${data.label}`);
        if (activeTag) activeTag.classList.add('active');

        // Add to history if new or enough time passed
        const now = Date.now();
        if (data.label !== lastDetection || now - lastDetectionTime > 2000) {
            addToHistory(data.label, confidence);
            lastDetection = data.label;
            lastDetectionTime = now;
        }
    } else {
        // Reset if no detection
        detectionResult.innerText = "Listening...";
        confidenceFill.style.width = '0%';
        confidenceVal.innerText = '0%';
        document.querySelectorAll('.class-tag').forEach(t => t.classList.remove('active'));
    }
}

function addToHistory(label, confidence) {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span><b>${label}</b></span>
        <span style="color: #a0a0a0">${confidence}% • ${time}</span>
    `;

    // Remove placeholder
    const placeholder = historyLog.querySelector('.log-placeholder');
    if (placeholder) placeholder.remove();

    historyLog.prepend(entry);

    // Keep only last 10
    if (historyLog.children.length > 10) {
        historyLog.lastElementChild.remove();
    }
}

async function startAudio() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        source = audioContext.createMediaStreamSource(stream);

        // ScriptProcessor for raw data (easier than Worklets for demo)
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);

            // Update visualizer array
            audioData = new Float32Array(inputData.length);
            audioData.set(inputData);

            // Stream to server
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(inputData.buffer);
            }
        };

        startBtn.disabled = true;
        stopBtn.disabled = false;
        initWS();

        draw();

    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Could not access microphone. Please ensure you are on localhost or HTTPS.');
    }
}

function stopAudio() {
    if (scriptProcessor) scriptProcessor.disconnect();
    if (source) source.disconnect();
    if (audioContext) audioContext.close();
    if (ws) ws.close();

    startBtn.disabled = false;
    stopBtn.disabled = true;
}

function draw() {
    if (startBtn.disabled === false) return; // Stop drawing when not listening

    requestAnimationFrame(draw);

    // Resize canvas
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Draw centered static line
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw waveform
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00f2ff';

    const sliceWidth = width / audioData.length;
    let x = 0;

    for (let i = 0; i < audioData.length; i++) {
        const v = audioData[i] * 5; // Sensitivity
        const y = (v * height / 2) + height / 2;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();
}

startBtn.onclick = startAudio;
stopBtn.onclick = stopAudio;
