// Curated examples for quick testing
const EXAMPLES = [
    "NASA's James Webb Space Telescope has captured deep-field infrared images revealing the earliest known galaxies formed shortly after the Big Bang.",
    "Secret government bunker in Antarctica contains living extraterrestrial beings communicating through telepathy and microwave ovens.",
    "Federal Reserve holds benchmark interest rate steady while monitoring labor market indicators and core inflation trends.",
    "Drinking warm saltwater and baking soda cures all known cancers and diabetes in forty-eight hours, doctors baffled.",
    "Open-source security researchers discover and patch a critical vulnerability in the Linux kernel network stack.",
    "Ancient pyramid discovered on Mars reveals human civilization originated in the constellation Orion and is guarded by androids."
];

let checkHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('newsInput');
    
    // Live counts
    input.addEventListener('input', updateCounts);

    // Keyboard shortcut: Ctrl + Enter
    input.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            checkNews();
        }
    });

    updateCounts();
});

function updateCounts() {
    const input = document.getElementById('newsInput');
    const text = input.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const chars = input.value.length;

    document.getElementById('wordCount').textContent = `${words} word${words === 1 ? '' : 's'}`;
    document.getElementById('charCount').textContent = `${chars} character${chars === 1 ? '' : 's'}`;
}

function loadExample(index) {
    if (EXAMPLES[index]) {
        const input = document.getElementById('newsInput');
        input.value = EXAMPLES[index];
        updateCounts();
        input.focus();
        checkNews();
    }
}

async function pasteFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        if (text) {
            const input = document.getElementById('newsInput');
            input.value = text;
            updateCounts();
            input.focus();
        }
    } catch (err) {
        alert("Clipboard read permission was denied or not supported by your browser.");
    }
}

function clearInput() {
    const input = document.getElementById('newsInput');
    input.value = '';
    updateCounts();
    document.getElementById('resultSection').classList.add('hidden');
    input.focus();
}

async function checkNews() {
    const input = document.getElementById('newsInput');
    const text = input.value.trim();
    const resultSection = document.getElementById('resultSection');
    const loader = document.getElementById('loader');
    const checkBtn = document.getElementById('checkBtn');

    if (!text) {
        alert("Please enter or paste a news headline or article first.");
        input.focus();
        return;
    }

    // Toggle loader UI
    loader.classList.remove('hidden');
    resultSection.classList.add('hidden');
    checkBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            renderResult(data, text);
            addToHistory(text, data.prediction, data.confidence);
        } else {
            alert(`Analysis Error: ${data.error || 'Server returned an invalid response.'}`);
        }
    } catch (error) {
        console.error('Prediction request error:', error);
        alert("Could not connect to the detection server. Please ensure app.py is running.");
    } finally {
        loader.classList.add('hidden');
        checkBtn.disabled = false;
    }
}

function renderResult(data, originalText) {
    const resultSection = document.getElementById('resultSection');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceScore = document.getElementById('confidenceScore');
    const distSummary = document.getElementById('distSummary');
    const realBar = document.getElementById('realBar');
    const fakeBar = document.getElementById('fakeBar');
    const keywordsCard = document.getElementById('keywordsCard');
    const keywordsList = document.getElementById('keywordsList');

    const isReal = data.prediction === 'REAL';
    predictionResult.textContent = isReal ? "AUTHENTIC (REAL)" : "FABRICATED (FAKE)";
    predictionResult.className = `verdict-title ${isReal ? 'result-real' : 'result-fake'}`;

    confidenceScore.textContent = `${data.confidence}%`;

    // Probability spectrum
    const realProb = data.probabilities ? (data.probabilities.REAL || 0) : (isReal ? parseFloat(data.confidence) : 100 - parseFloat(data.confidence));
    const fakeProb = data.probabilities ? (data.probabilities.FAKE || 0) : (isReal ? 100 - parseFloat(data.confidence) : parseFloat(data.confidence));

    realBar.style.width = `${realProb}%`;
    fakeBar.style.width = `${fakeProb}%`;
    distSummary.textContent = `REAL: ${realProb.toFixed(1)}% | FAKE: ${fakeProb.toFixed(1)}%`;

    // Keywords / Features
    if (data.top_keywords && data.top_keywords.length > 0) {
        keywordsList.innerHTML = '';
        data.top_keywords.forEach(kw => {
            const tag = document.createElement('span');
            tag.className = `keyword-tag ${kw.favors === 'REAL' ? 'favors-real' : 'favors-fake'}`;
            tag.textContent = `${kw.term} (${kw.favors})`;
            keywordsList.appendChild(tag);
        });
        keywordsCard.classList.remove('hidden');
    } else {
        keywordsCard.classList.add('hidden');
    }

    resultSection.classList.remove('hidden');
}

function addToHistory(text, prediction, confidence) {
    checkHistory.unshift({ text, prediction, confidence });
    if (checkHistory.length > 6) {
        checkHistory.pop();
    }
    renderHistory();
}

function renderHistory() {
    const historySection = document.getElementById('historySection');
    const historyList = document.getElementById('historyList');

    if (checkHistory.length === 0) {
        historySection.classList.add('hidden');
        return;
    }

    historyList.innerHTML = '';
    checkHistory.forEach((item, idx) => {
        const row = document.createElement('div');
        row.className = 'history-item';
        row.onclick = () => {
            document.getElementById('newsInput').value = item.text;
            updateCounts();
            checkNews();
        };

        const textSpan = document.createElement('span');
        textSpan.className = 'history-text';
        textSpan.textContent = item.text;

        const pill = document.createElement('span');
        pill.className = `history-pill ${item.prediction === 'REAL' ? 'real' : 'fake'}`;
        pill.textContent = `${item.prediction} (${item.confidence}%)`;

        row.appendChild(textSpan);
        row.appendChild(pill);
        historyList.appendChild(row);
    });

    historySection.classList.remove('hidden');
}

function clearHistory() {
    checkHistory = [];
    renderHistory();
}
