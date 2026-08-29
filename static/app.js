// Lovecraftian Multimodal Studio Client JS

let allTales = [];
let selectedTale = null;
let eventSource = null;

// DOM Elements
const searchInput = document.getElementById("search-input");
const talesList = document.getElementById("tales-list");
const activeTaleTitle = document.getElementById("active-tale-title");
const activeTaleMeta = document.getElementById("active-tale-meta");
const btnReadStory = document.getElementById("btn-read-story");
const btnGenerate = document.getElementById("btn-generate");

const chkSummary = document.getElementById("chk-summary");
const chkImage = document.getElementById("chk-image");
const chkAudio = document.getElementById("chk-audio");
const customPromptInput = document.getElementById("custom-prompt");
const stepsSlider = document.getElementById("steps-slider");
const stepsValue = document.getElementById("steps-value");

const coverImage = document.getElementById("cover-image");
const coverPlaceholder = document.getElementById("cover-placeholder");
const audioPlayer = document.getElementById("audio-player");
const audioSection = document.getElementById("audio-section");
const synopsisContent = document.getElementById("synopsis-content");
const consoleLogs = document.getElementById("console-logs");

const readerModal = document.getElementById("reader-modal");
const readerModalTitle = document.getElementById("reader-modal-title");
const readerModalBody = document.getElementById("reader-modal-body");
const btnCloseModal = document.getElementById("btn-close-modal");

// Initialize Canvas Mist
function initMistCanvas() {
    const canvas = document.getElementById("mist-canvas");
    const ctx = canvas.getContext("2d");
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener("resize", () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const particles = [];
    for (let i = 0; i < 45; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            radius: Math.random() * 80 + 40,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            hue: Math.random() > 0.6 ? 160 : 45, // Emerald green or gold
            alpha: Math.random() * 0.08 + 0.02
        });
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < -p.radius) p.x = width + p.radius;
            if (p.x > width + p.radius) p.x = -p.radius;
            if (p.y < -p.radius) p.y = height + p.radius;
            if (p.y > height + p.radius) p.y = -p.radius;

            const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius);
            grad.addColorStop(0, `hsla(${p.hue}, 100%, 50%, ${p.alpha})`);
            grad.addColorStop(1, "rgba(0,0,0,0)");

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fill();
        });
        requestAnimationFrame(animate);
    }
    animate();
}

// Load Tales Catalog from API
async function loadTales() {
    try {
        const res = await fetch("/api/tales");
        allTales = await res.json();
        renderTalesList(allTales);

        if (allTales.length > 0) {
            selectTale(allTales[0]);
        }
    } catch (err) {
        console.error("Failed to load tales:", err);
        appendLog(`[ERROR] Failed to load catalog: ${err.message}`);
    }
}

function renderTalesList(tales) {
    talesList.innerHTML = "";
    tales.forEach(tale => {
        const item = document.createElement("div");
        item.className = `tale-item ${selectedTale && selectedTale.name === tale.name ? "active" : ""}`;
        item.dataset.taleName = tale.name;

        item.innerHTML = `
            <span class="tale-item-title">${tale.title}</span>
            <div class="tale-badges">
                ${tale.has_synopsis ? '<span class="badge synopsis-ready">📜 Lore</span>' : ''}
                ${tale.has_audio ? '<span class="badge audio-ready">🎙️ WAV</span>' : ''}
                ${tale.has_image ? '<span class="badge image-ready">🖼️ Art</span>' : ''}
            </div>
        `;

        item.addEventListener("click", () => selectTale(tale));
        talesList.appendChild(item);
    });
}

async function selectTale(tale) {
    selectedTale = tale;

    // Update active highlight in the Grimoire list
    document.querySelectorAll(".tale-item").forEach(el => {
        el.classList.toggle("active", el.dataset.taleName === tale.name);
    });

    activeTaleTitle.textContent = tale.title;
    activeTaleMeta.textContent = `H. P. Lovecraft • ~${tale.words.toLocaleString()} words`;

    // 1. Cover Art display
    if (tale.image_url) {
        coverImage.src = tale.image_url + "?t=" + Date.now();
        coverImage.style.display = "block";
        coverPlaceholder.style.display = "none";
    } else {
        coverImage.style.display = "none";
        coverPlaceholder.style.display = "flex";
    }

    // 2. Audio display
    if (tale.has_audio && tale.audio_url) {
        audioPlayer.src = tale.audio_url;
        audioSection.style.display = "block";
    } else {
        audioPlayer.src = "";
        audioSection.style.display = "none";
    }

    // 3. Synopsis / Lore display
    if (tale.has_synopsis && tale.synopsis) {
        synopsisContent.innerHTML = `<p>${tale.synopsis.replace(/\n/g, "<br>")}</p>`;
    } else {
        synopsisContent.innerHTML = `<p><em>Selected <strong>${tale.title}</strong>. Click "Awaken the Old Ones" to generate synopsis, cover art, and Vincent Price audio.</em></p>`;
    }

    // 4. Custom prompt suggestion
    if (tale.image_prompt && !customPromptInput.value.trim()) {
        customPromptInput.placeholder = `Saved visual lore: "${tale.image_prompt.slice(0, 70)}..."`;
    } else {
        customPromptInput.placeholder = "e.g. Ancient monolith submerged under dark green algae and moonlight...";
    }

    appendLog(`[Catalog] Selected '${tale.title}' (${tale.filename})`);

    // 5. Asynchronously query API to guarantee we have the absolute latest data
    try {
        const res = await fetch(`/api/tale/${tale.name}`);
        if (res.ok) {
            const data = await res.json();
            if (data.has_synopsis && data.synopsis) {
                tale.has_synopsis = true;
                tale.synopsis = data.synopsis;
                tale.image_prompt = data.image_prompt;
                synopsisContent.innerHTML = `<p>${data.synopsis.replace(/\n/g, "<br>")}</p>`;
                renderTalesList(allTales);
            }
        }
    } catch (e) {
        console.warn("Could not refresh tale details:", e);
    }
}

// Search Filter
searchInput.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    const filtered = allTales.filter(t => t.title.toLowerCase().includes(q));
    renderTalesList(filtered);
});

// Slider updates
stepsSlider.addEventListener("input", (e) => {
    stepsValue.textContent = e.target.value;
});

// Story Reader Modal
btnReadStory.addEventListener("click", async () => {
    if (!selectedTale) return;
    readerModalTitle.textContent = selectedTale.title;
    readerModalBody.innerHTML = "<p><em>Loading ancient text...</em></p>";
    readerModal.classList.add("open");

    try {
        const res = await fetch(`/api/tale/${selectedTale.name}`);
        const data = await res.json();
        // Convert simple markdown to HTML paragraphs
        const formatted = data.content
            .split("\n\n")
            .filter(p => p.trim())
            .map(p => {
                if (p.startsWith("#")) return `<h3>${p.replace(/#+\s*/, "")}</h3>`;
                if (p.startsWith(">")) return `<blockquote>${p.replace(/^>\s*/, "")}</blockquote>`;
                return `<p>${p}</p>`;
            })
            .join("");
        readerModalBody.innerHTML = formatted;
    } catch (err) {
        readerModalBody.innerHTML = `<p style="color:red">Failed to load text: ${err.message}</p>`;
    }
});

btnCloseModal.addEventListener("click", () => readerModal.classList.remove("open"));
readerModal.addEventListener("click", (e) => {
    if (e.target === readerModal) readerModal.classList.remove("open");
});

// Logging helper
function appendLog(text) {
    const time = new Date().toLocaleTimeString();
    consoleLogs.textContent += `[${time}] ${text}\n`;
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

// Generate Pipeline Action (SSE Stream)
btnGenerate.addEventListener("click", () => {
    if (!selectedTale) return;

    const doSummary = chkSummary.checked;
    const doImage = chkImage.checked;
    const doAudio = chkAudio.checked;
    const customPrompt = customPromptInput.value.trim();
    const steps = stepsSlider.value;

    if (!doSummary && !doImage && !doAudio) {
        alert("Please select at least one component to generate.");
        return;
    }

    btnGenerate.disabled = true;
    btnGenerate.textContent = "⚡ Communing with the Void...";
    appendLog(`=== STARTING PIPELINE FOR '${selectedTale.title}' ===`);

    const query = new URLSearchParams({
        tale_name: selectedTale.name,
        do_summary: doSummary,
        do_image: doImage,
        do_audio: doAudio,
        custom_prompt: customPrompt,
        steps: steps
    });

    if (eventSource) eventSource.close();
    eventSource = new EventSource(`/api/generate/stream?${query.toString()}`);

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.log) {
                appendLog(data.log);
            }

            if (data.synopsis) {
                synopsisContent.innerHTML = `<p>${data.synopsis.replace(/\n/g, "<br>")}</p>`;
                selectedTale.has_synopsis = true;
                selectedTale.synopsis = data.synopsis;
                renderTalesList(allTales);
            }

            if (data.image_url) {
                coverImage.src = data.image_url + "?t=" + Date.now();
                coverImage.style.display = "block";
                coverPlaceholder.style.display = "none";
                selectedTale.has_image = true;
                selectedTale.image_url = data.image_url;
                renderTalesList(allTales);
            }

            if (data.audio_url) {
                audioPlayer.src = data.audio_url + "?t=" + Date.now();
                audioSection.style.display = "block";
                audioPlayer.play().catch(() => {});
                selectedTale.has_audio = true;
                selectedTale.audio_url = data.audio_url;
                renderTalesList(allTales);
            }

            if (data.done) {
                eventSource.close();
                btnGenerate.disabled = false;
                btnGenerate.textContent = "⚡ Awaken the Old Ones";
                appendLog("=== COMPLETED ALL TASKS ===");
            }
        } catch (e) {
            console.error("Error parsing event data:", e);
        }
    };

    eventSource.onerror = (err) => {
        if (eventSource && eventSource.readyState === EventSource.CLOSED) {
            appendLog("[INFO] Stream disconnected.");
            btnGenerate.disabled = false;
            btnGenerate.textContent = "⚡ Awaken the Old Ones";
        } else {
            console.warn("SSE connection state change:", eventSource ? eventSource.readyState : "null");
        }
    };
});

// Start Canvas and Load Initial Data
document.addEventListener("DOMContentLoaded", () => {
    initMistCanvas();
    loadTales();
});
