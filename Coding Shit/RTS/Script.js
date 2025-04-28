let canvas = document.getElementById("c");
let ctx = canvas.getContext("2d");
let width = 1024;
let height = 768;

// Configuration
const VERTEX_SPACING = 48;
const CHUNK_SIZE = 16;
const TILE_HEIGHT = 24;
const ISO_SCALE = 0.01;
const MAX_HEIGHT = 96;
const LOAD_DISTANCE = 3;
const FRAME_RATE = 60;

// System state
let viewport = { x: 0, y: 0, scale: 1.5, targetScale: 1.5 };
const chunks = new Map();
const keys = {};
let lastFrame = performance.now();
let colorCache = new Map();

class PerlinNoise {
    constructor() {
        this.grad3 = [
            [1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
            [1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
            [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1]
        ];
        this.perm = new Uint8Array(512);
        this.seed();
    }

    seed() {
        const p = new Uint8Array(256);
        for (let i = 0; i < 256; i++) p[i] = i;
        for (let i = 255; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [p[i], p[j]] = [p[j], p[i]];
        }
        for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255];
    }

    noise(x, y) {
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;
        x -= Math.floor(x);
        y -= Math.floor(y);
        const u = this.fade(x);
        const v = this.fade(y);
        
        const a = this.perm[X] + Y;
        const b = this.perm[X+1] + Y;
        
        return this.lerp(
            this.lerp(this.grad(this.perm[a], x, y),
            this.grad(this.perm[b], x-1, y), u),
            this.lerp(this.grad(this.perm[a+1], x, y-1),
            this.grad(this.perm[b+1], x-1, y-1), u), v
        );
    }

    fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    lerp(a, b, t) { return a + t * (b - a); }
    grad(hash, x, y) {
        const g = this.grad3[hash % 12];
        return g[0] * x + g[1] * y;
    }
}

const perlin = new PerlinNoise();

// Input handling
window.addEventListener('keydown', (e) => keys[e.key.toLowerCase()] = true);
window.addEventListener('keyup', (e) => keys[e.key.toLowerCase()] = false);

function updateViewport() {
    const speed = 800 / viewport.scale;
    if (keys['a'] || keys['arrowleft']) viewport.x += speed;
    if (keys['d'] || keys['arrowright']) viewport.x -= speed;
    if (keys['w'] || keys['arrowup']) viewport.y += speed;
    if (keys['s'] || keys['arrowdown']) viewport.y -= speed;
    
    if (keys['q']) viewport.targetScale = Math.min(3, viewport.scale * 1.05);
    if (keys['e']) viewport.targetScale = Math.max(0.5, viewport.scale * 0.95);
    viewport.scale += (viewport.targetScale - viewport.scale) * 0.1;
}

function worldToScreen(wx, wy, wz) {
    return {
        x: (wx - wy) * 0.866 * viewport.scale + width/2 + viewport.x,
        y: (wx + wy) * 0.5 * viewport.scale - wz * TILE_HEIGHT + height/2 + viewport.y
    };
}

function getChunk(cx, cy) {
    const key = `${cx},${cy}`;
    if (!chunks.has(key)) {
        chunks.set(key, { buffer: null, heights: null });
        generateChunk(cx, cy);
        return null;
    }
    return chunks.get(key);
}

function generateChunk(cx, cy) {
    const heights = new Float32Array((CHUNK_SIZE + 1) ** 2);
    const scale = 0.02;
    
    for (let y = 0; y <= CHUNK_SIZE; y++) {
        for (let x = 0; x <= CHUNK_SIZE; x++) {
            const wx = (cx * CHUNK_SIZE + x) * VERTEX_SPACING;
            const wy = (cy * CHUNK_SIZE + y) * VERTEX_SPACING;
            let h = perlin.noise(wx * scale, wy * scale);
            h += 0.5 * perlin.noise(wx * scale * 2, wy * scale * 2);
            heights[y * (CHUNK_SIZE + 1) + x] = h;
        }
    }
    
    const buffer = document.createElement('canvas');
    buffer.width = buffer.height = CHUNK_SIZE * VERTEX_SPACING;
    const btx = buffer.getContext('2d');
    
    for (let y = 0; y < CHUNK_SIZE; y++) {
        for (let x = 0; x < CHUNK_SIZE; x++) {
            const h = (heights[y * (CHUNK_SIZE + 1) + x] + 1) / 2;
            const color = getTerrainColor(h);
            btx.fillStyle = color;
            btx.fillRect(
                x * VERTEX_SPACING,
                y * VERTEX_SPACING,
                VERTEX_SPACING,
                VERTEX_SPACING
            );
        }
    }
    
    chunks.get(`${cx},${cy}`).buffer = buffer;
    chunks.get(`${cx},${cy}`).heights = heights;
}

function getTerrainColor(height) {
    const key = Math.floor(height * 100);
    if (colorCache.has(key)) return colorCache.get(key);
    
    const colors = [
        [3, 0, 168],    // Deep water
        [0, 98, 255],    // Shallow water
        [230, 211, 108], // Sand
        [59, 194, 54],   // Grass
        [105, 99, 79],   // Rock
        [200, 200, 200]  // Snow
    ];
    const thresholds = [0.0, 0.2, 0.25, 0.4, 0.7, 1.0];
    
    for (let i = 1; i < thresholds.length; i++) {
        if (height < thresholds[i]) {
            const t = (height - thresholds[i-1]) / (thresholds[i] - thresholds[i-1]);
            const color = colors[i-1].map((c, idx) => 
                Math.min(255, c + t * (colors[i][idx] - c))
            );
            const colorStr = `rgb(${color.join(',')})`;
            colorCache.set(key, colorStr);
            return colorStr;
        }
    }
    return 'rgb(200,200,200)';
}

function drawVisibleChunks() {
    const now = performance.now();
    if (now - lastFrame < 1000 / FRAME_RATE) return;
    lastFrame = now;

    ctx.clearRect(0, 0, width, height);
    
    const viewWidth = width / (0.866 * viewport.scale);
    const viewHeight = height / (0.5 * viewport.scale);
    const minCX = Math.floor((viewport.x - viewWidth) / (CHUNK_SIZE * VERTEX_SPACING)) - LOAD_DISTANCE;
    const maxCX = Math.ceil((viewport.x + viewWidth) / (CHUNK_SIZE * VERTEX_SPACING)) + LOAD_DISTANCE;
    const minCY = Math.floor((viewport.y - viewHeight) / (CHUNK_SIZE * VERTEX_SPACING)) - LOAD_DISTANCE;
    const maxCY = Math.ceil((viewport.y + viewHeight) / (CHUNK_SIZE * VERTEX_SPACING)) + LOAD_DISTANCE;

    for (let cy = minCY; cy <= maxCY; cy++) {
        for (let cx = minCX; cx <= maxCX; cx++) {
            const chunk = getChunk(cx, cy);
            if (!chunk?.buffer) continue;

            const screenPos = worldToScreen(
                cx * CHUNK_SIZE * VERTEX_SPACING,
                cy * CHUNK_SIZE * VERTEX_SPACING,
                0
            );
            
            ctx.drawImage(
                chunk.buffer,
                screenPos.x,
                screenPos.y,
                CHUNK_SIZE * VERTEX_SPACING * viewport.scale,
                CHUNK_SIZE * VERTEX_SPACING * viewport.scale
            );
        }
    }
}

function gameLoop() {
    updateViewport();
    drawVisibleChunks();
    requestAnimationFrame(gameLoop);
}

function setupCanvas() {
    canvas.width = width;
    canvas.height = height;
    canvas.style.imageRendering = "crisp-edges";
    ctx.imageSmoothingEnabled = false;
}

setupCanvas();
gameLoop();