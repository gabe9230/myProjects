const NBALLS = 5000;
const CENTER_COLOR = [0.5, 0.0, 0.0];
const FAR_COLOR = [1.0, 0.5, 0.5];
const MAX_SPEED = 5.0;
const ATTRACTION_STRENGTH = 0.0025;
const DAMPING = 1;
const CELL_SIZE = 12;
const INITIAL_RADIUS = 1;
const MERGE_THRESHOLD_SPEED = 3.25;

let gl, program, posBuffer, colorBuffer, radiusBuffer;
let nCols, nRows, buckets;
let ballPositions = new Float32Array(NBALLS * 2);
let ballColors = new Float32Array(NBALLS * 3);
let ballRadii = new Float32Array(NBALLS);
const active = new Uint8Array(NBALLS);
const physicsState = {
    x: new Float32Array(NBALLS),
    y: new Float32Array(NBALLS),
    vx: new Float32Array(NBALLS),
    vy: new Float32Array(NBALLS),
    r: new Float32Array(NBALLS)
};

// FPS variables
let fps = 0;
let lastTime = Date.now();
let frameCount = 0;
let fpsElement;

function initGL() {
    // Create FPS display
    fpsElement = document.createElement('div');
    fpsElement.style.position = 'fixed';
    fpsElement.style.top = '10px';
    fpsElement.style.right = '10px';
    fpsElement.style.color = 'white';
    fpsElement.style.fontFamily = 'Arial, sans-serif';
    fpsElement.style.zIndex = '1000';
    document.body.appendChild(fpsElement);

    const canvas = document.getElementById('c');
    gl = canvas.getContext('webgl', { antialias: false }) || 
         canvas.getContext('experimental-webgl', { antialias: false });
    if (!gl) {
        alert('WebGL unavailable!');
        return;
    }
    gl.clearColor(0, 0, 0, 1);
    
    const vertShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertShader, `
        attribute vec2 position;
        attribute vec3 color;
        attribute float radius;
        varying vec3 vColor;
        void main() {
            vColor = color;
            gl_PointSize = radius * 2.0;
            gl_Position = vec4(position, 0.0, 1.0);
        }
    `);
    gl.compileShader(vertShader);

    const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragShader, `
        precision mediump float;
        varying vec3 vColor;
        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            if(length(coord) > 0.5) discard;
            gl_FragColor = vec4(vColor, 1.0);
        }
    `);
    gl.compileShader(fragShader);

    program = gl.createProgram();
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    posBuffer = gl.createBuffer();
    colorBuffer = gl.createBuffer();
    radiusBuffer = gl.createBuffer();
    
    const posLoc = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(posLoc);
    const colorLoc = gl.getAttribLocation(program, 'color');
    gl.enableVertexAttribArray(colorLoc);
    const radiusLoc = gl.getAttribLocation(program, 'radius');
    gl.enableVertexAttribArray(radiusLoc);
}

function rebuildGrid() {
    nCols = Math.ceil(window.innerWidth / CELL_SIZE);
    nRows = Math.ceil(window.innerHeight / CELL_SIZE);
    buckets = Array(nCols * nRows).fill().map(() => []);
}

function bIdx(ix, iy) { return ix + iy * nCols; }

function resolveCollision(i, j) {
    if (!active[i] || !active[j] || i === j) return;
    
    const dx = physicsState.x[j] - physicsState.x[i];
    const dy = physicsState.y[j] - physicsState.y[i];
    const distSq = dx*dx + dy*dy;
    const minDist = physicsState.r[i] + physicsState.r[j];
    
    if (distSq >= minDist*minDist) return;

    const dist = Math.sqrt(distSq);
    const dvx = physicsState.vx[i] - physicsState.vx[j];
    const dvy = physicsState.vy[i] - physicsState.vy[j];
    const relSpeed = Math.hypot(dvx, dvy);
    
    if (relSpeed >= MERGE_THRESHOLD_SPEED) {
        const m1 = physicsState.r[i]**2;
        const m2 = physicsState.r[j]**2;
        const absorber = m1 > m2 ? i : j;
        const absorbed = absorber === i ? j : i;

        active[absorbed] = 0;
        physicsState.r[absorber] = Math.sqrt(m1 + m2)/1.15;
        physicsState.vx[absorber] = (physicsState.vx[i]*m1 + physicsState.vx[j]*m2)/(m1+m2);
        physicsState.vy[absorber] = (physicsState.vy[i]*m1 + physicsState.vy[j]*m2)/(m1+m2);
        return;
    }

    const nx = dx/dist;
    const ny = dy/dist;
    const vRel = dvx*nx + dvy*ny;
    
    if (vRel >= 0) return;

    const m1 = physicsState.r[i]**2;
    const m2 = physicsState.r[j]**2;
    const impulse = (-2 * vRel) / (1/m1 + 1/m2);
    
    physicsState.vx[i] += (impulse * nx) / m1;
    physicsState.vy[i] += (impulse * ny) / m1;
    physicsState.vx[j] -= (impulse * nx) / m2;
    physicsState.vy[j] -= (impulse * ny) / m2;

    const overlap = minDist - dist;
    if (overlap > 0) {
        const totalMass = m1 + m2;
        physicsState.x[i] -= nx * overlap * m2 / totalMass;
        physicsState.y[i] -= ny * overlap * m2 / totalMass;
        physicsState.x[j] += nx * overlap * m1 / totalMass;
        physicsState.y[j] += ny * overlap * m1 / totalMass;
    }
}

function resize() {
    const canvas = document.getElementById('c');
    const w = canvas.width = window.innerWidth;
    const h = canvas.height = window.innerHeight;
    gl.viewport(0, 0, w, h);
    rebuildGrid();

    const centerX = w/2;
    const centerY = h/2;

    active.fill(1);
    for(let i = 0; i < NBALLS; i++) {
        physicsState.r[i] = INITIAL_RADIUS;
        const angle = Math.random() * Math.PI * 2;
        const radius = 100 + Math.random() * Math.min(w, h) * 0.3;
        
        physicsState.x[i] = centerX + Math.cos(angle) * radius;
        physicsState.y[i] = centerY + Math.sin(angle) * radius;

        const baseSpeed = Math.sqrt(ATTRACTION_STRENGTH * radius);
        const direction = Math.random() > 0.5 ? 1 : -1;
        
        physicsState.vx[i] = Math.cos(angle + Math.PI/2 * direction) * baseSpeed;
        physicsState.vy[i] = Math.sin(angle + Math.PI/2 * direction) * baseSpeed;
    }
    updateColors();
}

function updateColors() {
    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        const speed = Math.hypot(physicsState.vx[i], physicsState.vy[i]);
        const t = Math.min(speed / MAX_SPEED, 1.0);
        
        ballColors[i*3] = CENTER_COLOR[0] * (1 - t) + FAR_COLOR[0] * t;
        ballColors[i*3+1] = CENTER_COLOR[1] * (1 - t) + FAR_COLOR[1] * t;
        ballColors[i*3+2] = CENTER_COLOR[2] * (1 - t) + FAR_COLOR[2] * t;
    }
}

function physics() {
    const centerX = window.innerWidth/2;
    const centerY = window.innerHeight/2;

    rebuildGrid();

    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        const cx = (physicsState.x[i]/CELL_SIZE) | 0;
        const cy = (physicsState.y[i]/CELL_SIZE) | 0;
        if (cx >= 0 && cy >= 0 && cx < nCols && cy < nRows) {
            buckets[bIdx(cx, cy)].push(i);
        }
    }

    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        
        const dx = centerX - physicsState.x[i];
        const dy = centerY - physicsState.y[i];
        const force = ATTRACTION_STRENGTH * 100 / ((dx*dx + dy*dy)/10 + 100);
        
        physicsState.vx[i] += dx * force;
        physicsState.vy[i] += dy * force;
        physicsState.x[i] += physicsState.vx[i];
        physicsState.y[i] += physicsState.vy[i];

        const margin = window.innerWidth * 0.1;
        physicsState.x[i] = Math.clamp(physicsState.x[i], -margin, window.innerWidth + margin);
        physicsState.y[i] = Math.clamp(physicsState.y[i], -margin, window.innerHeight + margin);
    }

    for(let cx = 0; cx < nCols; cx++) {
        for(let cy = 0; cy < nRows; cy++) {
            const cell = buckets[bIdx(cx, cy)];
            const len = cell.length;
            
            for(let a = 0; a < len; a++) {
                const i = cell[a];
                for(let b = a+1; b < len; b++) {
                    resolveCollision(i, cell[b]);
                }
            }
            
            const neighbors = [
                [cx+1, cy], [cx, cy+1],
                [cx-1, cy], [cx, cy-1]
            ];
            
            for(const [nx, ny] of neighbors) {
                if (nx >= 0 && ny >= 0 && nx < nCols && ny < nRows) {
                    const nCell = buckets[bIdx(nx, ny)];
                    for(const i of cell) {
                        for(const j of nCell) {
                            resolveCollision(i, j);
                        }
                    }
                }
            }
        }
    }
}

function render() {
    const activePositions = new Float32Array(NBALLS * 2);
    const activeColors = new Float32Array(NBALLS * 3);
    const activeRadii = new Float32Array(NBALLS);
    let count = 0;

    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        
        activePositions[count*2] = (physicsState.x[i]/window.innerWidth)*2 - 1;
        activePositions[count*2+1] = 1 - (physicsState.y[i]/window.innerHeight)*2;
        
        activeColors[count*3] = ballColors[i*3];
        activeColors[count*3+1] = ballColors[i*3+1];
        activeColors[count*3+2] = ballColors[i*3+2];
        
        activeRadii[count] = physicsState.r[i];
        count++;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, activePositions.subarray(0, count*2), gl.STREAM_DRAW);
    gl.vertexAttribPointer(gl.getAttribLocation(program, 'position'), 2, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, activeColors.subarray(0, count*3), gl.STREAM_DRAW);
    gl.vertexAttribPointer(gl.getAttribLocation(program, 'color'), 3, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, radiusBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, activeRadii.subarray(0, count), gl.STREAM_DRAW);
    gl.vertexAttribPointer(gl.getAttribLocation(program, 'radius'), 1, gl.FLOAT, false, 0, 0);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, count);
}

function animate() {
    frameCount++;
    const now = Date.now();
    const delta = now - lastTime;

    if (delta >= 1000) {
        fps = Math.round((frameCount * 1000) / delta);
        fpsElement.textContent = `FPS: ${fps}`;
        frameCount = 0;
        lastTime = now;
    }

    physics();
    updateColors();
    render();
    requestAnimationFrame(animate);
}

window.addEventListener('load', () => {
    initGL();
    resize();
    animate();
});
window.addEventListener('resize', resize);

Math.clamp = (value, min, max) => Math.min(Math.max(value, min), max);