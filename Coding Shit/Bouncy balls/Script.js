// main.js - Complete Version with Radius Fix and Logging
const NBALLS = 5000;
const CENTER_COLOR = [0.5, 0.0, 0.0];
const FAR_COLOR = [1.0, 0.6, 0.6];
const MAX_COLOR_DIST = 500;
const ATTRACTION_STRENGTH = 0.1;
const MUTUAL_ATTRACTION = 5;
const DAMPING = 0.98;
const CELL_SIZE = 80;
const INITIAL_RADIUS = 1;

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

function initGL() {
  const canvas = document.getElementById('c');
  gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
  if (!gl) {
      alert('WebGL unavailable!');
      return;
  }
  gl.clearColor(0, 0, 0, 1);
  
  // Vertex Shader with point size calculation
  const vertShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertShader, `
      attribute vec2 position;
      attribute vec3 color;
      attribute float radius;
      varying vec3 vColor;
      varying float vRadius;
      void main() {
          vColor = color;
          vRadius = radius;
          gl_PointSize = radius * 2.0;
          gl_Position = vec4(position, 0.0, 1.0);
      }
  `);
  gl.compileShader(vertShader);

  // Fragment Shader with circle rendering
  const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragShader, `
      precision mediump float;
      varying vec3 vColor;
      varying float vRadius;
      void main() {
          vec2 coord = gl_PointCoord - vec2(0.5);
          float dist = length(coord);
          float smoothing = 1.0 / (vRadius * 2.0);
          float alpha = 1.0 - smoothstep(0.5 - smoothing, 0.5, dist);
          gl_FragColor = vec4(vColor, alpha);
          
          // Discard fully transparent fragments
          if (alpha <= 0.0) discard;
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

function overlaps(i, j) {
    return active[i] && active[j] && 
           Math.hypot(physicsState.x[i]-physicsState.x[j], 
                      physicsState.y[i]-physicsState.y[j]) < 
           physicsState.r[i] + physicsState.r[j];
}

function resolveCollision(i, j) {
    if (!active[i] || !active[j]) return;
    
    const m1 = physicsState.r[i] ** 2;
    const m2 = physicsState.r[j] ** 2;
    
    let absorber, absorbed;
    if (m1 > m2 || (m1 === m2 && Math.random() > 0.5)) {
        absorber = i;
        absorbed = j;
    } else {
        absorber = j;
        absorbed = i;
    }

    active[absorbed] = 0;
    physicsState.r[absorber] = Math.sqrt(m1 + m2);
    
    physicsState.vx[absorber] = (physicsState.vx[absorber]*m1 + physicsState.vx[absorbed]*m2)/(m1+m2);
    physicsState.vy[absorber] = (physicsState.vy[absorber]*m1 + physicsState.vy[absorbed]*m2)/(m1+m2);
}

function resize() {
    const canvas = document.getElementById('c');
    const w = canvas.width = window.innerWidth;
    const h = canvas.height = window.innerHeight;
    gl.viewport(0, 0, w, h);
    rebuildGrid();

    const centerX = w / 2;
    const centerY = h / 2;

    active.fill(1);
    for(let i = 0; i < NBALLS; i++) {
        physicsState.r[i] = INITIAL_RADIUS;
        const angle = Math.random() * Math.PI * 2;
        const radius = 100 + Math.random() * Math.min(w, h) * 0.3;
        
        physicsState.x[i] = centerX + Math.cos(angle) * radius;
        physicsState.y[i] = centerY + Math.sin(angle) * radius;

        const baseSpeed = Math.sqrt(ATTRACTION_STRENGTH * radius);
        const direction = Math.random() > 0.5 ? 1 : -1;
        const speedMultiplier = 0.9 + Math.random() * 0.2;
        
        physicsState.vx[i] = Math.cos(angle + Math.PI/2 * direction) * baseSpeed * speedMultiplier;
        physicsState.vy[i] = Math.sin(angle + Math.PI/2 * direction) * baseSpeed * speedMultiplier;
    }
    updateColors();
}

function updateColors() {
    const centerX = window.innerWidth / 2;
    const centerY = window.innerHeight / 2;
    
    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        const dx = physicsState.x[i] - centerX;
        const dy = physicsState.y[i] - centerY;
        const dist = Math.hypot(dx, dy);
        const t = Math.min(dist / MAX_COLOR_DIST, 1.0);
        
        ballColors[i*3] = FAR_COLOR[0] * (1 - t) + CENTER_COLOR[0] * t;
        ballColors[i*3+1] = FAR_COLOR[1] * (1 - t) + CENTER_COLOR[1] * t;
        ballColors[i*3+2] = FAR_COLOR[2] * (1 - t) + CENTER_COLOR[2] * t;
    }
}

function physics() {
    const centerX = window.innerWidth / 2;
    const centerY = window.innerHeight / 2;

    rebuildGrid();

    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        const cx = Math.floor(physicsState.x[i] / CELL_SIZE);
        const cy = Math.floor(physicsState.y[i] / CELL_SIZE);
        if (cx >= 0 && cy >= 0 && cx < nCols && cy < nRows) {
            buckets[bIdx(cx, cy)].push(i);
        }
    }

    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        
        const dx = centerX - physicsState.x[i];
        const dy = centerY - physicsState.y[i];
        const distSq = dx*dx + dy*dy;
        const force = ATTRACTION_STRENGTH * 100 / ((distSq/10) + 100);
        
        physicsState.vx[i] += dx * force;
        physicsState.vy[i] += dy * force;

        physicsState.vx[i] *= DAMPING;
        physicsState.vy[i] *= DAMPING;
        physicsState.x[i] += physicsState.vx[i];
        physicsState.y[i] += physicsState.vy[i];

        const margin = window.innerWidth * 0.1;
        physicsState.x[i] = Math.max(-margin, Math.min(window.innerWidth + margin, physicsState.x[i]));
        physicsState.y[i] = Math.max(-margin, Math.min(window.innerHeight + margin, physicsState.y[i]));
    }

    for(let cx = 0; cx < nCols; cx++) {
        for(let cy = 0; cy < nRows; cy++) {
            const cell = buckets[bIdx(cx, cy)];
            for(let a = 0; a < cell.length; a++) {
                const i = cell[a];
                for(let b = a+1; b < cell.length; b++) {
                    const j = cell[b];
                    if (overlaps(i, j)) resolveCollision(i, j);
                }
                for(let dx = -1; dx <= 1; dx++) {
                    for(let dy = -1; dy <= 1; dy++) {
                        if (dx === 0 && dy === 0) continue;
                        const nx = cx + dx, ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= nCols || ny >= nRows) continue;
                        const neighbor = buckets[bIdx(nx, ny)];
                        if (neighbor) {
                            for(const j of neighbor) {
                                if (overlaps(i, j)) resolveCollision(i, j);
                            }
                        }
                    }
                }
            }
        }
    }
}

function logLargestBalls() {
    const balls = [];
    for(let i = 0; i < NBALLS; i++) {
        if (active[i]) {
            balls.push({
                index: i,
                radius: physicsState.r[i],
                x: physicsState.x[i],
                y: physicsState.y[i]
            });
        }
    }
    
    balls.sort((a, b) => b.radius - a.radius);
    const top10 = balls.slice(0, 10);
    
    console.log('Top 10 Largest Balls:');
    top10.forEach((ball, idx) => {
        console.log(`#${idx + 1}: Radius ${ball.radius.toFixed(2)} at (${ball.x.toFixed(1)}, ${ball.y.toFixed(1)})`);
    });
}

function render() {
    const activePositions = new Float32Array(NBALLS * 2);
    const activeColors = new Float32Array(NBALLS * 3);
    const activeRadii = new Float32Array(NBALLS);
    let activeCount = 0;

    for(let i = 0; i < NBALLS; i++) {
        if (!active[i]) continue;
        
        activePositions[activeCount*2] = (physicsState.x[i] / window.innerWidth) * 2 - 1;
        activePositions[activeCount*2+1] = 1 - (physicsState.y[i] / window.innerHeight) * 2;
        
        activeColors[activeCount*3] = ballColors[i*3];
        activeColors[activeCount*3+1] = ballColors[i*3+1];
        activeColors[activeCount*3+2] = ballColors[i*3+2];
        
        activeRadii[activeCount] = physicsState.r[i];
        
        activeCount++;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, activePositions.subarray(0, activeCount * 2), gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(gl.getAttribLocation(program, 'position'), 2, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, activeColors.subarray(0, activeCount * 3), gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(gl.getAttribLocation(program, 'color'), 3, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, radiusBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, activeRadii.subarray(0, activeCount), gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(gl.getAttribLocation(program, 'radius'), 1, gl.FLOAT, false, 0, 0);

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, activeCount);
}

function animate() {
    physics();
    updateColors();
    render();
    
    // Log largest balls every 5 seconds
    if (Math.floor(performance.now() / 5000) % 2 === 0) {
        logLargestBalls();
    }
    
    requestAnimationFrame(animate);
}

window.addEventListener('load', () => {
    initGL();
    resize();
    animate();
});
window.addEventListener('resize', resize);