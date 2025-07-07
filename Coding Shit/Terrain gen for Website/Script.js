// Terrain generator & renderer – simple Math.random() noise
// Draw scaling controlled via `drawSize`

/****************  configuration  ****************/
const canvas = document.getElementById('c')
const ctx = canvas.getContext('2d')

const width = 1028
const height = 1028
const drawSize = 1
const blurDist = 5
const blurSteps = 3

/****************  helper functions  ****************/
const lerp = (a, b, t) => a + (b - a) * t

function Blur(x,y,hm) {
    let blurVal = 0
    for (let dx = -blurDist; dx < blurDist; dx++) {
        for (let dy = -blurDist; dy < blurDist; dy++) {
            if (hm[x+dx]) {
                if (hm[x+dx][y+dy]) {
                    blurVal += hm[x+dx][y+dy]
                }
            }
        }
    }
    blurVal /= ((blurDist*2)+1)**2
    return blurVal * 1.25 //to adjust for darkening
}

/****************  Noise functions  ****************/
function makeNoiseGrid(cols, rows) {
    return Array.from({ length: rows }, () =>
        Float32Array.from({ length: cols }, () => Math.random())
    )
}

function sampleNoise(grid, u, v) {
    const rows = grid.length,
        cols = grid[0].length
    const x = u * (cols - 1),
        y = v * (rows - 1)
    const xi = Math.floor(x),
        yi = Math.floor(y)
    const xf = x - xi,
        yf = y - yi
    const x1 = Math.min(xi + 1, cols - 1),
        y1 = Math.min(yi + 1, rows - 1)

    const v00 = grid[yi][xi],
        v10 = grid[yi][x1]
    const v01 = grid[y1][xi],
        v11 = grid[y1][x1]
    return lerp(lerp(v00, v10, xf), lerp(v01, v11, xf), yf)
}

function layeredNoise(u, v, octaves = 4) {
    let amp = 1,
        freq = 1,
        sum = 0,
        norm = 0
    for (let o = 0; o < octaves; o++) {
        const gridSize = 16 * freq + 1
        layeredNoise._cache ??= {}
        if (!layeredNoise._cache[gridSize]) {
            layeredNoise._cache[gridSize] = makeNoiseGrid(gridSize, gridSize)
        }
        const g = layeredNoise._cache[gridSize]
        sum += sampleNoise(g, (u * freq) % 1, (v * freq) % 1) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0
    }
    return sum / norm
}

/****************  cellular-automata landmask  ****************/
function generateLandmask(cw, ch, fillProb = 0.45, steps = 5) {
    let grid = Array.from({ length: ch }, () =>
        Array.from({ length: cw }, () => Math.random() < fillProb)
    )

    const dirs = [
        [-1, -1],
        [0, -1],
        [1, -1],
        [-1, 0],
        [1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
    ]

    for (let s = 0; s < steps; s++) {
        const next = grid.map((r) => r.slice())
        for (let y = 0; y < ch; y++) {
            for (let x = 0; x < cw; x++) {
                let cnt = 0
                for (const [dx, dy] of dirs) {
                    const nx = x + dx,
                        ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= cw || ny >= ch);
                    else if (grid[ny][nx]) cnt++
                }
                next[y][x] = cnt > 3
            }
        }
        grid = next
    }
    return grid
}

/****************  heightmap synthesis  ****************/
function generateHeightmap(w = 512, h = 512) {
    const scale = 0.125
    const cw = Math.ceil(w * scale),
        ch = Math.ceil(h * scale)
    const landmask = generateLandmask(cw, ch)

    const hm = Array.from({ length: h }, () => new Float32Array(w))
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const land = landmask[Math.floor(y * scale)][Math.floor(x * scale)]
                ? 1
                : 0
            const n = layeredNoise(x / w, y / h)
            hm[y][x] = land ? n : n * 0.5 // water = 0, land 0.6-1.0
        }
    }
    for (let i = 0; i < blurSteps; i++) {
        for (let y = 1; y < h-1; y++) {
            for (let x = 1; x < w-1; x++) {
                hm[x][y] = Blur(x,y,hm)
            }
        }
    }
    return hm
}

const map = generateHeightmap(width, height)

/****************  rendering  ****************/
function drawHeightmap(hm) {
    const h = hm.length,
        w = hm[0].length,
        outW = w * drawSize,
        outH = h * drawSize
    const img = ctx.createImageData(outW, outH)

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const v = Math.floor(hm[y][x] * 255)
            for (let dy = 0; dy < drawSize; dy++) {
                for (let dx = 0; dx < drawSize; dx++) {
                    const px = x * drawSize + dx
                    const py = y * drawSize + dy
                    const idx = (py * outW + px) * 4 // RGBA stride
                    img.data[idx] = img.data[idx + 1] = img.data[idx + 2] = v
                    img.data[idx + 3] = 255
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0)
}

function setupCanvas() {
    canvas.width = width * drawSize
    canvas.height = height * drawSize
}

setupCanvas()
drawHeightmap(map)
