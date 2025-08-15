// Terrain generator & renderer – simple Math.random() noise
// Draw scaling controlled via `drawSize`

/****************  configuration  ****************/
const canvas = document.getElementById('c')
const ctx = canvas.getContext('2d')

const width = 2056
const height = 2056
const drawSize = 1 //needs to be an int >= 1
const blurDist = 4
const blurSteps = 8
const controls = []
const camSpeed = 2
const zoomSpeed = 1
let camX = Math.floor(width / 4) //Camera left
let camY = Math.floor(height / 4) //Camera top
let camZ = 0.5 // (camZ*drawSize*256)**2 = how many pixels the viewport covers
let camZMin = 0.5
/****************  helper functions  ****************/
const lerp = (a, b, t) => a + (b - a) * t
function clamp(num, min, max) {
    if (num < min) {
        return min
    } else if (num > max) {
        return max
    }
    return num
}
function Blur(x, y, hm) {
    let blurVal = 0
    for (let dx = -blurDist; dx < blurDist; dx++) {
        for (let dy = -blurDist; dy < blurDist; dy++) {
            if (hm[x + dx]) {
                if (hm[x + dx][y + dy]) {
                    blurVal += hm[x + dx][y + dy]
                } else {
                    if (0 <= x + dx && x + dx <= width) {
                        if (y + dy >= height) {
                            blurVal += hm[x + dx][y + dy - height]
                        }
                        if (y + dy <= 0) {
                            blurVal += hm[x + dx][y + dy + height]
                        }
                    } else if (0 <= y + dy && x + dy <= height) {
                        if (x + dx >= width) {
                            blurVal += hm[x + dx - width][y + dy]
                        }
                        if (x + dx <= 0) {
                            blurVal += hm[x + dx + width][y + dy]
                        }
                    }
                }
            }
        }
    }
    blurVal /= (blurDist * 2 + 1) ** 2
    return blurVal * (1 + blurDist / 20) //to adjust for darkening
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
function generateLandmask(cw, ch, fillProb = 0.4, steps = 5) {
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
    const scale = 1 / 16
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
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                hm[x][y] = Blur(x, y, hm)
            }
        }
    }
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            hm[x][y] = (hm[x][y] - 0) / (0.5 - 0)
        }
    }
    // for (let y = 1; y < h - 1; y++) {
    //     for (let x = 1; x < w - 1; x++) {
    //         const angle = 0.8 // radians → SW-NE
    //         const ridge = Math.sin((x * Math.cos(angle) + y * Math.sin(angle)) * 0.005) * 0.3
    //         hm[x][y] += ridge
    //     }
    // }
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            let h = hm[x][y]
            h = Math.pow(h, 1)
            h = clamp(h, 0, 1)
            hm[x][y] = h
        }
    }
    return hm
}

const map = generateHeightmap(width, height)

/****************  rendering  ****************/
function drawHeightmap(hm) {
    let v
    const h = hm.length
    const w = hm[0].length
    const outW = w * drawSize
    const outH = h * drawSize

    const img = ctx.createImageData(outW, outH)
    ctx.clearRect(0, 0, width, height)
    for (let y = camY; y < camY + camZ * drawSize * 256; y++) {
        for (let x = camX; x < camX + camZ * drawSize * 256; x++) {
            v = Math.floor(hm[Math.floor(camZ * y)][Math.floor(camZ * x)] * 255)
            for (let dy = 0; dy < drawSize; dy++) {
                for (let dx = 0; dx < drawSize; dx++) {
                    const px = x * drawSize + dx - camX * drawSize
                    const py = y * drawSize + dy - camY * drawSize
                    const idx = (py * outW + px) * 4
                    img.data[idx] = img.data[idx + 1] = img.data[idx + 2] = v
                    img.data[idx + 3] = 255
                }
            }
        }
    }
    ctx.putImageData(img, 0, 0)
}
/****************  Event Listeners  ****************/
document.addEventListener('keydown', function (event) {
    if (event.key === 'w') {
        controls[0] = true
    }
    if (event.key === 'a') {
        controls[1] = true
    }
    if (event.key === 's') {
        controls[2] = true
    }
    if (event.key === 'd') {
        controls[3] = true
    }
    if (event.key === 'q') {
        controls[4] = true
    }
    if (event.key === 'e') {
        controls[5] = true
    }
})

document.addEventListener('keyup', function (event) {
    if (event.key === 'w') {
        controls[0] = false
    }
    if (event.key === 'a') {
        controls[1] = false
    }
    if (event.key === 's') {
        controls[2] = false
    }
    if (event.key === 'd') {
        controls[3] = false
    }
    if (event.key === 'q') {
        controls[4] = false
    }
    if (event.key === 'e') {
        controls[5] = false
    }
})


/****************  Game Loop  ****************/
function handleInputs() {
    if (controls[0]) {
        if (camY - camSpeed >= 0) {
            camY -= camSpeed
        }
    }
    if (controls[1]) {
        if (camX - camSpeed >= 0) {
            camX -= camSpeed
        }
    }
    if (controls[2]) {
        if (camSpeed + camY + camZ * drawSize * 256 <= height) {
            camY += camSpeed
        }
    }
    if (controls[3]) {
        if (camSpeed + camX + camZ * drawSize * 256 <= width) {
            camX += camSpeed
        }
    }
    if (controls[4]) {
        if (camZ - zoomSpeed/100 >= camZMin) {
        camZ -= zoomSpeed/100
        // console.log(camZ)
        }
    }
    if (controls[5]) {
        if (camX+((camZ + zoomSpeed/100)* drawSize * 256) <= width && camY+((camZ + zoomSpeed/100)* drawSize * 256) <= height ) {
        camZ += zoomSpeed/100
        // console.log(camZ)
        } else if (camX - ((camZ + zoomSpeed/100)* drawSize * 256) >= 0) {
            camZ += zoomSpeed/100
            camX -= camSpeed
            console.log("!")
        }
         else if (camY - ((camZ + zoomSpeed/100)* drawSize * 256) >= 0) {
            camZ += zoomSpeed/100
            camY -= camSpeed
        }
    }
}
function mainLoop() {
    handleInputs()
    drawHeightmap(map)
}
document.addEventListener('keyup', function (event) {
    controls[event.key] = false
})
function setupCanvas() {
    canvas.width = width * drawSize * camZ
    canvas.height = height * drawSize * camZ
}

setupCanvas()
drawHeightmap(map)

setInterval(mainLoop, 20)
