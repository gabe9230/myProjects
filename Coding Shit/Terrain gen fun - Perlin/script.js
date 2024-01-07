let canvas = document.getElementById("c")
let ctx = canvas.getContext("2d")
let pixelSize = 16
let width = Math.floor(window.innerWidth/pixelSize)
let height = Math.floor(window.innerHeight/pixelSize)
let grid = []
let heightMap = []
let moistureMap = []
let noiseFrequency = 3
let layer1Mult = 1
let layer2Mult = 0.5
let layer3Mult = 0.25
let cameraX = 500
let cameraY = 500
let player = {
  x: cameraX+(width/2),
  y: cameraY+(width/2),
}
let controls = []

let perlin = {
    rand_vect: function(){
        let theta = Math.random() * 2 * Math.PI;
        return {x: Math.cos(theta), y: Math.sin(theta)};
    },
    dot_prod_grid: function(x, y, vx, vy){
        let g_vect;
        let d_vect = {x: x - vx, y: y - vy};
        if (this.gradients[[vx,vy]]){
            g_vect = this.gradients[[vx,vy]];
        } else {
            g_vect = this.rand_vect();
            this.gradients[[vx, vy]] = g_vect;
        }
        return d_vect.x * g_vect.x + d_vect.y * g_vect.y;
    },
    smootherstep: function(x){
        return 6*x**5 - 15*x**4 + 10*x**3;
    },
    interp: function(x, a, b){
        return a + this.smootherstep(x) * (b-a);
    },
    seed: function(){
        this.gradients = {};
        this.memory = {};
    },
    get: function(x, y) {
        if (this.memory.hasOwnProperty([x,y]))
            return this.memory[[x,y]];
        let xf = Math.floor(x);
        let yf = Math.floor(y);
        //interpolate
        let tl = this.dot_prod_grid(x, y, xf,   yf);
        let tr = this.dot_prod_grid(x, y, xf+1, yf);
        let bl = this.dot_prod_grid(x, y, xf,   yf+1);
        let br = this.dot_prod_grid(x, y, xf+1, yf+1);
        let xt = this.interp(x-xf, tl, tr);
        let xb = this.interp(x-xf, bl, br);
        let v = this.interp(y-yf, xt, xb);
        this.memory[[x,y]] = v;
        return v;
    }
}
perlin.seed();

function noise(x,y) {
  var out = perlin.get(x,y)
  return out
}

function dist(x1,y1,x2,y2) {
  var a = x1 - x2
  var b = y1 - y2

  return Math.sqrt( a*a + b*b )
}

function perlinToGrid(e,m) {
  if (e < 0.015) {return 0} else
  if (e < 0.05) {return 1} else
  if (e < 0.07) {return 2} else
  if (e < 0.1) {return 3} else
  if (e > 0.8) {
    if (m < 0.1) {return 4} else if (m < 0.2) {return 5} else if (m < 0.5) {return 6} else if (m <= 1.1) {return 7} 
  } else if (e > 0.6) {
    if (m < 0.33) {return 8} else if (m < 0.66) {return 9} else  if (m < 1.1) {return 16} 
  } else if (e > 0.3) {
    if (m < 0.16) {return 8} else if (m < 0.50) {return 10} else if (m < 0.83) {return 11} else if (m < 1.1) {return 12}
  } else if (m < 0.16) {return 13} else
  if (m < 0.33) {return 10} else if (m < 0.66) {return 14} else if (m < 1.1) {return 15}
}
function perlinMask() {
  for (let x = cameraX-1;x<width+cameraX+1;x++) {
    for (let y = cameraY-1; y<height+cameraY+1;y++) {
      var nx = (x/width + 1 - 0.5)
      var ny = (y/height + 1 - 0.5)
      var e = layer1Mult * noise(noiseFrequency * nx, noiseFrequency * ny) +  layer2Mult * noise(noiseFrequency * 2 * nx, noiseFrequency * 2 * ny) + layer3Mult * noise(noiseFrequency * 4 * nx, noiseFrequency * 4 * ny)
      e = e / (layer1Mult + layer2Mult + layer3Mult)
      heightMap[x][y] = e
    }
  }
  perlin.seed
  for (let x = cameraX-1;x<width+cameraX+1;x++) {
    for (let y = cameraY-1; y<height+cameraY+1;y++) {
      var nx = (x/width + 1 - 0.5)
      var ny = (y/height + 1 - 0.5)
      var m = layer1Mult * noise(noiseFrequency * nx, noiseFrequency * ny) +  layer2Mult * noise(noiseFrequency * 2 * nx, noiseFrequency * 2 * ny) + layer3Mult * noise(noiseFrequency * 4 * nx, noiseFrequency * 4 * ny)
      m = m / (layer1Mult + layer2Mult + layer3Mult)
      moistureMap[x][y] = m
    }
  }
  for (let x = cameraX-1;x<width+cameraX+1;x++) {
    for (let y = cameraY-1; y<height+cameraY+1;y++) {
      grid[x][y].val = perlinToGrid(heightMap[x][y],moistureMap[x][y])
    }
  }
}
function setup() {
  canvas.width = width*pixelSize
  canvas.height = height*pixelSize
}

function setupGrid() {
  for (let x = 0;x<cameraX*2;x++) {
    grid.push([])
    for (let y = 0; y<cameraY*2;y++) {
      grid[x].push({
        val: 0,
      })
    }
  }
  for (let x = 0;x<cameraX*2;x++) {
    heightMap.push([])
    for (let y = 0; y<cameraY*2;y++) {
      heightMap[x].push(0)
    }
  }
  for (let x = 0;x<cameraX*2;x++) {
    moistureMap.push([])
    for (let y = 0; y<cameraY*2;y++) {
      moistureMap[x].push(0)
    }
  }
}
function removeEdge() {
  for (let x = 0;x<grid.length;x++) {
    for (let y = 0;y<grid[x].length;y++) {
      if (x === 0 || x === grid.length || y === 0 || y === grid[x].length) {
        grid[x][y].val = 0
      }
    }
  }
}

// deepWater = 0
// water = 1
// shallowWater = 2
// beach = 3
// scorched = 4
// bare = 5
// tundra = 6
// snow = 7
// temperateDesert = 8
// shrubland = 9
// grassLand = 10
// temperateDecForest = 11
// temperateRainForest = 12
// subTropicalDesert = 13
// tropicalSeasonalForest = 14
// tropicalRainforest = 15
// taiga = 16
function draw() {
  for (x = cameraX+1;x<width+cameraX-1;x++) {
    for (y = cameraY+1;y<height+cameraY-1;y++) {
      if (grid[x][y].val == 0) {
        ctx.fillStyle = "rgb(28, 10, 173)"
      } 
      if (grid[x][y].val == 1) {
        ctx.fillStyle = "rgb(0, 98, 255)"
      }
      if (grid[x][y].val == 2) {
        ctx.fillStyle = "rgb(59, 134, 255)"
      }
      if (grid[x][y].val == 3) {
        ctx.fillStyle = "rgb(230, 211, 108)"
      }
      if (grid[x][y].val == 4) {
        ctx.fillStyle = "rgb(46, 46, 46)"
      }
      if (grid[x][y].val == 5) {
        ctx.fillStyle = "rgb(105, 99, 79)"
      }
      if (grid[x][y].val == 6) {
        ctx.fillStyle = "rgb(133, 166, 133)"
      }
      if (grid[x][y].val == 7) {
        ctx.fillStyle = "rgb(200, 200, 200)"
      }
      if (grid[x][y].val == 8) {
        ctx.fillStyle = "rgb(196, 217, 145)"
      }
      if (grid[x][y].val == 9) {
        ctx.fillStyle = "rgb(156, 219, 140)"
      }
      if (grid[x][y].val == 10) {
        ctx.fillStyle = "rgb(59, 194, 54)"
      }
      if (grid[x][y].val == 11) {
        ctx.fillStyle = "rgb(84, 194, 56)"
      }
      if (grid[x][y].val == 12) {
        ctx.fillStyle = "rgb(41, 179, 19)"
      }
      if (grid[x][y].val == 13) {
        ctx.fillStyle = "rgb(217, 190, 37)"
      }
      if (grid[x][y].val == 14) {
        ctx.fillStyle = "rgb(51, 179, 0)"
      }
      if (grid[x][y].val == 15) {
        ctx.fillStyle = "rgb(45, 128, 13)"
      }
      if (grid[x][y].val == 16) {
        ctx.fillStyle = "rgb(19, 66, 0)"
      }
      if (x === player.x && y === player.y) {
          ctx.fillStyle = "rgb(0,0,0)"
      }
      ctx.fillRect((x-cameraX)*pixelSize,(y-cameraY)*pixelSize,pixelSize,pixelSize)
    }
  }
}

function west() {
  cameraX--
  player.x = Math.floor(cameraX+(width/2))
  perlinMask()
  draw()
}

function east() {
  cameraX++
  player.x = Math.floor(cameraX+(width/2))
  perlinMask()
  draw()
}

function north() {
  cameraY--
  player.y = Math.floor(cameraY+(height/2))
  perlinMask()
  draw()
}

function south() {
  cameraY++
  player.y = Math.floor(cameraY+(height/2))
  perlinMask()
  draw()
}

document.addEventListener('keydown', function(event) {
  if (event.key === "w") {
    controls[0] = true
  }
  if (event.key === "a") {
    controls[1] = true
  }
  if (event.key === "s") {
    controls[2] = true
  }
  if (event.key === "d") {
    controls[3] = true
  }
}) 

document.addEventListener('keyup', function(event) {
  if (event.key === "w") {

    controls[0] = false
  }
  if (event.key === "a") {
    controls[1] = false
  }
  if (event.key === "s") {
    controls[2] = false
  }
  if (event.key === "d") {
    controls[3] = false
  }
})

document.addEventListener('keyup', function(event) {
  controls[event.key] = false
}) 

function loop() {
  if (controls[0]) {
    north()
  }
  if (controls[1]) {
    west()
  }
  if (controls[2]) {
    south()
  }
  if (controls[3]) {
    east()
  }
  
}

setInterval(loop, 10)

function start() {
  setup()
  setupGrid()
  removeEdge()
  perlinMask()
  player.x = Math.floor(cameraX+(width/2))
  player.y = Math.floor(cameraY+(height/2))
  draw()
}
start()