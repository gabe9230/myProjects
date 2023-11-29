let canvas = document.getElementById("c")
let ctx = canvas.getContext("2d")
let width = 150
let height = 100
let offsetMax = 15
let itcount = 8
let line = []
let particles
let gravity = 0.5; // Downward force
let wind = 0.05; // Horizontal force, positive to the right


function setup() {
  // Assuming 'width' and 'height' are the dimensions of the particle array
  canvas.width = width * 4;  // Scale up the canvas width
  canvas.height = height * 4; // Scale up the canvas height
}


function init() {
  line.push([0, height / 1.2])
  line.push([width, height / 2])
}

function iterate() {
  for (let i = 0; i < line.length - 1; i += 2) {
    line.splice(i + 1, 0, [
      (line[i][0] + line[i + 1][0]) / 2,
      (Math.round((line[i][1] + line[i + 1][1]) / 2) + (Math.random() * offsetMax - offsetMax / 2))-(((line[i][0] + line[i + 1][0]) / 2)**3)*0.00000001
    ]);
  }
}



function smoothLine(smoothingLevel) {
  for (let level = 0; level < smoothingLevel; level++) {
    let newLine = [];

    // Always keep the first point
    newLine.push(line[0]);

    for (let i = 0; i < line.length - 1; i++) {
      let p0 = line[i];
      let p1 = line[i + 1];

      let q = [0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]];
      let r = [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]];

      newLine.push(q);
      newLine.push(r);
    }

    // Always keep the last point
    newLine.push(line[line.length - 1]);

    line = newLine;
  }
}


function getPixelsUnderneathLine() {
  let pixels = new Array(width).fill().map(() => new Array(height));

  // First pass: Create air and sand particles
  for (let x = 0; x < width; x++) {
    let yLine = getYCoordinateOfLineAtX(x);

    for (let y = 0; y < height; y++) {
      if (y > yLine) {
        // Below the line: sand particle
        pixels[x][y] = { type: 'sand', density: 1, adhesion: 0.05, gravityAccumulator: 0, windAccumulator: 0 };
      } else {
        // Above the line: air particle
        pixels[x][y] = { type: 'air', density: 0.01, adhesion: 0, gravityAccumulator: 0, windAccumulator: 0 };
      }
    }
  }

  // Second pass: Replace air with water below sea level
  for (let x = 0; x < width; x++) {
    for (let y = height*0.7; y < height; y++) {
      if (pixels[x][y].type === 'air') {
        // Replace air with water below sea level
        pixels[x][y] = { type: 'water', density: 0.8, adhesion: 0.1, gravityAccumulator: 0, windAccumulator: 0 };
      }
    }
  }

  return pixels;
}


function updatePhysics() {
  // Create a copy of the current state of particles
  let newParticles = JSON.parse(JSON.stringify(particles));

  // Generate a list of all particle positions and shuffle it
  let positions = [];
  for (let x = 0; x < particles.length; x++) {
    for (let y = 0; y < particles[x].length; y++) {
      positions.push([x, y]);
    }
  }
  shuffleArray(positions); // Randomize the update order

  for (let pos of positions) {
    let x = pos[0];
    let y = pos[1];
    let particle = particles[x][y];

    if (particle) {
      // Accumulate gravity effect for all particles
      particle.gravityAccumulator += gravity;

      // Accumulate wind effect based on adhesion
      particle.windAccumulator += wind * (1 - particle.adhesion);

      let moveX = 0;
      let moveY = 0;

      // Check if accumulated gravity is enough to move the particle down
      if (particle.gravityAccumulator >= 1) {
        moveY = 1;
        particle.gravityAccumulator -= 1;
      }

      // Check if accumulated wind is enough to move the particle horizontally
      if (particle.windAccumulator >= 1) {
        moveX = 1;
        particle.windAccumulator -= 1;
      } else if (particle.windAccumulator <= -1) {
        moveX = -1;
        particle.windAccumulator += 1;
      }

      // Calculate new position
      let newX = x + moveX;
      let newY = y + moveY;

      // Boundary checks and position updates
      newX = Math.min(Math.max(newX, 0), width - 1);
      newY = Math.min(Math.max(newY, 0), height - 1);

      // Check if the new position is within bounds and not occupied
      if (newParticles[newX][newY] === null) {
        newParticles[newX][newY] = particle;
        newParticles[x][y] = null; // Clear the old position
      }
    }
  }

  // Update the particles array with the new state
  particles = newParticles;
}

function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    let j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function getYCoordinateOfLineAtX(x) {
  // Find the two points in 'line' between which 'x' lies
  for (let i = 0; i < line.length - 1; i++) {
    if (x >= line[i][0] && x <= line[i + 1][0]) {
      // Linear interpolation
      let t = (x - line[i][0]) / (line[i + 1][0] - line[i][0]);
      return line[i][1] * (1 - t) + line[i + 1][1] * t;
    }
  }
  return height; // Default if not found
}

function draw() {
  ctx.clearRect(0, 0, width, height);
  for (let x = 0; x < particles.length; x++) {
    for (let y = 0; y < particles[x].length; y++) {
      let particle = particles[x][y];
      if (particle) {
        switch (particle.type) {
          case 'sand':
            ctx.fillStyle = "RGB(156, 126, 76)"; // Color for sand
            break;
          case 'air':
            ctx.fillStyle = "RGBA(112, 226, 255, 0.5)"; // Color for air
            break;
          case 'water':
            ctx.fillStyle = "RGB(70, 130, 180)"; // Color for water
            break;
          default:
            continue; // Skip unknown types
        }
        ctx.fillRect(x * 4, y * 4, 4, 4); // Draw each particle as 4x4 pixels
      }
    }
  }
}





function start() {
  setup()
  init()
  for (let i = 0; i < itcount; i++) {
    iterate()
  }
  smoothLine(3)
  particles = getPixelsUnderneathLine()
  draw()
}

start()

function animate() {
  updatePhysics();
  draw();
  requestAnimationFrame(animate);
}

// Start the animation
animate();
