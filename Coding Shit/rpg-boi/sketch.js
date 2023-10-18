// function setup() {
let audio = document.getElementById("audio")
audio.loop = true
let canvas = document.getElementById("canvas")
let ctx = canvas.getContext("2d")
let villageSpr = document.getElementById("village")
let slimeSpr = document.getElementById("slime")
let playerSprite = document.getElementById("player")
let grassSprite = document.getElementById("grass")
let forestSprite = document.getElementById("forest")
let beachSpr = document.getElementById("beach")
let stoneSpr = document.getElementById("stone")
let iceSpr = document.getElementById("ice")
let snekSpr = document.getElementById("snek")
let ogreSpr = document.getElementById("ogre")
let oceanSpr = document.getElementById("ocean")
let pineSpr = document.getElementById("pine")

let camSize = 30
// let eastereggCode = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65]
let mainMap = []
var worldsize = 512;
let multiplier = 0.05
var keys = [];
let encounterChance = 0.02
let playerHealth = 20
let playerDamage = 5
let defPDamage = 5
let defPHealth = 20
let monsterCount = 35
let moistureMap = []
let gold = 500
let maxhp = 20
let boat = false
let arrows = 0
let bow = false
let shaking = false


function clamp(value, min, max) {
  if (value < min) return min;
  else if (value > max) return max;
  return value;
}


let monsters = []


function populate() {
  for (var posx = camSize; posx < worldsize - camSize; posx++) {
    for (var posy = camSize; posy < worldsize - camSize; posy++) {
      if (Math.random() <= encounterChance) {
        if (map[posx][posy].landType !== 0) {
          if (Math.random() < 1 / monsterCount) {
            monsters.push({
              hp: 10,
              dp: 2,
              name: "slime",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 1,
              dpGainMin: 1,
              hpGainMax: 3,
              hpGainMin: 2,
              goldDrop: 5,
              agro: false,
              agroRange: 2,
              followDist: 6,
              followCount: 0
            })
          } else if (Math.random() < 1 / monsterCount) {
            monsters.push({
              hp: 15,
              dp: 10,
              name: "skeleton",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 3,
              dpGainMin: 2,
              hpGainMax: 5,
              hpGainMin: 3,
              goldDrop: 10,
              agro: false,
              agroRange: 2,
              followDist: 6,
              followCount: 0
            })
          } else if (Math.random() < 1 / monsterCount) {
            monsters.push({
              hp: 20,
              dp: 13,
              name: "zombie",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 3,
              dpGainMin: 2,
              hpGainMax: 5,
              hpGainMin: 3,
              goldDrop: 20,
              agro: false,
              agroRange: 2,
              followDist: 6,
              followCount: 0
            })
          } else if (Math.random() < 0.7 / monsterCount) {
            monsters.push({
              hp: 1,
              dp: 1,
              name: "chicken",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 0,
              dpGainMin: 0,
              hpGainMax: 0,
              hpGainMin: 0,
              goldDrop: 5,
              agro: false,
              agroRange: 4,
              followDist: 50,
              followCount: 0
            })
          } else if (Math.random() < 1 / monsterCount) {
            monsters.push({
              hp: 10,
              dp: 15,
              name: "snek",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 2,
              dpGainMin: 3,
              hpGainMax: 5,
              hpGainMin: 6,
              goldDrop: 10,
              agro: false,
              agroRange: 1,
              followDist: 8,
              followCount: 0
            })
          } else if (Math.random() < 0.5 / monsterCount) {
            monsters.push({
              hp: 30,
              dp: 15,
              name: "ogre",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 5,
              dpGainMin: 6,
              hpGainMax: 9,
              hpGainMin: 10,
              goldDrop: 30,
              agro: false,
              agroRange: 3,
              followDist: 9,
              followCount: 0
            })
          } else if (Math.random() < 0.25 / monsterCount) {
            monsters.push({
              hp: 40,
              dp: 25,
              name: "fire elemental",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 10,
              dpGainMin: 11,
              hpGainMax: 13,
              hpGainMin: 14,
              goldDrop: 50,
              agro: false,
              agroRange: 5,
              followDist: 8,
              followCount: 0
            })
          } else if (Math.random() < 0.25 / monsterCount) {
            monsters.push({
              hp: 50,
              dp: 15,
              name: "ice elemental",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 10,
              dpGainMin: 11,
              hpGainMax: 13,
              hpGainMin: 14,
              goldDrop: 50,
              agro: false,
              agroRange: 5,
              followDist: 6,
              followCount: 0
            })
          } else if (Math.random() < 0.125 / monsterCount) {
            monsters.push({
              hp: 75,
              dp: 35,
              name: "leyoth",
              x: clamp(posx, camSize, worldsize - camSize),
              y: clamp(posy, camSize, worldsize - camSize),
              dpGainMax: 15,
              dpGainMin: 16,
              hpGainMax: 17,
              hpGainMin: 18,
              goldDrop: 100,
              agro: false,
              agroRange: 5,
              followDist: 5,
              followCount: 0
            })
          }
        }
      }
    }
  }
}


function getDist(x1, y1, x2, y2) {
  var a = x1 - x2
  var b = y1 - y2
  var c = Math.sqrt(a * a + b * b)
  return c
}


function agroAi() {
  for (var i = 0; i < monsters.length - 1; i++) {
    if (getDist(playerx, playery, monsters[i].x, monsters[i].y) < monsters[i].agroRange) {
      if (monsters[i].followDist > monsters[i].followCount) {
        if (playerx > monsters[i].x) {
          monsters[i].x += 1
        }
        if (playerx < monsters[i].x) {
          monsters[i].x -= 1
        }
        if (playery > monsters[i].y) {
          monsters[i].y += 1
        }
        if (playery < monsters[i].y) {
          monsters[i].y -= 1
        }
        monsters[i].followCount += 1
      } else {
        monsters[i].followCount = 0
      }
    }
  }
}


window.setInterval(checkOverlap, 10)
let attackOrDefend = "attack"
let combatOver = true


function PStrike(monster) {
  monsters[monster].hp -= playerDamage
  if (monsters[monster].hp <= 0) {
    let hpGained = Math.round(Math.random() * monsters[monster].hpGainMin) + monsters[monster].hpGainMax
    let dpGained = Math.round(Math.random() * monsters[monster].dpGainMin) + monsters[monster].dpGainMax
    changeText("Congrats! you beat a: " + monsters[monster].name + "! and leveled up gaining: " + hpGained + " hp, and " + dpGained + " dp!")
    defPDamage += dpGained
    playerHealth += hpGained
    maxhp += hpGained
    gold += monsters[monster].goldDrop
    monsters.splice(monster, 1)
    combatOver = true
    attackOrDefend = "defend"
  } else {

    attackOrDefend = "defend"
  }
}


function EStrike(monster) {
  if (shieldHp > 0) {
    shieldHp -= monsters[monster].dp
  } else {
    playerHealth -= monsters[monster].dp
  }
  if (playerHealth <= 0) {
    combatOver = true
    activateEgg()
  }
  if (playerHealth > 0) {
    attackOrDefend = "attack"
  }
}


function combat(chosen) {
  combatOver = false
  let chosenMonster = chosen
  while (combatOver === false) {
    if (attackOrDefend === "attack") {
      PStrike(chosenMonster)
    }
    if (attackOrDefend === "defend") {
      EStrike(chosenMonster)
    }
  }
}
let keyDown = false


function encounter() {
  for (var i = 0; i < monsters.length - 1; i++) {
    if (playerx === monsters[i].x && playery === monsters[i].y) {
      combat(i)
    }
  }
}


document.onkeydown = function(e) {
  audio.play()
  if (e.keyCode === 37) {
    if (keyDown === false) {
      if (arrows > 0) {
        // spawnArrow("left")
        keyDown = true
      }
    }
  }
  if (e.keyCode === 38) {
    if (keyDown === false) {
      if (arrows > 0) {
        // spawnArrow("up")
        keyDown = true
      }
    }
  }
  if (e.keyCode === 39) {
    if (keyDown === false) {
      if (arrows > 0) {
        // spawnArrow("right")
        keyDown = true
      }
    }
  }
  if (e.keyCode === 40) {
    if (keyDown === false) {
      if (arrows > 0) {
        // spawnArrow("down")
        keyDown = true
      }
    }
  }
  keys[e.keyCode] = true
  if (e.keyCode === 49) {
    TradeHp()
  }
  if (e.keyCode === 50) {
    TradeDp()
  }
  if (e.keyCode === 51) {
    tradeBoat()
  }
  if (e.keyCode === 52) {
    tradeShield()
  }
}


document.onkeyup = function(e) {
  if (e.keyCode === 37) {
    if (keyDown === true) {
      keyDown = false
    }
  }
  if (e.keyCode === 38) {
    if (keyDown === true) {
      keyDown = false
    }
  }
  if (e.keyCode === 39) {
    if (keyDown === true) {
      keyDown = false
    }
  }
  if (e.keyCode === 40) {
    if (keyDown === true) {
      keyDown = false
    }
  }
  keys[e.keyCode] = false
}


let text01 = document.getElementById("text01")
canvas.width = camSize * 32
canvas.height = camSize * 32
let position = ""
let map = []
let playerPlaced = false
let scale = 0.02


function generateMap(x1, y1) {
  for (var x = 0; x < x1; x++) {
    map.push([])
    moistureMap.push([])
    for (var y = 0; y < y1; y++) {
      map[x].push({ landType: 0, player: false, city: false })
      moistureMap[x].push(0)
    }
  }
}



let playerx = Math.floor(worldsize / 2)
let playery = Math.floor(worldsize / 2)


function placeCities() {
  for (var x = 1; x < worldsize - 1; x++) {
    for (var y = 1; y < worldsize - 1; y++) {
      if (map[x][y].landType !== 0 && map[x][y].landType !== 1 && map[x][y].landType !== 6 && map[x][y].landType !== 5) {
        if (Math.random() < 0.005) {
          map[x][y].city === true
          console.log(String(x)+" "+String(y))
        }
      }
    }
  }
}


function tradeBoat() {
  if (gold >= 50) {
    if (map[playerx][playery].landType === 7) {
      if (boat === false) {
        gold -= 50
        boat = true
      }
    }
  }
}


function floodCheck(x, y, type0, type1) {
  if (map[x][y].landType === type0) {
    if (map[x + 1][y + 1].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x + 1, y + 1, type0, type1)
    } else if (map[x - 1][y - 1].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x - 1, y - 1, type0, type1)
    } else if (map[x + 1][y - 1].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x + 1, y - 1, type0, type1)
    } else if (map[x - 1][y + 1].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x - 1, y + 1, type0, type1)
    } else if (map[x][y + 1].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x, y + 1, type0, type1)
    } else if (map[x + 1][y].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x + 1, y, type0, type1)
    } else if (map[x][y - 1].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x, y - 1, type0, type1)
    } else if (map[x - 1][y].landType === type0) {
      map[x][y].landType = type1
      floodCheck(x - 1, y, type0, type1)
    }
  }
}


function checkOverlap() {
  for (var i = 0; i < monsters.length; i++) {
    for (var q = 0; q < monsters.length; q++) {
      if (i !== q) {
        if (monsters[i].x === monsters[q].x && monsters[i].y === monsters[q].y) {
          monsters.splice(i, 1)
          if (monsters[i].direction !== undefined || monsters[q].direction !== undefined) {
            monsters.splice(q, 1)
          }
        }

      }
    }
  }
}


function checkIfBeach() {
  for (var x = 1; x < worldsize - 1; x++) {
    for (var y = 1; y < worldsize - 1; y++) {
      if (map[x + 1][y + 1].landType === 6) {
        map[x][y].landType = 5
      } else if (map[x - 1][y - 1].landType === 6) {
        map[x][y].landType = 5
      } else if (map[x + 1][y - 1].landType === 6) {
        map[x][y].landType = 5
      } else if (map[x - 1][y + 1].landType === 6) {
        map[x][y].landType = 5
      } else if (map[x][y + 1].landType === 6) {
        map[x][y].landType = 5
      } else if (map[x + 1][y].landType === 6) {
        map[x][y].landType = 5
      } else if (map[x][y - 1].landType === 6) {
        map[x][y].landType = 5
      } else if (map[x - 1][y].landType === 6) {
        map[x][y].landType = 5
      }
    }
  }
}


//Ocean = 0
//beach = 1
//grassLands = 2
//forest = 3
//mountainForest = 4
//mountains = 5
//snow = 6
//wall = 7
//desert = 8
//scorched = 9
//rainforest = 10
function biomes(m, h, x, y) {
  if (h < 10) {
    map[x][y].landType = 0
  } else if (h < 20) {
    map[x][y].landType = 1
  } else if (h < 25) {
    if (m < 30) {
      map[x][y].landType = 8
    } else if (m < 60) {
      map[x][y].landType = 2
    } else {
      map[x][y].landType = 10
    }
  } else if (h < 35) {
    if (m < 30) {
      map[x][y].landType = 9
    } else if (m < 60) {
      map[x][y].landType = 3
    } else {
      map[x][y].landType = 0
    }
  } else if (h < 65) {
    if (m < 30) {
      map[x][y].landType = 5
    } else if (m < 60) {
      map[x][y].landType = 4
    } else {
      map[x][y].landType = 4
    }
  } else {
    map[x][y].landType = 6
  }

}

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
perlin.seed()
function noise(x,y) {
var out = perlin.get(x,y)
return out
}


function applyNoise() {
  var yoff = 0
  for (var x = 0; x < worldsize; x++) {
    var xoff = 0
    for (var y = 0; y < worldsize; y++) {
      var r = noise(xoff, yoff) * 100
      map[x][y].val = r+10
      xoff += multiplier
    }
    yoff += multiplier
  }
  yoff = 0
  for (var x = 0; x < worldsize; x++) {
    xoff = 0
    for (var y = 0; y < worldsize; y++) {
      r = noise(xoff, yoff) * 100
      moistureMap[x][y] = r+10
      xoff += multiplier
    }
    yoff += multiplier
  }
  for (var x = 0; x < worldsize; x++) {
    for (var y = 0; y < worldsize; y++) {
      biomes(moistureMap[x][y], map[x][y].val, x, y)
    }
  }
}


function changeText(val) {
  text01.innerHTML = val
}


let shield = false
function tradeShield() {
  if (map[playerx][playery].landType === 7) {
    if (gold >= 20) {
      if (shield === false) {
        gold -= 20
        shield = true
        shieldHp = 10
      }
    }
  }
}


function tradeBow() {
  if (map[playerx][playery].landType === 7) {
    if (gold >= 60) {
      if (bow === false) {
        gold -= 60
        bow = true
      }
    }
  }
}


function tradeArrow() {
  if (map[playerx][playery].landType === 7) {
    if (gold >= 10) {
      gold -= 10
      arrows += 1
    }
  }
}


function activateEgg() {
  playerx = 1
  playery = 1
  camSize = 3
  worldsize = 4
  canvas.width = camSize * 32
  canvas.height = camSize * 32
  map = []
  generateMap(4, 4)
  map[1][1].player = true
  for (var x = 0; x < worldsize; x++) {
    for (var y = 0; y < worldsize; y++) {
      map[x][y].landType = 5
    }
  }
  changeText("you died...")
}


let keysPressed = [false, false, false, false, false, false, false, false, false, false]
function checkEgg() {
  if (keys[eastereggCode[0]] === true) {
    keysPressed[0] = true
  }
  if (keys[eastereggCode[1]] === true) {
    keysPressed[1] = true
  }
  if (keys[eastereggCode[2]] === true) {
    keysPressed[2] = true
  }
  if (keys[eastereggCode[3]] === true) {
    keysPressed[3] = true
  }
  if (keys[eastereggCode[4]] === true) {
    keysPressed[4] = true
  }
  if (keys[eastereggCode[5]] === true) {
    keysPressed[5] = true
  }
  if (keys[eastereggCode[6]] === true) {
    keysPressed[6] = true
  }
  if (keys[eastereggCode[7]] === true) {
    keysPressed[7] = true
  }
  if (keys[eastereggCode[8]] === true) {
    keysPressed[8] = true
  }
  if (keys[eastereggCode[9]] === true) {
    keysPressed[9] = true
  }
  if (keysPressed[0] === true) {
    if (keysPressed[1] === true) {
      if (keysPressed[2] === true) {
        if (keysPressed[3] === true) {
          if (keysPressed[4] === true) {
            if (keysPressed[5] === true) {
              if (keysPressed[6] === true) {
                if (keysPressed[7] === true) {
                  if (keysPressed[8] === true) {
                    if (keysPressed[9] === true) {
                      activateEgg()
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}


function playerSpawn() {
  if (map[playerx][playery].landType === 0) {
    playerx = clamp(Math.floor(Math.random() * worldsize), camSize, worldsize - camSize)
    playery = clamp(Math.floor(Math.random() * worldsize), camSize, worldsize - camSize)
    screenx = playerx - 10
    screeny = playery - 10
  }
  loop()
}


function arrowsFunc() {
  for (var i = 1; i < monsters.length; i++) {
    if (monsters[i - 1].direction !== undefined) {
      // console.log(monsters[i - 1].direction)
      if (monsters[i].direction === "up") {
        monsters[i].y -= 1
      }
      if (monsters[i].direction === "down") {
        monsters[i].y += 1
      }
      if (monsters[i].direction === "left") {
        monsters[i].x -= 1
      }
      if (monsters[i].direction === "right") {
        monsters[i].x += 1
      }
      monsters[i].time += 1
      if (monsters[i].time >= 4) {
        monsters.splice(i, 1)
      }
    }
  }
}


function boundries() {
  for (var x = 0; x < worldsize; x++) {
    for (var y = 0; y < worldsize; y++) {
      if (x === camSize - 1) {
        if (y > camSize - 2 && y < worldsize - (camSize) + 2) {
          map[x][y].landType = 7
        }
      }
      if (x === worldsize - (camSize) + 1) {
        if (y > camSize - 2 && y < worldsize - (camSize) + 2) {
          map[x][y].landType = 7
        }
      }
      if (y === camSize - 1) {
        if (x > camSize - 2 && x < worldsize - (camSize) + 2) {
          map[x][y].landType = 7
        }
      }
      if (y === worldsize - (camSize) + 1) {
        if (x > camSize - 2 && x < worldsize - (camSize) + 2) {
          map[x][y].landType = 7
        }
      }
    }
  }
}


function TradeHp() {
  if (map[playerx][playery].landType === 7) {
    if (gold >= 2) {
      gold -= 2
      maxhp += 1
      playerHealth += 1
    }
  }
}


function TradeDp() {
  if (map[playerx][playery].landType === 7) {
    if (gold >= 4) {
      gold -= 4
      defPDamage += 1
    }
  }
}


function regen() {
  if (map[playerx][playery].landType === 7) {
    if (playerHealth < maxhp) {
      playerHealth += 1
    }
  }
  if (shield) {
    if (shieldHp < 10) {
      shieldHp += 1
    }
  }
}


function spawnArrow(direction1) {
  if (direction1 === "left") {
    monsters.push({
      x: playerx - 1,
      y: playery,
      direction: "left",
      time: 0
    })
  }
  if (direction1 === "right") {
    monsters.push({
      x: playerx + 1,
      y: playery,
      direction: "right",
      time: 0
    })
  }
  if (direction1 === "up") {
    monsters.push({
      x: playerx,
      y: playery - 1,
      direction: "up",
      time: 0
    })

  }
  if (direction1 === "down") {
    monsters.push({
      x: playerx,
      y: playery + 1,
      direction: "down",
      time: 0
    })
    // console.log("down")
  }
  arrows -= 1
}


let shieldHp = 0
// window.setInterval(arrowsFunc, 10)
// window.setInterval(regen, 500)
generateMap(worldsize, worldsize)
applyNoise()
checkIfBeach()
placeCities()
mainMap = map
// populate()
boundries()
playerSpawn()

function loop() {
  changeText("dp: " + defPDamage + ", hp: " + (playerHealth + shieldHp) + ", x: " + playerx + ", y: " + playery + " gold: " + gold + ", boat: " + boat + ", shield: " + shield)


  if (boat === false) {
    if (map[playerx + 1][playery].landType !== 0) {
      if (playerx <= worldsize - camSize - 1) {
        if (keys[68] === true) {
          playerx += 1
        }
      }
    }
    if (map[playerx - 1][playery].landType !== 0) {
      if (playerx >= camSize + 1) {
        if (keys[65] === true) {
          playerx -= 1
        }
      }
    }
    if (map[playerx][playery + 1].landType !== 0) {
      if (playery <= worldsize - camSize - 1) {
        if (keys[83] === true) {
          playery += 1
        }
      }
    }
    if (map[playerx][playery - 1].landType !== 0) {
      if (playery >= camSize + 1) {
        if (keys[87] === true) {
          playery -= 1
        }
      }
    }
  } else {
    if (playerx <= worldsize - camSize - 1) {
      if (keys[68] === true) {
        playerx += 1
      }
    }
    if (playerx >= camSize + 1) {
      if (keys[65] === true) {
        playerx -= 1
      }
    }
    if (playery <= worldsize - camSize - 1) {
      if (keys[83] === true) {
        playery += 1
      }
    }
    if (playery >= camSize + 1) {
      if (keys[87] === true) {
        playery -= 1
      }
    }
  }


  encounter()
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = "rgb(139, 195, 74)"
  ctx.fillRect(0, 0, canvas.width, canvas.width)
  var camX = clamp(playerx - (camSize/2), 0, Math.floor(worldsize/32))
  var camY = clamp(playery - (camSize/2), 0, Math.floor(worldsize/32))
  ctx.translate(-playerx * 32 + ((canvas.width / 2)), -playery * 32 + ((canvas.height / 2)));
  map[playerx][playery].player = true


  for (var x = camX; x < playerx + Math.floor(camSize/2); x++) {
    for (var y = camY; y < playery + Math.floor(camSize/2); y++) {
      if (x !== playerx || y !== playery) {
        map[x][y].player = false

        // console.log(map[x][y].city)
        if (map[x][y].landType === 0) {
          ctx.drawImage(oceanSpr, (x * 32), (y * 32), 32, 32)
          // ctx.fillStyle = "rgb(0,255,0)"
          // ctx.fillRect((x*32),(y*32),32,32)
        } else if (map[x][y].landType === 1) {
          ctx.drawImage(beachSpr, (x * 32), (y * 32), 32, 32)
          // ctx.fillStyle = "rgb(194,178,128)"
          // ctx.fillRect((x*32),(y*32),32,32)
        } else if (map[x][y].landType === 2) {
          ctx.drawImage(grassSprite, (x * 32), (y * 32), 32, 32)
          // ctx.fillStyle = "rgb(0,240,0)"
          // ctx.fillRect((x*32),(y*32),32,32)
        } else if (map[x][y].landType === 3) {
          ctx.drawImage(forestSprite, (x * 32), (y * 32), 32, 32)
          // ctx.fillStyle = "rgb(0,180,0)"
          // ctx.fillRect((x*32),(y*32),32,32)
        } else if (map[x][y].landType === 4) {
          ctx.drawImage(pineSpr, (x * 32), (y * 32), 32, 32)
          // ctx.fillStyle = "rgb(0,100,0)"
          // ctx.fillRect((x*32),(y*32),32,32)
        } else if (map[x][y].landType === 5) {
          ctx.drawImage(stoneSpr, (x * 32), (y * 32), 32, 32)
          // ctx.fillStyle = "rgb(120,120,120)"
          // ctx.fillRect((x*32),(y*32),32,32)
        } else if (map[x][y].landType === 6) {
          ctx.drawImage(iceSpr, (x * 32), (y * 32), 32, 32)
          // ctx.fillStyle = "rgb(240,240,255)"
          // ctx.fillRect((x*32),(y*32),32,32)
        }
        if (map[x][y].city === true) {
          console.log('ya')
          ctx.drawImage(villageSpr,x*32,y*32,32,32)
          ctx.fillStyle = "black"
          ctx.fillRect(x * 32, y * 32, 32, 32)
        }

        ctx.fillStyle = "black"

        ctx.fillRect(playerx * 32, playery * 32, 32, 32)


        for (var i = 0; i < monsters.length - 1; i++) {
          if (x === monsters[i].x && y === monsters[i].y) {
            ctx.fillStyle = "black"
            ctx.font = "32px Arial"
            if (monsters[i].name === "skeleton") {
              ctx.fillText("sk", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "slime") {
              ctx.fillText("sl", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "zombie") {
              ctx.fillText("zb", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "chicken") {
              ctx.fillText("ck", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "snek") {
              ctx.fillText("sn", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "ogre") {
              ctx.fillText("og", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "fire elemental") {
              ctx.fillText("fe", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "ice elemental") {
              ctx.fillText("ie", x * 32, (y * 32) + 32)
            }
            if (monsters[i].name === "leyoth") {
              ctx.fillText("ly", x * 32, (y * 32) + 32)
            }
          }
        }
      }
    }
  }
  // console.log(monsters.length)
  // window.requestAnimationFrame(agroAi)
}
window.setInterval(loop,50)
// }
// function draw() {

// }