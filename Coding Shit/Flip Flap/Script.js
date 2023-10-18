//setups the canvas and sizes it to the window
const canvas = document.getElementById("c")
const ctx = canvas.getContext("2d")
function canvasSetup() {
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight
}
canvasSetup()

//declares the player character object
let player = {
    y: canvas.height/2,
    x: Math.floor(canvas.width/20),
    vy: 0,
    alive: false,
    color: "rgb(224, 224, 49)",
    width: 60,
    height: 30,
    killed: false
}

//declares score
let score = 0
let highScore = 0
//handles high-score
if (localStorage.getItem("highScore") !== null) {
    highScore = localStorage.getItem("highScore")
    console.log(highScore)
} else {
    localStorage.setItem("highScore", 0)
    highScore = 0
}

//determines the speed of the pipes
let gameSpeed = 5

//declare framecount for adding pipes every so often
let frameCount = 0

//everything in game is contained in the scene object
let scene = {
    sky: {
        color: "rgb(49, 183, 224)"
    },
    player,
    ground: {
        x:0,
        y: canvas.height * (3/4),
        width: canvas.width,
        height: canvas.height/4,
        color: "rgb(100, 207, 43)"
    },
    pipes: {
        array: [
            {
                top: {
                    y: 0,
                    height: canvas.height/3,
                    x: canvas.width* 0.8,
                    width: 100,
                },
                bottom: {
                    y: canvas.height * (3/5),
                    height: canvas.height - (canvas.height * (3/5)),
                    x: canvas.width* 0.8,
                    width: 100,
                },
            }
        ],
        color: "rgb(12, 110, 51)",
    }
}

//physics
function phys() {
    if (player.vy <= 10) {
        player.vy += 0.5
    }
    player.y += player.vy
}

//move the pipes
function pipeMove() {
    for (let i = 0; i < scene.pipes.array.length; i++) {
        scene.pipes.array[i].top.x -= gameSpeed
        scene.pipes.array[i].bottom.x -= gameSpeed
    }
}

//collision handler
function collision() {
    if (collisonDet(player,scene.ground)) {
        player.alive = false
        player.killed = true
        player.color = "rgb(227, 64, 39)"
    }
    for (let i = 0; i < scene.pipes.array.length; i++) {
        if (collisonDet(player,scene.pipes.array[i].top)) {
            player.alive = false
            player.killed = true
        }
        if (collisonDet(player,scene.pipes.array[i].bottom)) {
            player.alive = false
            player.killed = true
            player.color = "rgb(227, 64, 39)"
        }
    }
}

//collison detection (can only handle rectangles for now need to improve in the future)
function collisonDet(rect1,rect2) {
    if (
        rect1.x < rect2.x + rect2.width &&
        rect1.x + rect1.width > rect2.x &&
        rect1.y < rect2.y + rect2.height &&
        rect1.height + rect1.y > rect2.y
      ) {
        return true
      } else {
        return false
      }
}

// add a pipe with random gap position, but make sure it's at least 750 pixels from the previous pipe
function addPipe() {
    const lastPipe = scene.pipes.array[scene.pipes.array.length - 1]
    const lastPipeX = lastPipe.top.x
    const pipeX = lastPipeX + 750
    const gapPos = Math.round((Math.random() * (canvas.height * 0.45)) - (canvas.height * 0.15)) + canvas.height / 8
    scene.pipes.array.push({
        top: {
            y: 0,
            height: gapPos,
            x: pipeX,
            width: 100
        },
        bottom: {
            y: gapPos + canvas.height * 0.25,
            height: canvas.height - (gapPos + canvas.height * 0.25),
            x: pipeX,
            width: 100
        },
    })
}


//remove any pipe that leaves the screen
function removePipe() {
    for (let i = 0; i < scene.pipes.array.length; i++) {
        if (scene.pipes.array[i].top.x <= -scene.pipes.array[i].top.width - 10) {
            scene.pipes.array.splice(i,1)
            addPipe()
        }
    }
}

//handles all pipe functions
function pipeLogic() {
    gameSpeed += 0.0025
    for (let i = 0; i < scene.pipes.array.length; i++) {
        if (player.x >= scene.pipes.array[i].top.x+scene.pipes.array[i].top.width && player.x <= scene.pipes.array[i].top.x+scene.pipes.array[i].top.width + gameSpeed) {
            score++
            if (score > highScore) {
                highScore = score
                localStorage.setItem("highScore", highScore)
            }
        }
    }
    pipeMove()
    removePipe()
}

//draws the scene to canvas
function draw() {
    ctx.clearRect(0,0,canvas.width,canvas.height)

    ctx.fillStyle = "rgb(49, 183, 224)"
    ctx.fillRect(0,0,canvas.width,canvas.height)

    ctx.fillStyle = player.color
    ctx.fillRect(player.x,player.y,player.width,player.height)

    ctx.fillStyle = scene.pipes.color
    for (let i = 0; i < scene.pipes.array.length; i++) {
        ctx.fillRect(scene.pipes.array[i].top.x,scene.pipes.array[i].top.y,scene.pipes.array[i].top.width,scene.pipes.array[i].top.height)
        ctx.fillRect(scene.pipes.array[i].bottom.x,scene.pipes.array[i].bottom.y,scene.pipes.array[i].bottom.width,scene.pipes.array[i].bottom.height)
    }

    ctx.fillStyle = scene.ground.color
    ctx.fillRect(scene.ground.x,scene.ground.y,scene.ground.width,scene.ground.height)

    ctx.fillStyle = "black"
    ctx.font = "80px typoRound"
    ctx.fillText(score,canvas.width/2.5,canvas.height*(7/8))

    if (!player.alive && player.killed) {
        ctx.fillStyle = "rgb(49, 183, 224)"
        ctx.fillRect(0,0,canvas.width,canvas.height)
        
        ctx.fillStyle = "black"
        ctx.font = "80px typoRound"
        ctx.fillText("Score: "+score.toString," Highscore: "+highScore.toString,canvas.width/2.5,canvas.height*(7/8))
    }
}
draw()

//handles the game loop
function loop() {
    frameCount++
    phys()
    pipeLogic()
    collision()
    draw()
}

//control handler
document.onkeydown = function(e) {
    if (player.alive) {
        if (e.key.charCodeAt(0) === 32) {
            // alert("space")
            player.vy = -10
        }
    } else if(!player.killed) {
        player.alive = true
        player.vy = -10
    }

    if (player.killed && !player.alive) {
        if (e.key.charCodeAt(0) === 114) {
            console.log("reset")
            location.reload()
        }
    }
}

//handles the framerate
const interval = setInterval(function() {
    if (player.alive) {
        loop()
    }
  }, 15);

addPipe()
addPipe()
addPipe()