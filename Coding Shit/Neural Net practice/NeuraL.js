
let pool = []
let net = []
let poolsize = 10
let nWidth = 8
let nHeight = 4
let nInputs = 10
let nOutputs = 4

function poolPopulate() {
    for (let i = 0; i < poolsize; i++) {
        net = []
        var inputCol = []
        for (let i = 0; i < nInputs; i++) {
            inputCol.push(0)
        }
        net.push(inputCol)
        var midlayer = []
        for (let x = 0; x < nWidth; x++) {
            midlayer.push([])
            for (let y = 0; y <= nHeight; y++) {
            midlayer[x].push(Math.random())  
            } 
        }
        net.push(midlayer)
        var outputCol = []
        for (let i = 0; i < nOutputs; i++) {
            outputCol.push(0)
        }
        net.push(outputCol)
        pool.push(net)
    }
}

poolPopulate()