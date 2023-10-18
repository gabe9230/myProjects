const canvas = document.getElementById("c")
const ctx = canvas.getContext("2d")

let player = {
    money: 0,
    cores: 8,
    property: {
        warehouses: {},
        factories: {},
        extractors: {},
        bots: {
            utility: {
                harvesters: 0,
                transporters: 0,
                maintenance: 0
            },
            military: {
                line: 0,
                tank: 0,
                cannon: 0
            }
        }
    },
    upgrades: {
        personal: {
            hackingSpeed: 1,
            hackingStealth: 1
        },
        property: {
            warehouseSize: 1,
            factoryEff: 0.3,
            factorySpeed: 1,
            factoryScale: 1,
            extractorEff: 0.1,
            extractorSpeed: 1,
            extractorScale: 1
        },
        bots: {
            utility: {
                harvester: {
                    reliability: 5
                },
                transporter: {
                    reliability: 5,
                    storage: 50,
                    speed: 60
                },
                maintenance: {
                    reliability: 5,
                    repairSpeed: 3,
                }
            },
            military: {
                line: {
                    health:3,
                    damage:3,
                    armor: 0,
                    range: 1
                },
                tank: {
                    health: 20,
                    damage: 5,
                    armor: 3,
                    range: 1
                },
                cannon: {
                    health: 5,
                    damage: 20,
                    armor: 0,
                    range: 3
                }
            }
        }
    },
    resources: {
        steel: 0,
        fuel: 0,
        sensorComponents: 0,
        cables: 0,
        weapons: 0,
        titanium: 0,
        hardenedArmor: 0,
        copper: 0,
        batterys: 0,
        lithium: 0,
        circuitBoards: 0,
        oil: 0,
        silicon: 0,
        explosives: 0,
        ammo: 0,
        spareParts: 0
    }
}

let turn = 0

let input

let hackTargets = [
    {
        name: "Tor Bank",
        balance: 1382432843,
        diff: 65
    },
    {
        name: "Robert's Cakes",
        balance: 3112,
        diff: 1
    },
    {
        name: "Harriet Insurance Group",
        balance: 2328425,
        diff: 20
    },
    {
        name: "Quantum Financials",
        balance: 975364892,
        diff: 80
    },
    {
        name: "Cyberlux Electric",
        balance: 17643298,
        diff: 30
    },
    {
        name: "Alpha Crypto Exchange",
        balance: 458374652,
        diff: 70
    },
    {
        name: "Zeus Energy",
        balance: 33849622,
        diff: 40
    },
    {
        name: "Vertex Pharma",
        balance: 125673489,
        diff: 50
    },
    {
        name: "Nova Defense Contractors",
        balance: 387654213,
        diff: 75
    },
    {
        name: "Seraph Healthcare",
        balance: 64739214,
        diff: 35
    },
    {
        name: "Omega Mega Corp",
        balance: 2154863927,
        diff: 90
    },
    {
        name: "Lunar Travel Agency",
        balance: 8734961,
        diff: 15
    },
    {
        name: "Solaris Studios",
        balance: 34687645,
        diff: 25
    },
    {
        name: "Pixie Bookstore",
        balance: 5472,
        diff: 10
    },
    {
        name: "Bumblebee Florist",
        balance: 1328,
        diff: 5
    },
    {
        name: "Raindrop Laundromat",
        balance: 2891,
        diff: 15
    }
]

const mainMenutxt = "1: Hack \n2: Manage Property \n3: Manage Bots \n4: Manage Upgrades \n5: View Stockpiles"

function print(thing) {
    console.log(thing)
}


function startup() {
    print(mainMenutxt)
    input = prompt("Select Desired Action: ")
    if (input === "1") {
        hack()
    }
}

function hack() {
    print("!")
    let targets = []
    for (let i = 0; i < 3; i++) {
        targets.push(hackTargets[Math.round(Math.random()*(hackTargets.length-1))])
    }
    print("Available targets:"+
    "\n1: \nName: "+String(targets[0].name)+"\nBalance: "+String(targets[0].balance)+"\nDifficulty: "+String(targets[0].name)+
    "\n2: \nName: "+String(targets[1].name)+"\nBalance: "+String(targets[1].balance)+"\nDifficulty: "+String(targets[1].name)+
    "\n3: \nName: "+String(targets[2].name)+"\nBalance: "+String(targets[2].balance)+"\nDifficulty: "+String(targets[2].name)
    )
    prompt("Please select a target to attempt to hack: ")
}

function manageProperty() {

}

function manageBots() {

}

function manageUpgrades() {

}

function stockpileView() {

}