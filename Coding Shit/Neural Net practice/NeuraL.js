// Import the necessary modules from Synaptic.js
const { Layer, Network } = require('synaptic');

// Define the sigmoid activation function
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Create the neural network architecture
const inputLayer = new Layer(2); // Two input neurons (wingspan and length)
const hiddenLayer = new Layer(3); // Three hidden neurons
const outputLayer = new Layer(3); // Three output neurons (small, medium, big)

// Connect the layers
inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

// Create the neural network
const myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer],
    output: outputLayer,
});

// Sample dataset (replace with your actual data)
const dataset = [
    { wingspan: 2.5, length: 0.8, size: 'small' },
    { wingspan: 3.0, length: 1.2, size: 'medium' },
    // ... add more samples ...
];

// Train the neural network
for (let epoch = 0; epoch < 10000; epoch++) {
    for (const sample of dataset) {
        const input = [sample.wingspan, sample.length];
        const target = [sample.size === 'small', sample.size === 'medium', sample.size === 'big'];

        myNetwork.activate(input);
        myNetwork.propagate(0.1, target);
    }
}

// Example usage
const newWingspan = 2.8; // Replace with actual wingspan
const newLength = 1.0; // Replace with actual length

const output = myNetwork.activate([newWingspan, newLength]);
const predictedSize = ['small', 'medium', 'big'][output.indexOf(Math.max(...output))];

console.log(`Predicted size: ${predictedSize}`);

