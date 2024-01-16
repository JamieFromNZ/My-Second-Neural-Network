let Network = require('./Network');
let babyNames = require('./names.json');

let numberOfInputs = 2;
let hiddenLayerNeurons = 2;
let outputLayerNeurons = 2;

let network = new Network(numberOfInputs, hiddenLayerNeurons, outputLayerNeurons, babyNames);

network.printNetworkDiagram();

network.input([0.05, 0.10], [0.01, 0.99]);

for (let i = 0; i < 100000; i++) {
    let int1 = Math.floor(Math.random() * 10) / 100;
    let int2 = Math.floor(Math.random() * 10) / 10;

    network.input([int1, int2], [0.01, 0.99])
}