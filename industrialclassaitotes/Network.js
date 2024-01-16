let Neuron = require('./Neuron');

class Network {
    constructor(numberOfInputs, hiddenLayerNeurons, outputLayerNeurons, babyNames) {
        this.hiddenLayer = [];
        this.outputLayer = [];

        this.numberOfInputs = numberOfInputs;

        this.learningRate = 0.01;

        // maybe make get baby names function? oop
        for (let i = 0; i < hiddenLayerNeurons; i++) this.hiddenLayer.push(new Neuron(this, numberOfInputs, babyNames.names[Math.floor(Math.random() * babyNames.names.length)], 'hidden'));
        for (let i = 0; i < outputLayerNeurons; i++) this.outputLayer.push(new Neuron(this, hiddenLayerNeurons, babyNames.names[Math.floor(Math.random() * babyNames.names.length)], 'output'));

        // add diagram here
    }

    input(inputs, target) {
        // don't have an input layer cuz don't need it so going straight to hidden
        let hiddenLayerResults = this.hiddenLayer.map(neuron => neuron.forwardSend(inputs));

        // process output layer
        let outputLayerResults = this.outputLayer.map(neuron => neuron.forwardSend(hiddenLayerResults));

        // if it's a training input
        if (target) {
            this.backPropagateEachNeuron(outputLayerResults, target);
        }

        console.log(outputLayerResults);
    }

    // makes the weights better, back propagation
    backPropagateEachNeuron(results, targets) {
        // start by calculating total error as that's something not independent of each neuron I suppose
        let totalError = 0;

        for (let i = 0; i < results.length; i++) {
            let err = 0.5 * (Math.pow(targets[i] - results[i], 2));
            totalError += err;
        }

        // now it's fixin time
        for (let i = 0; i < this.outputLayer.length; i++) {
            this.outputLayer[i].backPropagate(totalError, this.hiddenLayer, results, targets);
        }

        console.log(results, ' results');
    }

    calcError(result, target) {
        return 0.5 * (target - result) ** 2;
    }

    getWeightedSum(array) {
        let weightedSum = 0;
        for (let i = 0; i < array.length; i++) {
            weightedSum += array[i];
        }
        return weightedSum
    }

    printNetworkDiagram() {
        console.log("Neural Network Diagram:");

        // Print input layer
        console.log("Input Layer:");
        for (let i = 0; i < this.numberOfInputs; i++) {
            console.log(`  Input Node ${i + 1}`);
        }

        // Print hidden layer
        console.log("Hidden Layer:");
        this.hiddenLayer.forEach(neuron => {
            console.log(`  Neuron ${neuron.name}: Weights - [${neuron.weights.join(", ")}], Bias - ${neuron.bias}`);
        });

        // Print output layer
        console.log("Output Layer:");
        this.outputLayer.forEach(neuron => {
            console.log(`  Neuron ${neuron.name}: Weights - [${neuron.weights.join(", ")}], Bias - ${neuron.bias}`);
        });
    }
}

module.exports = Network;