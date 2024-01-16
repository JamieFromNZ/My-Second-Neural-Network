class Neuron {
    constructor(network, numberOfInputs, name, layer) {
        // one weight per connection with prev layer
        this.weights = Array.from({ length: numberOfInputs }, () => Math.random());

        this.bias = Math.random();

        this.learningRate = network.learningRate;

        this.output = null;
        this.name = name;
        this.layer = layer;
    }

    forwardSend(inputs) {
        console.log(`\n⏩ Forward sending through neuron ${this.name} from ${this.layer} layer`);
        console.log(`⏩ His inputs: ${inputs}`);
        // AKA: total net input
        let weightedSum = inputs.reduce((sum, input, index) => sum + input * this.weights[index], 0);
        weightedSum += this.bias;

        // squash using logistic function
        let squashedWeightedSum = this.squashFunction(weightedSum);

        this.output = squashedWeightedSum; // saving it 4 future use

        return squashedWeightedSum;
    }

    /*
    activationFunction(x) {
        return 1 / (1 + Math.exp(-x));
    }
    */

    // logistic function is what it's known as I believe
    squashFunction(weightedSum) {
        return 1 / (1 + (Math.pow(2.7182818284590452353602874713526624977572, -1 * weightedSum)));
    }

    /*
    activationFunctionDerivative(x) {
        const fx = this.activationFunction(x);
        return fx * (1 - fx);
    }
    */

    backPropagate(totalError, prevLayer, results, targets) {
        console.log(`\nBack propagating through ol ${this.name}.`);
        console.log(`⏪ Current weights: \n🧍 ${this.weights}`);

        // looping through all weights
        for (let i = 0; i < this.weights.length; i++) {
            // delta rule
            let changeAmount = -(targets[i] - results[i]) * results[i] * (1 - results[i]) * prevLayer[i].output;

            // updating the weight
            this.weights[i] -= this.learningRate * changeAmount;

            let biasDelta = -(targets[i] - results[i]) * results[i] * (1 - results[i]);
            this.bias -= this.learningRate * biasDelta;
        }

        // as an example, let's change w5
        /*
            equation goes:
            pd of total error respect to squashed output
            pd of squashed output respect to weighted sum
            pd of weighted sum to w5
        */

        // let's first find pd of total e respect to squashed output
        // if we remember the equation for total error (remember, just liek chain rule, derive b4 you substitute (but we already substitued so we gotta back track a bit)) IMPORTANT
        // 0.5*(target-output_1)^2 + 0.5*(target-output_2)^2
        // and since we doing partial d we make the output_2 0 IMPORTANT
        //let partialDerivativeOfTotalErrorWithRespectToSquashedOutput = 2 * 0.5 (target-output_1)^2-1 * -1 + 0;
        // = -(target - output_1)

        // let partialDerivativeOfTotalErrorWithRespectToSquashedOutput = -(target_one_connected_to_w5 - output_one_connected_to_w5)

        // next up! how does the squashed output of o_1 chaneg with respect to its total weighted sum?
        // the pd of the logistic (squashifier) function is the output times by 1 minus the output so (I DUNNO WHY)
        // output_1 = 1/1+e^-weightedsum
        // pd squashed outpud/pd wieghted sum = output_1*(1-output_1)
        // = whatever

        // finally, how much does the total weighted sum of o_1 change with respect to w_5
        // if you remember how to find weighted sum, I can't be bothered typing it out
        // actually, I will
        // weighted_sum = w_5 * output_h1 + w_6 * output_h1 + b_2
        // pd of it with respect to a weight is:
        // 1*output_h1 * w_5(1-1) + 0
        // and by some miracle, that simplifies to output_h1, neat

        // now, you multiply it all together and you get a number
        // oh and the final equation is:
        // -(target_o1 - output_o1) * output_o1(1-output_o1) * output_h1
        // this is known as the delta rule

        // that number you get you subtract from your current weight (times by learning rate)


        // now that's for the hidden layer output layer weights
        // now for the input layer hidden layer weights

        // we do a similar thing but we shift everything across:

        /*
            equation goes:
            pd of total error with respect to  your weight is equal to =
            pd of total error over squashed output of hidden layer neuron it's connected to
            times
            pd of this squashed output over the net weighted sum of this connected neuron
            times
            pd this net weighted sum with respect to the weight

            and the total error over the squashed output of the hidden layer can be found by:
            pd of error of one output respect to hidden layer neuron squashed plus that of other output

            QUOTEFROM TUTORIAL:
            We’re going to use a similar process as we did for the output layer, but slightly different to account for the fact that the output of each hidden layer neuron contributes to the output (and therefore error) of multiple output neurons. We know that out_{h1} affects both out_{o1} and out_{o2} therefore the \frac{\partial E_{total}}{\partial out_{h1}} needs to take into consideration its effect on the both output neurons:
        */

        // but you might be wondering how we get the pd of single error of one output with respect to the squashed output of the hidden layer neuron which is what we need to find the pd of total error with respect to  it

        /*
            pd E_o1 respect to output_h1 = 
            pd E_o1 respect to net weight_o1
            times
            pd net weight_o1 respect to squashed output_h1

            cuz the net weights  of o1 cancel, it's the common var
            and using this we can calculate pd E_o1 respect to net_o1


            and as it happens (with some cool cancelling and pd stuff)
            pd of net weight_o1 respect to output_h1 is the weight between h1 and o1

            and you can DO SAME PROCESS FOR OTHER ERROR OUTPUT RESPECT TO SAME HIDDEN NEURON TO ADD TOGETHER


            okay, now we have pd of error respect to output_h1, we need to find pd of output_h1 resect to net weight_h1 and then pd net weight respesct to the weight


            now to find pd out_h1 respect net weight_h1
            if you recall the logistic function, out_h1 = 1/1+e^-net_h1

            thus the derivative is out_h1(1-out_h1)


            and finally, net weight respect to w

            net_h1 = w*i_1 + w_3 * i_2 + b_1 * 1

            pd net respect w = i_1 = 0.05


            and then you multiply them all together ofc     
        */

        /*
        OLD CODE
        // don't understand this tbh, or at least the relevance
        const derivative = this.activationFunctionDerivative(this.output); // in future, change activation function cuz we don't want 0  or 1

        
            C = cost function
            y = neuron outut
            z = weighted sum input to neuron

            gradientCostFunction = pdC/pdz 
            error = pdC/pdy
            derivative = pdy/pdz
        
        //  how much the cost function changes with a small change in the weighted sum 
        const gradientCostFunction = error * derivative;

        for (let i = 0; i < this.weights.length; i++) {
            console.log(`⏪ Back propagating through ${this.name}'s (from the ${this.layer} layer) and ${prevLayer[i].name}'s connection (a lad from the ${prevLayer[i].layer} layer).`);
            // delWta is how far it should move
            // prevLayer[i] is the neuron connected sharing the weight
            // prevLayer[i].output is pdz/pdw 
            let delta = gradientCostFunction * prevLayer[i].output; // prevLayer[i].output needs to be stored during forward pass

            // Update weight
            this.weights[i] -= this.learningRate * delta;
        }

        // Update bias
        this.bias -= this.learningRate * gradientCostFunction;
        */

        console.log(`⏪ New weights: \n🧍 ${this.weights}`);
        console.log(`Done back propagating through ${this.name}\n`);
    }

    calcGradient() {

    }
}

module.exports = Neuron;