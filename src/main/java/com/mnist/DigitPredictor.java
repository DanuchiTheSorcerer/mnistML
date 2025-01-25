package com.mnist;


import com.trainsetLoader.Mnist;
import com.trainset.TrainSet;
/**
 * Hello world!
 *
 */
public class DigitPredictor 
{
    public static void main( String[] args )
    {
        NeuralNetwork predictor = NeuralNetwork.newNeuralNetwork(new int[] {784,70,35, 10});
        TrainSet set = Mnist.createTrainSet(0, 4999);
        TrainSet testSet = Mnist.createTrainSet(5000, 9999);

        Vector[] inputs = makeInputs(set);
        Vector[] outputs = makeOutputs(set);

        Vector[] testInputs = makeInputs(testSet);
        Vector[] testOutputs = makeOutputs(testSet);


        long score = 0;
        for (int i = 0; i < inputs.length;i++) {
            Vector networkOutput = predictor.feedForward(inputs[i]).softmax();
            int highestIndex = 0;
            float highestValue = networkOutput.getComponent(0);
            for (int j = 1;j < networkOutput.getSize();j++) {
                if (networkOutput.getComponent(j) > highestValue) {
                    highestIndex = j;
                }
                highestValue = Math.max(highestValue,networkOutput.getComponent(j));
                if (highestValue == networkOutput.getComponent(j)) {
                    highestIndex = j;
                }
            }
            if (outputs[i].getComponent(highestIndex) == 1) {
                score++;
            }
  
        }
        System.out.println("Score: " + score + "/" + set.size());
        long startTime = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            predictor.backpropagate(inputs, outputs, 0.3f);
            if (i % 100 == 0) {
                System.out.println("Epoch: " + i);
                System.out.println("Time Elapsed: " + (System.nanoTime() - startTime)/ 1_000_000_000 + " sec");
            }
        }
        long score2 = 0;
        for (int j = 0; j < testInputs.length;j++) {
            Vector networkOutput = predictor.feedForward(testInputs[j]).softmax();
            int highestIndex = 0;
            float highestValue = networkOutput.getComponent(0);
            for (int k = 1;k < networkOutput.getSize();k++) {
                if (networkOutput.getComponent(k) > highestValue) {
                    highestIndex = k;
                }
                highestValue = Math.max(highestValue,networkOutput.getComponent(k));
            }
            if (testOutputs[j].getComponent(highestIndex) == 1) {
                score2++;
            }
            
        }
        System.out.println("Score: " + score2 + "/" + set.size());
    }
    public static Vector[] makeInputs(TrainSet set) {
        Vector[] inputs = new Vector[set.size()];
        for (int i = 0; i < set.size(); i++) {
            double[] inputDoubles = set.getInput(i);
            float[] inputFloats = new float[inputDoubles.length];
            for (int j = 0; j < inputDoubles.length; j++) {
                inputFloats[j] = (float) inputDoubles[j];
            }
            inputs[i] = LinearAlgebra.newVector(set.size(), inputFloats);
        }
        return inputs;
    }
    public static Vector[] makeOutputs(TrainSet set) {
        Vector[] outputs = new Vector[set.size()];
        for (int i = 0; i < set.size(); i++) {
            double[] outputDoubles = set.getOutput(i);
            float[] outputFloats = new float[outputDoubles.length];
            for (int j = 0; j < outputDoubles.length; j++) {
                outputFloats[j] = (float) outputDoubles[j];
            }
            outputs[i] = LinearAlgebra.newVector(set.size(), outputFloats);
        }
        return outputs;
    }
}
