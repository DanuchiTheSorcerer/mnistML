package com.mnist;

/**
 * Hello world!
 *
 */
public class DigitPredictor 
{
    public static void main( String[] args )
    {
        NeuralNetwork predictor = NeuralNetwork.newNeuralNetwork(new int[] {1,50, 1});
        System.out.println(predictor.feedForward(Vector.newVector(1, new float[] {1})).getComponent(0));
        System.out.println(" " + predictor.weights[0].getComponent(0, 0) + ", " + predictor.biases[0].getComponent(0));


        Vector[] inBatch = new Vector[1];
        Vector[] outBatch = new Vector[1];
        inBatch[0] = Vector.newVector(1, new float[] {1});
        outBatch[0] = Vector.newVector(1, new float[] {0.5f});
        for (int i = 0; i < 100000; i++) {
            predictor.Backpropagate(inBatch, outBatch, 0.01f);
        }
        System.out.println(predictor.feedForward(Vector.newVector(1, new float[] {1})).getComponent(0));
        System.out.println(" " + predictor.weights[0].getComponent(0, 0) + ", " + predictor.biases[0].getComponent(0));
    }
}
