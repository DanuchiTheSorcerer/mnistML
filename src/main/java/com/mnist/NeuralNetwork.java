package com.mnist;

import java.util.Random;

public class NeuralNetwork {
    int[] layerSizes;
    Matrix[] weights;
    Vector[] biases;
    public static void main(String[] args) {
        
    }
    public Vector feedForward(Vector input) {
        Vector a = input;
        for (int i = 0; i < layerSizes.length - 1; i++) {
            Vector z = weights[i].transform(a).add(biases[i]);
            a = LinearAlgebra.activationFunction(z);
        }
        return a;
    }
    public static NeuralNetwork newNeuralNetwork(int[] layerSizes) {
        NeuralNetwork nn = new NeuralNetwork();
        nn.layerSizes = layerSizes;
        nn.weights = new Matrix[layerSizes.length - 1];
        nn.biases = new Vector[layerSizes.length - 1];
        Random rand = new Random();
        
        for (int i = 0; i < layerSizes.length - 1; i++) {
            int fan_in = layerSizes[i];
            float stddev = (float) Math.sqrt(2.0 / fan_in);
            float[][] weightData = new float[layerSizes[i + 1]][layerSizes[i]];
            
            for (int j = 0; j < layerSizes[i + 1]; j++) {
                for (int k = 0; k < layerSizes[i]; k++) {
                    weightData[j][k] = (float) rand.nextGaussian() * stddev;
                }
            }
            
            nn.weights[i] = Matrix.newMatrix(layerSizes[i + 1], layerSizes[i], weightData);
            nn.biases[i] = Vector.newVector(layerSizes[i + 1], new float[layerSizes[i + 1]]); // Biases initialized to zero
        }
        return nn;
    }
    public void Backpropagate(Vector[] inBatch, Vector[] outBatch, float learningRate) {
        int batchSize = inBatch.length;
        Matrix[] weightGradientsTotal = new Matrix[layerSizes.length];
        Vector[] biasGradientsTotal = new Vector[layerSizes.length];
        for (int i = 1; i < layerSizes.length; i++) {
            weightGradientsTotal[i] = Matrix.newMatrix(layerSizes[i], layerSizes[i-1], new float[layerSizes[i]][layerSizes[i-1]]);
            biasGradientsTotal[i] = Vector.newVector(layerSizes[i], new float[layerSizes[i]]);
        }
        for (int i = 0; i < batchSize; i++) {
            Matrix[] weightGradients = new Matrix[layerSizes.length];
            Vector[] biasGradients = new Vector[layerSizes.length];
            for (int j = 1; j < layerSizes.length; j++) {
                weightGradients[j] = Matrix.newMatrix(layerSizes[j], layerSizes[j-1], new float[layerSizes[j]][layerSizes[j-1]]);
                biasGradients[j] = Vector.newVector(layerSizes[j], new float[layerSizes[j]]);
            }
            Vector[] activations = new Vector[layerSizes.length ];
            Vector[] zActivations = new Vector[layerSizes.length];
            for (int j = 0; j < layerSizes.length; j++) {
                if (j == 0) {
                    activations[0] = inBatch[i];
                } else {
                    zActivations[j] = weights[j-1].transform(activations[j-1]).add(biases[j-1]);
                    activations[j] = LinearAlgebra.activationFunction(zActivations[j]);
                }
            }
            Vector deltaA = LinearAlgebra.newVector(layerSizes[layerSizes.length-1], new float[layerSizes[layerSizes.length-1]]);
            for (int j = 0; j < layerSizes[layerSizes.length-1]; j++) {
                deltaA.set(j, activations[activations.length-1].getComponent(j) - outBatch[i].getComponent(j));
            } 
            for (int l = layerSizes.length -1;l>0;l--) {
                biasGradients[l] = deltaA.scale(learningRate).hadamard(LinearAlgebra.activationFunctionPrime(zActivations[l]));
                Vector[] vecArray = new Vector[weightGradients[l].getCols()];
                for (int j = 0; j < weightGradients[l].getCols(); j++) {
                    vecArray[j] = biasGradients[l].scale(activations[l-1].getComponent(j));
                }
                weightGradients[l] = LinearAlgebra.composeMatrix(vecArray);
                deltaA = weights[l-1].transpose().transform(biasGradients[l]);
            }
            for (int j = 1; j < layerSizes.length; j++) {
                weightGradientsTotal[j] = weightGradientsTotal[j].add(weightGradients[j]);
                biasGradientsTotal[j] = biasGradientsTotal[j].add(biasGradients[j]);
            }
        }
        for (int i = 1; i < layerSizes.length; i++) {
            weights[i-1] = weights[i-1].subtract(weightGradientsTotal[i].scale(1.0f / batchSize));
            biases[i-1] = biases[i-1].subtract(biasGradientsTotal[i].scale(1.0f / batchSize));
        }
    }
}
