package com.trainsetLoader;

import com.trainset.TrainSet;

import java.io.File;

/**
 * Created by Luecx on 10.08.2017.
 */
public class Mnist {
    public static void main(String[] args) {
        TrainSet set = createTrainSet(0,4999);
        // for (int i = 0; i < 28*28; i++) {
        //     System.out.print(set.getInput(17)[i] + " ");
        //     if(i % 28 == 0) System.out.println();
        // }
        // System.out.println();
        // for (int i = 0; i < 10; i++) {
        //     System.out.print(set.getOutput(17)[i] + " ");
        // }

        TrainSet testSet = createTrainSet(5000,9999);
    }

    public static TrainSet createTrainSet(int start, int end) {

        TrainSet set = new TrainSet(28 * 28, 10);

        try {

            String path = new File("").getAbsolutePath();

            MnistImageFile m = new MnistImageFile(path + "/res/trainImage.idx3-ubyte", "rw");
            MnistLabelFile l = new MnistLabelFile(path + "/res/trainLabel.idx1-ubyte", "rw");

            for(int i = start; i <= end; i++) {
                if(i % 100 ==  0){
                    System.out.println("prepared: " + i);
                }

                double[] input = new double[28 * 28];
                double[] output = new double[10];

                output[l.readLabel()] = 1d;
                for(int j = 0; j < 28*28; j++){
                    input[j] = (double)m.read() / (double)256;
                }

                set.addData(input, output);
                m.next();
                l.next();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

         return set;
    }
}
