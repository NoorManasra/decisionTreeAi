package org.example.demo5;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Arrays;
import java.util.Random;
public class DecisionTreeAssignment {

    public static void main(String[] args) throws Exception {
        // Load the dataset from the specified file path
        String datasetFilePath = "C:\\Users\\twitter store\\Desktop\\processed_dataset2222222222.csv";
        DataSource dataSource = new DataSource(datasetFilePath); // Create a data source from the dataset path
        Instances dataset = dataSource.getDataSet(); // Load the dataset into an Instances object

        // Check if the class index is not set, and set it to the last attribute
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1); // Set the class index to the last attribute
        }

        // Print the distribution of the target class
        System.out.println("Distribution of the target class:");
        System.out.println(Arrays.toString(dataset.attributeStats(dataset.classIndex()).nominalCounts)); // Print the nominal counts of the target class

        // Randomize the dataset using a fixed seed for reproducibility
        dataset.randomize(new Random(1));

        // Calculate the number of instances for the training set (70% of the dataset)
        int trainingSetSize70 = (int) Math.round(dataset.numInstances() * 0.7);
        // Calculate the number of instances for the test set (remaining 30% of the dataset)
        int testSetSize30 = dataset.numInstances() - trainingSetSize70;
        // Create the training set for model 70-30 split
        Instances trainingData70 = new Instances(dataset, 0, trainingSetSize70);
        // Create the test set for model 70-30 split
        Instances testData30 = new Instances(dataset, trainingSetSize70, testSetSize30);

        // Print the size of the training and test sets for the 70-30 split
        System.out.println("\nAfter splitting the dataset into 70% training and 30% test for Model 1:");
        System.out.println("Training Set Size (70-30 Split): " + trainingSetSize70 + " instances");
        System.out.println("Test Set Size (70-30 Split): " + testSetSize30 + " instances");

        // Create a new J48 decisi on tree classifier for the 70-30 split model
        J48 decisionTree70 = new J48();
        decisionTree70.setUnpruned(false); // Enable pruning for the decision tree
        decisionTree70.setConfidenceFactor(0.25F); // Set the confidence factor for pruning
        decisionTree70.buildClassifier(trainingData70); // Train the decision tree on the training data for 70-30 split

        // Evaluate the trained model on the test data
        Evaluation evaluation70 = new Evaluation(trainingData70); // Initialize the evaluation object
        evaluation70.evaluateModel(decisionTree70, testData30); // Evaluate the model using the test data
        System.out.println("\nResults for Model 1 (70-30 Split):");
        System.out.println("Accuracy: " + evaluation70.pctCorrect()); // Print the accuracy of Model 1
        System.out.println("F1 Score: " + evaluation70.weightedFMeasure()); // Print the F1 score of Model 1

        // Print the generated decision tree for model 70-30 split
        System.out.println("\nDecision Tree for Model 1 (70-30 Split):");
        System.out.println(decisionTree70.toString());

        // Calculate the number of instances for the training set (50% of the dataset) for model 50-50 split
        int trainingSetSize50 = (int) Math.round(dataset.numInstances() * 0.5);
        // Calculate the number of instances for the test set (remaining 50% of the dataset) for model 50-50 split
        int testSetSize50 = dataset.numInstances() - trainingSetSize50;
        // Create the training set for model 50-50 split
        Instances trainingData50 = new Instances(dataset, 0, trainingSetSize50);
        // Create the test set for model 50-50 split
        Instances testData50 = new Instances(dataset, trainingSetSize50, testSetSize50);

        // Print the size of the training and test sets for the 50-50 split
        System.out.println("\nAfter splitting the dataset into 50% training and 50% test for Model 2:");
        System.out.println("Training Set Size (50-50 Split): " + trainingSetSize50 + " instances");
        System.out.println("Test Set Size (50-50 Split): " + testSetSize50 + " instances");

        // Create a new J48 decision tree classifier for the 50-50 split model
        J48 decisionTree50 = new J48();
        decisionTree50.setUnpruned(false); // Enable pruning for the decision tree
        decisionTree50.setConfidenceFactor(0.25F); // Set the confidence factor for pruning
        decisionTree50.buildClassifier(trainingData50); // Train the decision tree on the training data for 50-50 split

        // Evaluate the trained model on the test data
        Evaluation evaluation50 = new Evaluation(trainingData50); // Initialize the evaluation object
        evaluation50.evaluateModel(decisionTree50, testData50); // Evaluate the model using the test data
        System.out.println("\nResults for Model 2 (50-50 Split):");
        System.out.println("Accuracy: " + evaluation50.pctCorrect()); // Print the accuracy of Model 2
        System.out.println("F1 Score: " + evaluation50.weightedFMeasure()); // Print the F1 score of Model 2

        // Print the generated decision tree for model 50-50 split
        System.out.println("\nDecision Tree for Model 2 (50-50 Split):");
        System.out.println(decisionTree50.toString());

        // Print a comparison of the accuracy and F1 score between models 1 and 2
        System.out.println("\nComparison between Model 1 (70-30 Split) and Model 2 (50-50 Split):");
        System.out.println("Model 1 Accuracy: " + evaluation70.pctCorrect());
        System.out.println("Model 1 F1 Score: " + evaluation70.weightedFMeasure());
        System.out.println("Model 2 Accuracy: " + evaluation50.pctCorrect());
        System.out.println("Model 2 F1 Score: " + evaluation50.weightedFMeasure());
    }
}
