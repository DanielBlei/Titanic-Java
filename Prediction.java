package titanic;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToBinary;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;

public class Prediction {

    public static void main(String[] args) throws Exception {

        //Loading the test set

        String test_ARFF = "/home/danielblei/Weka/files/test_edited.arff";

        ArffLoader testLoader = new ArffLoader();
        testLoader.setSource(new File(test_ARFF));
        Instances test_Set = testLoader.getDataSet();

        //Creating the Target attribute to be predicted
        Attribute newAttribute = new Attribute("Target");

        //Setting the position of the new target as the train set
        test_Set.insertAttributeAt(newAttribute,0);

        //Setting index
        test_Set.setClassIndex(test_Set.numAttributes()-1);

        //Converting attributes to binary
        String[] binary = new String[]{"-R","2,6"};
        NumericToBinary convertSurvived = new NumericToBinary();
        convertSurvived.setOptions(binary);
        convertSurvived.setInputFormat(test_Set);
        test_Set = Filter.useFilter(test_Set,convertSurvived);

        //Loading the trained model
        Classifier model = (Classifier) SerializationHelper.read("titanic.model");

        //Creating the Prediction and saving it in an Array List
        Instances Predict = new Instances(test_Set);

        ArrayList<Double> Predicted = new ArrayList<>();

        for( int i=0; i < test_Set.numInstances(); i++){

            double Survive = model.classifyInstance(test_Set.instance(i));
            Predict.instance(i).setClassValue(Survive);
            Predicted.add(Survive);
            ;
        }
        //Wrinting the Array List as a unique CSV Column
        BufferedWriter br = new BufferedWriter(new FileWriter("Submission.csv"));
        StringBuilder sb = new StringBuilder();

        // Append the values from array
        for (Double element : Predicted) {
            sb.append(element);
            sb.append(",\n");
        }

        br.write(sb.toString());
        br.close();

    }
}