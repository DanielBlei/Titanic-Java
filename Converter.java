package titanic;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class Converter {

    public Converter(String location, String output){

            try {

                //Load a CSV file:
                CSVLoader loader = new CSVLoader();
                loader.setSource(new File(location));
                Instances data = loader.getDataSet();
                System.out.println("Data Converted\n");// loaded

                //Save as Arff
                ArffSaver saver = new ArffSaver();
                saver.setInstances(data); // data to convert
                saver.setFile(new File(output));
                saver.writeBatch();
            }
            catch (Exception e){
                System.out.println("Delete special characters from the CSV File");
            }
        }

}
