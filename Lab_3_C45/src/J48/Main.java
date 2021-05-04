package J48;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class Main {

	public static void main(String[] args) throws Exception {
		DataSource dataSource = new DataSource("data/bank-new.arff");
		Instances dataset = dataSource.getDataSet();
		dataset.setClassIndex(dataset.numAttributes() -2);
		J48 j48 = new J48();
		j48.buildClassifier(dataset);
		System.out.println(j48);
	}

}
