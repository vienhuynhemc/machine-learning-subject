package classifiers;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.RemovePercentage;

public class SVM {

	public SVM() {
	}

	public Instances loadCsv(String path) {
		CSVLoader csvLoader = new CSVLoader();
		try {
			csvLoader.setSource(new File(path));
			return csvLoader.getDataSet();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	public Instances removePercentage(Instances instances, int p) throws Exception {
		RemovePercentage rp = new RemovePercentage();
		rp.setInputFormat(instances);
		rp.setPercentage(p);
		Instances result = Filter.useFilter(instances, rp);
		return result;
	}

	public Instances finalizeInstances(Instances instances, String indices) throws Exception {
		if (instances.classAttribute().isNumeric()) {
			NumericToNominal numericToNominal = new NumericToNominal();
			numericToNominal.setInputFormat(instances);
			numericToNominal.setAttributeIndices(indices);
			instances = Filter.useFilter(instances, numericToNominal);
		}
		return instances;
	}

	public Instances addLabel(String nameCol, String nameAttr, String index, Instances data) throws Exception {
		Add add = new Add();
		add.setAttributeIndex(index);
		add.setNominalLabels(nameAttr);
		add.setAttributeName(nameCol);
		add.setInputFormat(data);
		return Filter.useFilter(data, add);
	}

	public Instances fillLabelTest(Instances data, SMO svm) throws Exception {
		for (Instance instance : data)
			instance.setClassValue(svm.classifyInstance(instance));

		return data;
	}

	public void save(Instances data, String path) throws IOException {
		CSVSaver saver = new CSVSaver();
		saver.setInstances(data);
		saver.setFile(new File(path));
		saver.writeBatch();
	}

	public void statistical(Instances instances, String path, String nameAtrri) throws IOException {
		Map<String, Integer> map = new LinkedHashMap<String, Integer>();
		String[] array = nameAtrri.split(",");
		for (String s : array) {
			map.put(s.trim(), 0);
		}
		System.out.println("Line TEST: " + instances.numInstances());
		System.out.println("-------------------------------------");
		for (Instance instance : instances) {
			String key = instance.stringValue(0);
			map.put(key, map.get(key) + 1);
		}

		for (Map.Entry<String, Integer> entry : map.entrySet())
			System.out.println("Label " + entry.getKey() + ": " + entry.getValue());

	}

	public void run() throws Exception {
		Instances dataTrain = loadCsv("data/train.csv");
		dataTrain.setClassIndex(0);
		// Delete 60% because the file is too large
		dataTrain = removePercentage(dataTrain, 60);
		// numeric -> nominal
		dataTrain = finalizeInstances(dataTrain, "first");
		// USE SVM
		SMO svm = new SMO();
		svm.buildClassifier(dataTrain);
		System.out.println(svm);

		// Test
		Instances dataTest = loadCsv("data/test.csv");
		// Because the test lacks a label column, we add a label
		String nameAttri = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9";
		dataTest = addLabel("label", nameAttri, "first", dataTest);
		dataTest.setClassIndex(0);
		// fill label test
		dataTest = fillLabelTest(dataTest, svm);

		// save and display result
		String pathResult = "data/result.csv";
		save(dataTest, pathResult);
		statistical(dataTest, pathResult, nameAttri);
	}

	public static void main(String[] args) throws Exception {
		new SVM().run();
	}

}
