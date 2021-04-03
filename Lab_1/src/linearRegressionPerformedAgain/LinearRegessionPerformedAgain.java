package linearRegressionPerformedAgain;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class LinearRegessionPerformedAgain {

	// Khai báo các thuộc tính
	// Dữ liệu được lưu vào map
	private Map<String, ArrayList<Double>> datasetTrain;
	private Map<String, ArrayList<Double>> datasetTest;
	// Biến + tham số W.
	private Map<String, Double> W;
	private double W0;
	// Hồi quy cho thuộc tính nào
	private int classIndex;
	// Hệ số học
	private double learningRate;
	// Kết quả cũ
	private double deviationOld;

	public LinearRegessionPerformedAgain(String sourceFile, String testFile, double learningRate) {
		this.learningRate = learningRate;
		// nạp dữ liệu
		loadFile(sourceFile, testFile);
	}

	public void run() {
		// Chạy kiếm ra model
		model();
	}

	private void model() {
		// Lấy tên thuộc tính Y
		String nameY = getNameClassIndex();
		// Lây danh sách giá trị thực của Y
		List<Double> valuesY = datasetTrain.get(nameY);
		for (int i = 0; i < 10000; i++) {
			// Tính độ sai lệch và in ra màn hình
			double deviationNew = deviation(nameY, valuesY);
			if (deviationNew == deviationOld) {
				break;
			} else {
				deviationOld = deviationNew;
			}
			// Tính lại các W
			descent(nameY, valuesY);
		}
		System.out.println(nameY + " = \n");
		for (Map.Entry<String, Double> entry : W.entrySet()) {
			if (!entry.getKey().equals(nameY)) {
				System.out.println("\t" + String.format("%,.4f", entry.getValue()) + " * " + entry.getKey() + " +");
			}
		}
		System.out.println("\t" + String.format("%,.4f", W0));
	}

	private void descent(String nameY, List<Double> valuesY) {
		// Tạo Map hỗ trợ
		Map<String, Double> helpMap = new LinkedHashMap<>();
		for (Map.Entry<String, Double> entry : W.entrySet()) {
			if (!entry.getKey().equals(nameY)) {
				helpMap.put(entry.getKey(), 0.0);
			}
		}
		// W0;
		double sumW0 = 0;
		for (int i = 0; i < valuesY.size(); i++) {
			double diff = f(i, nameY) - valuesY.get(i);
			sumW0 += diff;
			// Wt
			for (Map.Entry<String, Double> entry : helpMap.entrySet()) {
				entry.setValue(entry.getValue() + diff * W.get(entry.getKey()));
			}
		}
		// W0
		W0 -= (sumW0 / valuesY.size()) * learningRate;
		for (Map.Entry<String, Double> entry : helpMap.entrySet()) {
			W.put(entry.getKey(), W.get(entry.getKey()) - (entry.getValue() / valuesY.size()) * learningRate);
		}
	}

	private double deviation(String nameY, List<Double> valuesY) {
		double sum = 0;
		for (int i = 0; i < valuesY.size(); i++) {
			double diff = f(i, nameY) - valuesY.get(i);
			sum += diff * diff;
		}
		return sum / (2 * valuesY.size());
	}

	private double f(int index, String nameY) {
		double sum = W0;
		for (Map.Entry<String, Double> entry : W.entrySet()) {
			sum += entry.getValue() * datasetTrain.get(entry.getKey()).get(index);
		}
		return sum;
	}

	private String getNameClassIndex() {
		int count = 0;
		String result = null;
		for (Map.Entry<String, Double> entry : W.entrySet()) {
			if (count == classIndex) {
				result = entry.getKey();
				break;
			}
			count++;
		}
		return result;
	}

	private void loadFile(String sourceFile, String testFile) {
		// Khởi tạo 2 tập dữ liệu
		datasetTrain = new LinkedHashMap<>();
		datasetTest = new LinkedHashMap<>();
		// Khởi tạo biến + tham số W
		W = new LinkedHashMap<>();
		// Khởi tạo random để random các W0, W1, W2,... đầu tiên
		Random random = new Random();
		W0 = random.nextDouble();
		// Đọc file
		BufferedReader bufferedReader;
		// File train
		try {
			bufferedReader = new BufferedReader(new FileReader(new File(sourceFile)));
			// Lấy thuộc tính
			String line = bufferedReader.readLine();
			String[] arrayAttribute = line.split(",");
			for (String nameAttribue : arrayAttribute) {
				datasetTrain.put(nameAttribue.substring(1, nameAttribue.length() - 1), new ArrayList<>());
				// add vô W
				W.put(nameAttribue.substring(1, nameAttribue.length() - 1), random.nextDouble());
			}
			while (true) {
				line = bufferedReader.readLine();
				if (line == null) {
					break;
				}
				String[] arrayData = line.split(",");
				int count = 0;
				for (Map.Entry<String, ArrayList<Double>> entry : datasetTrain.entrySet()) {
					entry.getValue().add(Double.parseDouble(arrayData[count]));
					count++;
				}
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		// File test
		try {
			bufferedReader = new BufferedReader(new FileReader(new File(testFile)));
			// Lấy thuộc tính
			String line = bufferedReader.readLine();
			String[] arrayAttribute = line.split(",");
			for (String nameAttribue : arrayAttribute) {
				datasetTest.put(nameAttribue.substring(1, nameAttribue.length() - 1), new ArrayList<>());
			}
			while (true) {
				line = bufferedReader.readLine();
				if (line == null) {
					break;
				}
				String[] arrayData = line.split(",");
				int count = 0;
				for (Map.Entry<String, ArrayList<Double>> entry : datasetTest.entrySet()) {
					entry.getValue().add(Double.parseDouble(arrayData[count]));
					count++;
				}
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	// Hàm trả về số thuộc tính tối đa
	public int numAttributes() {
		return W.size();
	}

	public void setClassIndex(int classIndex) {
		this.classIndex = classIndex;
	}

}
