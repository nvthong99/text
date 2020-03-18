package weka.pro;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Debug;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveBayesModel extends IOArff {
	
	NaiveBayes bayes;
	
	public NaiveBayesModel() {
		super();
	}
	public NaiveBayesModel(String filename) throws Exception {
		super(filename);
	}
	
	public void buildNaiveBayesModel(String filename) throws Exception {
		setTrainset(filename);
		this.trainset.setClassIndex(this.trainset.numAttributes() - 1);
		//huan luyen
		this.bayes = new NaiveBayes();
		bayes.buildClassifier(this.trainset);
	}
	
	public void evaluatetoNaivebayes(String filename) throws Exception{
		setTestset(filename);
		this.testset.setClassIndex(this.testset.numAttributes() - 1);
		
		// danh gia mo hinh
		Random rnd = new Debug.Random();
		int folds = 10;
		Evaluation eval = new Evaluation(this.trainset);
		eval.crossValidateModel(bayes, this.testset, folds, rnd);
		System.out.println(eval.toSummaryString("NaiveBayesMolef \n", false));
	//	System.out.println("True Negative::::" + eval.numTrueNegatives(0));
		
	}
	
	public void predictClassLabel(String FileIn, String fileOut) throws Exception{
		DataSource ds = new DataSource(FileIn);
		Instances unlable = ds.getDataSet();
		unlable.setClassIndex(unlable.numAttributes() - 1);
		// du doan ket qua 
		for ( int i = 0; i < unlable.numInstances(); i++) {
			double predict = bayes.classifyInstance(unlable.instance(i));
			unlable.instance(i).setClassValue(predict);
		}
		
		//xuat ket qua
		
		BufferedWriter outWrite = new BufferedWriter(new FileWriter(fileOut));
		outWrite.write(unlable.toString());
		outWrite.newLine();
		outWrite.flush();
		outWrite.close();
	}
}

