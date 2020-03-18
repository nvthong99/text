package weka.pro;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Debug;
import weka.core.Debug.Random;

public class SvmModel extends IOArff {
	SMO svm;
	
	public SvmModel() {
	}

	public SvmModel(String filename) throws Exception {
		super(filename);
	}
	
	public void buildSVM(String filename) throws Exception {
		setTrainset(filename);
		this.trainset.setClassIndex(this.trainset.numAttributes() - 1);
		
		this.svm = new SMO();
		//svm.setOptions(this.model_option);
		svm.buildClassifier(this.trainset);
		
	}
	
	public void evaluatetoSVM(String filename) throws Exception{
		setTestset(filename);
		this.testset.setClassIndex(this.testset.numAttributes() - 1);
		
		// danh gia mo hinh
		Random rnd = new Debug.Random();
		int folds = 10;
		Evaluation eval = new Evaluation(this.trainset);
		eval.crossValidateModel(svm, this.testset, folds, rnd);
		System.out.println(eval.toSummaryString("SVMmodel: \n",false));
		//System.out.println("True Negative:	" + eval.numTrueNegatives(0));
	}
}
