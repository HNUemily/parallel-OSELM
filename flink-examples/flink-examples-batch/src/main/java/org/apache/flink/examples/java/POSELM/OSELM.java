package org.apache.flink.examples.java.POSELM;


import com.esotericsoftware.kryo.io.Output;
import no.uib.cipr.matrix.*;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Matrix;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class OSELM {


	public static void main(String[] args) throws Exception {

		final ParameterTool params = ParameterTool.fromArgs(args);
		// set up the execution environment
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

		// make parameters available in the web interface
		env.getConfig().setGlobalJobParameters(params);

		//oselm结果最终的
		double TrainingTime = 0.0;
		double TestingTime = 0.0;
		double TrainingAccuracy = 0.0;
		double TestingAccuracy = 0.0;

		int OSElm_Type = params.getInt("Elm_Type", 1);
		int NumberofHiddenNeurons = params.getInt("NumberofHiddenNeurons", 20);
		int NumberofOutputNeurons =params.getInt("NumberofOutputNeurons", 2);                        //also the number of classes
		int NumberofInputNeurons =params.getInt("NumberofInputNeurons", 8);                         //also the number of attribution
		int BlockSize = params.getInt("BlockSize", 200);
		String activefunc = params.get("activefunc", "sin");
		int[] label;

		//生成A和B计算隐藏层输出矩阵
		DenseMatrix Aweight = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, NumberofInputNeurons);
		DenseMatrix Bweight = (DenseMatrix) Matrices.random(NumberofHiddenNeurons, 1);

		// get input data
		DataSet<Tuple4<Integer, Integer, Double[], Double[]>> trainingdata;
		DataSet<Tuple4<Integer, Integer, Double[], Double[]>> testingdata;

		//将变量转为dataset
		DataSet<Integer> BOSELM_TYPE = env.fromElements(OSElm_Type);
		DataSet<Integer> BNumberofHiddenNeurons = env.fromElements(NumberofHiddenNeurons);
		DataSet<Integer> BNumberofInputNeurons = env.fromElements(NumberofInputNeurons);
		DataSet<Integer> BNumberofOutputNeurons = env.fromElements(NumberofOutputNeurons);
		DataSet<String> Bfunc = env.fromElements(activefunc);
		DataSet<Integer> BBLOCK = env.fromElements(BlockSize);
		DataSet<DenseMatrix> BAweight=env.fromElements(Aweight);
		DataSet<DenseMatrix> BBweight=env.fromElements(Bweight);

		/*  不使用DenseMatrix库函数 用于性能对比
		List<Tuple2<Integer,Double[]>> Bias=new ArrayList<>();
		for(int i=0;i<Bweight.numRows();i++){
          Double[] array=new Double[NumberofHiddenNeurons];
          for(int j=0;j<Bweight.numColumns();j++){
          	array[j]=Bweight.get(i,j);
		  }
		  Bias.add(new Tuple2<>(i,array));
		}
		DataSet<Tuple2<Integer,Double[]>> BroadcastBias=env.fromCollection(Bias);

		//broadcast inputweight
		List<Tuple2<Integer,Double[]>> InputWeight=new ArrayList<>();
		for(int i=0;i<Aweight.numRows();i++)
		{
			Double[] Array=new Double[NumberofInputNeurons];
			for(int j=0;j<Aweight.numColumns();j++)
			{
				Array[j]=Aweight.get(i,j);
			}
			InputWeight.add(new Tuple2<>(i,Array));
		}
		DataSet<Tuple2<Integer,Double[]>> BroadcastInputWeight=env.fromCollection(InputWeight);
		*/


        /* ************************************************************************************** */
		/*                                 Training                                               */
		/* ************************************************************************************** */

		long start_time_train = System.currentTimeMillis();

		//获得隐藏层输出矩阵  map并行过程
		trainingdata = env.readTextFile(params.get("traindata"))
			.map(new GettrainData())
			.withBroadcastSet(BOSELM_TYPE, "BOSELM_TYPE")
			.withBroadcastSet(BNumberofHiddenNeurons, "BNumberofHiddenNeurons")
			.withBroadcastSet(BNumberofInputNeurons, "BNumberofInputNeurons")
			.withBroadcastSet(BNumberofOutputNeurons, "BNumberofOutputNeurons")
			.withBroadcastSet(Bfunc, "Bfunc")
			.withBroadcastSet(BBLOCK, "BBLOCK")
			.withBroadcastSet(BAweight,"BAweight")
			.withBroadcastSet(BBweight,"BBweight");

		//将每条H和T按照BLOCKID合并成DenseMatrix
		DataSet<Tuple3<Integer, DenseMatrix, DenseMatrix>> getHandT = trainingdata
			.groupBy(0)
			.reduce(new BlockCombine())
			.map(new toDenseMatrix())
			.withBroadcastSet(BNumberofHiddenNeurons, "BNumberofHiddenNeurons")
			.withBroadcastSet(BNumberofOutputNeurons, "BNumberofOutputNeurons");

		//将计算的H和T缓存到list中
		List<Tuple3<Integer, DenseMatrix, DenseMatrix>> tempHandT = getHandT.collect();
		System.out.println("GetHandT: "+tempHandT.size());


		//循环迭代求出OutputWeight
		int length = tempHandT.size();
		DenseMatrix OutputWeight = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
		DenseMatrix Pinv;

		Tuple3<Integer, DenseMatrix, DenseMatrix> value = tempHandT.get(0);
		DenseMatrix H0 = value.f1;
		DenseMatrix T0 = value.f2;
		DenseMatrix transH0 = new DenseMatrix(H0.numColumns(), H0.numRows());
		DenseMatrix K0 = new DenseMatrix(NumberofHiddenNeurons, NumberofHiddenNeurons);
		H0.transpose(transH0);
		transH0.mult(H0, K0);
		Inverse invers0 = new Inverse(K0);
		Pinv = invers0.getMPInverse();
		DenseMatrix temp=new DenseMatrix(NumberofHiddenNeurons,H0.numRows());
		Pinv.mult(transH0, temp);
		temp.mult(T0, OutputWeight);  //is all right 1

		for (int i = 1; i < length; i++) {
			value = tempHandT.get(i);
			DenseMatrix H = value.f1;
			DenseMatrix T = value.f2;

			DenseMatrix transH = new DenseMatrix(H.numColumns(), H.numRows());
			DenseMatrix K = new DenseMatrix(NumberofHiddenNeurons, NumberofHiddenNeurons);
			H.transpose(transH);

			DenseMatrix I = new DenseMatrix(H.numRows(),H.numRows());

			I.zero();
			for (int j = 0; j < H.numRows(); j++) {
				I.set(j, j, 1.0);
			}

			DenseMatrix tempp1 = new DenseMatrix(H.numRows(), H.numColumns());
			H.mult(Pinv, tempp1);
			DenseMatrix temp1=new DenseMatrix(H.numRows(),H.numRows());
			tempp1.mult(transH, temp1);
			temp1.add(I);

			DenseMatrix tempp2;
			DenseMatrix tempp3 = new DenseMatrix(H.numColumns(), H.numRows());
			DenseMatrix tempp4 = new DenseMatrix(H.numColumns(), H.numRows());
			DenseMatrix tempp5 = new DenseMatrix(H.numColumns(), H.numColumns());
			Inverse invers = new Inverse(temp1);
			tempp2 = invers.getMPInverse();
			Pinv.mult(transH, tempp3);
			tempp3.mult(tempp2, tempp4);
			tempp4.mult(H, tempp5);
			tempp5.mult(Pinv, tempp5);
			for (int k = 0; k < Pinv.numRows(); k++)
				for (int j = 0; j < Pinv.numColumns(); j++) {
					double num = Pinv.get(k, j) - tempp5.get(k, j);
					Pinv.set(k, j, num);
				}

			//update outputweight
			DenseMatrix tempp6 = new DenseMatrix(H.numRows(), NumberofOutputNeurons);
			DenseMatrix tempp7 = new DenseMatrix(NumberofHiddenNeurons, NumberofOutputNeurons);
			H.mult(OutputWeight, tempp6);
			for (int k = 0; k < H.numRows(); k++) {
				for (int j = 0; j < NumberofOutputNeurons; j++) {
					double num = T.get(k, j) - tempp6.get(k, j);
					tempp6.set(k, j, num);
				}
			}

			DenseMatrix tempp8=new DenseMatrix(Pinv.numRows(),H.numRows());
			Pinv.mult(transH,tempp8);
			tempp8.mult(tempp6, tempp7);
			OutputWeight.add(tempp7);

		}

		long end_time_train = System.currentTimeMillis();
		TrainingTime = (end_time_train - start_time_train) * 1.0f / 1000;

		//将OutputWeight设置为广播变量
		DataSet<DenseMatrix> OutputWeightToBroacast = env.fromElements(OutputWeight);

		//calculate Accurancy
		if (OSElm_Type == 0) {
			DataSet<Tuple2<Double, Double>> getMseandNum = trainingdata.map(new CalculateMseAndNum())
				.withBroadcastSet(OutputWeightToBroacast, "OutputWeightToBroacast")
				.withBroadcastSet(BNumberofHiddenNeurons, "BNumberofHiddenNeurons")
				.reduce(new SumMseAndNum());

			List<Tuple2<Double, Double>> result;
			result = getMseandNum.collect();

			Tuple2<Double, Double> MseAndNum = result.get(0);
			double Mse = MseAndNum.f0;
			double Number = MseAndNum.f1;
			TrainingAccuracy = Math.sqrt(Mse / Number);
		} else {
			DataSet<Tuple2<Double, Double>> getMissdataAndNum = trainingdata.map(new CalculateMissdataAndNum())
				.withBroadcastSet(OutputWeightToBroacast, "OutputWeightToBroacast")
				.withBroadcastSet(BNumberofHiddenNeurons, "BNumberofHiddenNeurons")
				.reduce(new SumMissdataAndNum());

			List<Tuple2<Double, Double>> result;
			result = getMissdataAndNum.collect();

			Tuple2<Double, Double> MissdataAndNum = result.get(0);
			double MissClassificationRate_Train = MissdataAndNum.f0;
			double numTrainData = MissdataAndNum.f1;
			TrainingAccuracy = 1 - MissClassificationRate_Train / numTrainData;
			System.out.println("Miss: "+MissClassificationRate_Train);
			System.out.println("number: "+numTrainData);
		}


		System.out.println("TrainingAccuracy: "+TrainingAccuracy);
		System.out.println("TrainingTime: "+TrainingTime);



		 /* ************************************************************************************** */
		/*                                 Testing                                                */
		/* ************************************************************************************** */

		start_time_train = System.currentTimeMillis();

		DataSet<Tuple2<Double,Double>> MsgAndDatanum=env.readTextFile(params.get("testdata"))
			.map(new CalculateMsgandDatanum())
			.withBroadcastSet(BOSELM_TYPE, "BOSELM_TYPE")
			.withBroadcastSet(BNumberofHiddenNeurons, "BNumberofHiddenNeurons")
			.withBroadcastSet(BNumberofInputNeurons, "BNumberofInputNeurons")
			.withBroadcastSet(BNumberofOutputNeurons, "BNumberofOutputNeurons")
			.withBroadcastSet(Bfunc, "Bfunc")
			.withBroadcastSet(BBLOCK, "BBLOCK")
			.withBroadcastSet(BAweight, "BAweight")
			.withBroadcastSet(BBweight, "BBweight")
			.withBroadcastSet(OutputWeightToBroacast,"OutputWeightToBroacast")
			.reduce(new GetMsgAndDatanum());

		end_time_train = System.currentTimeMillis();
		TestingTime = (end_time_train - start_time_train) * 1.0f / 1000;

		List<Tuple2<Double, Double>> TestResult;
		TestResult = MsgAndDatanum.collect();

		//求accuracy
		if(OSElm_Type==0)
		{
			double TestMsg=TestResult.get(0).f0;
			double TestdataNumber=TestResult.get(0).f1;
			TestingAccuracy=Math.sqrt(TestMsg/TestdataNumber);
		}
		else{
			double MissClassificationRate_Test=TestResult.get(0).f0;
			double numTestData=TestResult.get(0).f1;
			TestingAccuracy = 1 - MissClassificationRate_Test / numTestData;
			System.out.println("MissClassificationRate_Test: "+MissClassificationRate_Test);
			System.out.println("numTestData: "+numTestData);
		}
		//env.execute();
		//print()方法自动会调用execute()方法，造成错误，所以注释掉env.execute()即可上传flink后台运行

		System.out.println("TrainingAccuracy: "+TrainingAccuracy);
		System.out.println("TrainingTime: "+TrainingTime);
		System.out.println("TestingAccurancy: "+TestingAccuracy);
		System.out.println("TestingTime: "+TestingTime);
	}

	/* ******************************* */
	/*    USER FUNCTION                */
	/* ******************************* */

	public static class GettrainData extends RichMapFunction<String, Tuple4<Integer, Integer, Double[], Double[]>> {

		private Collection<Integer> Inputnodenumber;
		private Collection<Integer> Hiddennodenumber;
		private Collection<Integer> Outputnodenumber;
		private Collection<Integer> oselm_type;
		private Collection<Integer> blocksize;
		private Collection<String> activefunc;
		private Collection<DenseMatrix> Aweight;
		private Collection<DenseMatrix> Bweight;

		@Override
		public void open(Configuration parameters) throws Exception {
			this.Inputnodenumber = getRuntimeContext().getBroadcastVariable("BNumberofInputNeurons");
			this.Hiddennodenumber = getRuntimeContext().getBroadcastVariable("BNumberofHiddenNeurons");
			this.Outputnodenumber = getRuntimeContext().getBroadcastVariable("BNumberofOutputNeurons");
			this.oselm_type = getRuntimeContext().getBroadcastVariable("BOSELM_TYPE");
			this.activefunc = getRuntimeContext().getBroadcastVariable("Bfunc");
			this.blocksize = getRuntimeContext().getBroadcastVariable("BBLOCK");
			this.Aweight=getRuntimeContext().getBroadcastVariable("BAweight");
			this.Bweight=getRuntimeContext().getBroadcastVariable("BBweight");
		}

		//获得广播变量
		int OSElmType;
		int NumberOfInputNeurons;
		int NumberOfHiddenNeurons;
		int NumberOfOutputNeurons;
		int BLOCKSIZE;
		String activefunciton;
		DenseMatrix A;
		DenseMatrix B;

		@Override
		public Tuple4<Integer, Integer, Double[], Double[]> map(String value) {
			// normalize and split the line 一个或多个空格
			String[] tokens = value.split("\\s+");

			for (int it : oselm_type) {
				OSElmType = it;
			}

			for (int it : Inputnodenumber) {
				NumberOfInputNeurons = it;
			}

			for (int it : Hiddennodenumber) {
				NumberOfHiddenNeurons = it;
			}

			for (String it : activefunc) {
				activefunciton = it;
			}

			for (int it : blocksize) {
				BLOCKSIZE = it;
			}

			for (int it : Outputnodenumber) {
				NumberOfOutputNeurons = it;
			}

			A=new DenseMatrix(NumberOfHiddenNeurons,NumberOfInputNeurons);
			for (DenseMatrix it : Aweight) {
				A = it;
			}

			B=new DenseMatrix(1,NumberOfHiddenNeurons);
			for (DenseMatrix it : Bweight) {
				B = it;
			}

			int i = 0;
			int BLOCKID = Integer.parseInt(tokens[0]) / BLOCKSIZE;
			int OFFSET = Integer.parseInt(tokens[0]) % BLOCKSIZE;
			Double[] array = new Double[NumberOfInputNeurons];
			Double[] H = new Double[NumberOfHiddenNeurons];

			//获得H
			for (i = 2; i < NumberOfInputNeurons+2;) {
					array[i - 2] = Double.parseDouble(tokens[i]);
					i++;
			}

			DenseMatrix arraytoMatrix = new DenseMatrix(NumberOfInputNeurons, 1);
			DenseMatrix CalculateH = new DenseMatrix(NumberOfHiddenNeurons, 1);

			for (i = 0; i < NumberOfInputNeurons; i++) {
				arraytoMatrix.set(i, 0, array[i]);
			}

			//calculate H. H is the same in both regression and classify
			A.mult(arraytoMatrix, CalculateH);
			CalculateH.add(B);
			for (i = 0; i < NumberOfHiddenNeurons; i++) {
				double num=CalculateH.get(i, 0);
				double result=Math.sin(num);
				H[i]=result;
				//H[i] = function();
			}

			//calculate T ,according to oselm-type
			Double[] arrayT;
			if (OSElmType == 0) {
				arrayT = new Double[1];
				arrayT[0] = Double.parseDouble(tokens[1]);
			} else {
				int[] Label = new int[NumberOfOutputNeurons];
				arrayT = new Double[NumberOfOutputNeurons];
				for (i = 0; i < NumberOfOutputNeurons; i++)
					Label[i] = i;

				double flagtemp = Double.parseDouble(tokens[1]);
				int flag = 1;
				for (i = 0; i < NumberOfOutputNeurons; i++) {
					arrayT[i] = -1.0;}
				for (i = 0; i < NumberOfOutputNeurons; i++) {
					if (flagtemp == Label[i]) {
						arrayT[i] = 1.0;
						break;
					}
				}
			}

			return new Tuple4<Integer, Integer, Double[], Double[]>(BLOCKID, OFFSET, H, arrayT);
		}

		public double function(double value) {
			String func = activefunciton.toLowerCase();
			double result = 0.0;

			switch (func) {
				case "sin":
					result = Math.sin(value);
					break;
				case "sig":
					result = 1.0f / (1 + Math.exp(-value));
					break;
			}
			return result;
		}
	}

	public static final class BlockCombine implements
		ReduceFunction<Tuple4<Integer, Integer, Double[], Double[]>> {

		@Override
		public Tuple4<Integer, Integer, Double[], Double[]> reduce(Tuple4<Integer, Integer, Double[], Double[]> value1, Tuple4<Integer, Integer, Double[], Double[]> value2) {
			int length1, length2, length3, length4, i;
			Double[] combinearray1;
			Double[] combinearray2;

			length1 = value1.f2.length;
			length2 = value2.f2.length;
			length3 = value1.f3.length;
			length4 = value2.f3.length;
			combinearray1 = new Double[length1 + length2];
			combinearray2 = new Double[length3 + length4];

			//获得H
			for (i = 0; i < length1; i++) {
				combinearray1[i] = value1.f2[i];
			}
			for (i = 0; i < length2; i++) {
				combinearray1[i + length1] = value2.f2[i];
			}

			//获得T
			for (i = 0; i < length3; i++) {
				combinearray2[i] = value1.f3[i];
			}
			for (i = 0; i < length4; i++) {
				combinearray2[i + length3] = value2.f3[i];
			}

			return new Tuple4<>(value1.f0, value1.f1, combinearray1, combinearray2);
		}
	}

	public static class toDenseMatrix extends RichMapFunction<Tuple4<Integer, Integer, Double[], Double[]>,
		Tuple3<Integer, DenseMatrix, DenseMatrix>> {

		private Collection<Integer> NumberOfHiddenNeurons;
		private Collection<Integer> NumberOfOutputNeurons;

		@Override
		public void open(Configuration parameters) throws Exception {
			this.NumberOfHiddenNeurons = getRuntimeContext().getBroadcastVariable("BNumberofHiddenNeurons");
			this.NumberOfOutputNeurons = getRuntimeContext().getBroadcastVariable("BNumberofOutputNeurons");
		}

		int NumberOfHiddenNode;
		int NumberOfOutputNode;

		@Override
		public Tuple3<Integer, DenseMatrix, DenseMatrix> map(Tuple4<Integer, Integer, Double[], Double[]> value) {
			int length1, length2, i, j, k;
			int rownumber1, rownumber2;

			for (int it : NumberOfHiddenNeurons) {
				NumberOfHiddenNode = it;
			}

			for (int it : NumberOfOutputNeurons) {
				NumberOfOutputNode = it;
			}

			length1 = value.f2.length;
			length2 = value.f3.length;
			rownumber1 = length1 / NumberOfHiddenNode;
			rownumber2 = length2 / NumberOfOutputNode;

			DenseMatrix tempH = new DenseMatrix(rownumber1, NumberOfHiddenNode);
			DenseMatrix tempT = new DenseMatrix(rownumber2, NumberOfOutputNode);

			for (i = 0; i < rownumber1; i++) {
				for (j = 0; j < NumberOfHiddenNode; j++) {
					tempH.set(i, j, value.f2[i * NumberOfHiddenNode + j]);
				}

				for (k = 0; k < NumberOfOutputNode; k++) {
					tempT.set(i, k, value.f3[i * NumberOfOutputNode + k]);
				}
			}
			return new Tuple3<>(value.f0, tempH, tempT);
		}
	}

	public static class CalculateMseAndNum extends RichMapFunction<Tuple4<Integer, Integer, Double[], Double[]>, Tuple2<Double, Double>> {

		private Collection<DenseMatrix> GetBroadcastOutputWeight;
		private Collection<Integer> GetBroadcastHiddenNumb;

		@Override
		public void open(Configuration parameters) throws Exception {
			this.GetBroadcastOutputWeight = getRuntimeContext().getBroadcastVariable("OutputWeightToBroacast");
			this.GetBroadcastHiddenNumb = getRuntimeContext().getBroadcastVariable("BNumberofHiddenNeurons");
		}

		@Override
		public Tuple2<Double, Double> map(Tuple4<Integer, Integer, Double[], Double[]> value) {

			int i;
			double Mse;
			int NumberOfHiddenNeurons = 1;
			for (int it : GetBroadcastHiddenNumb) {
				NumberOfHiddenNeurons = it;
			}

			DenseMatrix OutputWeight = new DenseMatrix(NumberOfHiddenNeurons, 1);
			for (DenseMatrix it : GetBroadcastOutputWeight) {
				OutputWeight = it;
			}

			DenseMatrix Htemp = new DenseMatrix(1, value.f2.length);
			for (i = 0; i < value.f2.length; i++)
				Htemp.set(0, i, value.f2[i]);

			DenseMatrix HMultiOutputWeight = new DenseMatrix(1, 1);
			Htemp.mult(OutputWeight, HMultiOutputWeight);

			Mse = HMultiOutputWeight.get(0, 0) - value.f3[0];
			Mse = Mse * Mse;

			return new Tuple2<>(Mse, 1.0);
		}
	}

	public static final class SumMseAndNum implements
		ReduceFunction<Tuple2<Double, Double>> {

		@Override
		public Tuple2<Double, Double> reduce(Tuple2<Double, Double> value1, Tuple2<Double, Double> value2) {

			return new Tuple2<Double, Double>(value1.f0 + value2.f0, value1.f1 + value2.f1);
		}
	}

	public static class CalculateMissdataAndNum extends RichMapFunction<Tuple4<Integer, Integer, Double[], Double[]>, Tuple2<Double, Double>> {

		private Collection<DenseMatrix> GetBroadcastOutputWeight;
		private Collection<Integer> GetBroadcastHiddenNumb;

		@Override
		public void open(Configuration parameters) throws Exception {
			this.GetBroadcastOutputWeight = getRuntimeContext().getBroadcastVariable("OutputWeightToBroacast");
			this.GetBroadcastHiddenNumb = getRuntimeContext().getBroadcastVariable("BNumberofHiddenNeurons");
		}

		@Override
		public Tuple2<Double, Double> map(Tuple4<Integer, Integer, Double[], Double[]> value) {

			int i;
			double Missdata;
			int NumberOfHiddenNeurons = 1;
			for (int it : GetBroadcastHiddenNumb) {
				NumberOfHiddenNeurons = it;
			}

			DenseMatrix OutputWeight = new DenseMatrix(NumberOfHiddenNeurons, 1);
			for (DenseMatrix it : GetBroadcastOutputWeight) {
				OutputWeight = it;
			}

			DenseMatrix Htemp = new DenseMatrix(1, value.f2.length);
			for (i = 0; i < value.f2.length; i++)
				Htemp.set(0, i, value.f2[i]);

			DenseMatrix HMultiOutputWeight = new DenseMatrix(1, value.f3.length);
			Htemp.mult(OutputWeight, HMultiOutputWeight);

			double maxtag1 = HMultiOutputWeight.get(0, 0);
			double maxtag2 = value.f3[0];
			int tag1 = 0;
			int tag2 = 0;

			for (i = 1; i < value.f3.length; i++) {

				if (HMultiOutputWeight.get(0, i) > maxtag1) {
					maxtag1 = HMultiOutputWeight.get(0, i);
					tag1 = i;
				}

				if (value.f3[i] > maxtag2) {
					maxtag2 = value.f3[i];
					tag2 = i;
				}
			}

			if (tag1 == tag2)
				Missdata = 0;
			else
				Missdata = 1;

			return new Tuple2<>(Missdata, 1.0);
		}
	}

	public static final class SumMissdataAndNum implements
		ReduceFunction<Tuple2<Double, Double>> {
		@Override
		public Tuple2<Double, Double> reduce(Tuple2<Double, Double> value1, Tuple2<Double, Double> value2) {

			return new Tuple2<Double, Double>(value1.f0 + value2.f0, value1.f1 + value2.f1);
		}
	}

	public static class  CalculateMsgandDatanum extends RichMapFunction<String,Tuple2<Double,Double>> {

		private Collection<Integer> Inputnodenumber;
		private Collection<Integer> Hiddennodenumber;
		private Collection<Integer> Outputnodenumber;
		private Collection<Integer> oselm_type;
		private Collection<String> activefunc;
		private Collection<DenseMatrix> Aweight;
		private Collection<DenseMatrix> Bweight;
		private Collection<DenseMatrix> BroadcastOutputweight;

		@Override
		public void open(Configuration parameters) throws Exception {
			this.Inputnodenumber = getRuntimeContext().getBroadcastVariable("BNumberofInputNeurons");
			this.Hiddennodenumber = getRuntimeContext().getBroadcastVariable("BNumberofHiddenNeurons");
			this.Outputnodenumber = getRuntimeContext().getBroadcastVariable("BNumberofOutputNeurons");
			this.oselm_type = getRuntimeContext().getBroadcastVariable("BOSELM_TYPE");
			this.activefunc = getRuntimeContext().getBroadcastVariable("Bfunc");
			this.Aweight = getRuntimeContext().getBroadcastVariable("BAweight");
			this.Bweight = getRuntimeContext().getBroadcastVariable("BBweight");
			this.BroadcastOutputweight = getRuntimeContext().getBroadcastVariable("OutputWeightToBroacast");
		}

		//获得广播变量
		int OSElm;
		int NumberOfInputNeurons;
		int NumberOfHiddenNeurons;
		int NumberOfOutputNeurons;
		String funciton;
		DenseMatrix Inputweight;
		DenseMatrix Outputweight;
		DenseMatrix Biaso;

		@Override
		public Tuple2<Double, Double> map(String value) {

			//获得broadcast的值
			for (int it : oselm_type) {
				OSElm = it;
			}

			for (String it : activefunc) {
				funciton = it;
			}

			for (int it : Inputnodenumber) {
				NumberOfInputNeurons = it;
			}

			for (int it : Hiddennodenumber) {
				NumberOfHiddenNeurons = it;
			}

			for (int it : Outputnodenumber) {
				NumberOfOutputNeurons = it;
			}

			for (DenseMatrix it : Aweight) {
				Inputweight = it;
			}

			for (DenseMatrix it : Bweight) {
				Biaso = it;
			}

			for (DenseMatrix it : BroadcastOutputweight) {
				Outputweight = it;
			}

			//获得T和P
			String[] tokens = value.split("\\s+");
			int i;
			DenseMatrix P = new DenseMatrix(NumberOfInputNeurons, 1);
			DenseMatrix H = new DenseMatrix(NumberOfHiddenNeurons, 1);
			DenseMatrix transH = new DenseMatrix(1, NumberOfHiddenNeurons);
			DenseMatrix Ht = new DenseMatrix(1, NumberOfOutputNeurons);
			for (i = 0; i < NumberOfInputNeurons; i++) {
				double a = Double.parseDouble(tokens[i + 2]);
				P.set(i, 0, a);
			}

			//获得Ht
			Inputweight.mult(P, H);
			H.add(Biaso);
			H.transpose(transH);

			for(i=0;i<transH.numColumns();i++){
				double a=transH.get(0,i);

				switch (funciton.toLowerCase()){
					case "sin": a = Math.sin(a);break;
					case "sig": a = 1.0f / (1 + Math.exp(-a));break;
				}

				transH.set(0,i,a);
			}



			transH.mult(Outputweight, Ht);

			//获得T
			Double[] T;
			if (OSElm == 0) {
				T = new Double[1];
				T[0] = Double.parseDouble(tokens[1]);
			} else {
				int[] Label = new int[NumberOfOutputNeurons];
				T = new Double[NumberOfOutputNeurons];

				for (i = 0; i < NumberOfOutputNeurons; i++)
					Label[i] = i;

				double flag = Double.parseDouble(tokens[1]);

				for (i = 0; i < NumberOfOutputNeurons; i++) {
					T[i] = -1.0;
				}

				for (i = 0; i < NumberOfOutputNeurons; i++) {
					if (flag == Label[i]) {
						T[i] = 1.0;
						break;
					}
				}
			}

			double data = 0.0;
			if (OSElm == 0) {
				double Msg = Ht.get(0, 0) - T[0];
				data = Msg * Msg;
			} else {
				double maxtag1 = Ht.get(0, 0);
				double maxtag2 = T[0];
				int tag1 = 0;
				int tag2 = 0;

				for (i = 1; i < T.length; i++) {
					if (Ht.get(0, i) > maxtag1) {
						maxtag1 = Ht.get(0, i);
						tag1 = i;
					}

					if (T[i] > maxtag2) {
						maxtag2 = T[i];
						tag2 = i;
					}
				}

				if (tag1 == tag2)
					data = 0;
				else
					data = 1;
			}

			return new Tuple2<>(data, 1.0);
		}
	}

	public static final class GetMsgAndDatanum implements
		ReduceFunction<Tuple2<Double, Double>> {
		@Override
		public Tuple2<Double, Double> reduce(Tuple2<Double, Double> value1, Tuple2<Double, Double> value2) {
			return new Tuple2<>(value1.f0 + value2.f0, value1.f1 + value2.f1);
		}
	}
			//end of the class oselm
}







