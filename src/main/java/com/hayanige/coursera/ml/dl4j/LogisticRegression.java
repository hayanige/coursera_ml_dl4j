/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.hayanige.coursera.ml.dl4j;

import java.io.File;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LogisticRegression {

  public static void main(String args[]) throws Exception {
    // load data
    File file = new File("src/main/resources/ex2data1.txt");
    RecordReader recordReader = new CSVRecordReader(0, ",");
    recordReader.initialize(new FileSplit(file));
    DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,
        1000);
    DataSet ds = iterator.next();
    INDArray dsArray = ds.getFeatures();
    while (iterator.hasNext()) {
      ds = iterator.next();
      dsArray = Nd4j.vstack(dsArray, ds.getFeatures());
    }

    INDArray y = dsArray.getColumn(2);
    int m = y.length();
    INDArray ones = Nd4j.ones(m, 1);
    INDArray X = Nd4j.hstack(ones, dsArray.getColumn(0), dsArray.getColumn(1));
    INDArray initialTheta = Nd4j.zeros(X.columns(), 1);

    // Compute and display initial cost and gradient
    System.out.println("Cost at initial theta (zeros): "
        + computeCost(initialTheta, X, y));
    System.out.println("Expected cost (approx): 0.693");
    System.out.println("Gradient at initial theta (zeros): "
        + computeGrad(initialTheta, X, y));
    System.out.println("Expected gradients (approx): "
        + "\n -0.1000\n -12.0092\n -11.2628");

    // Compute and display cost and gradient with non-zero theta
    INDArray testTheta = Nd4j
        .create(new double[]{-24, 0.2, 0.2}, new int[]{3, 1});
    System.out.println("Cost at test theta: "
        + computeCost(testTheta, X, y));
    System.out.println("Expected cost (approx): 0.218");
    System.out.println("Gradient at test theta: "
        + computeGrad(testTheta, X, y));
    System.out.println("Expected gradients (approx): "
        + "\n 0.043\n 2.566\n 2.647");
  }

  static double computeCost(INDArray theta, INDArray X, INDArray y) {
    int m = y.length();
    double child = y.mul(-1)
        .mul(Transforms.log(Transforms.sigmoid(X.mmul(theta))))
        .sub(y.mul(-1).add(1).mul(
            Transforms.log(Transforms.sigmoid(X.mmul(theta)).mul(-1).add(1))))
        .sumNumber().doubleValue();
    return child / m;
  }

  static INDArray computeGrad(INDArray theta, INDArray X, INDArray y) {
    int m = y.length();
    INDArray grad = Nd4j.zeros(theta.length(), 1);
    double deriv;
    for (int j = 0; j < theta.length(); j++) {
      deriv = Transforms.sigmoid(X.mmul(theta)).sub(y).mul(X.getColumn(j))
          .sumNumber().doubleValue() / m;
      grad.putScalar(j, 0, deriv);
    }
    return grad;
  }
}
