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

public class LinearRegression {

  public static void main(String args[]) throws Exception {

    // load data
    File file = new File("src/main/resources/ex1data1.txt");
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

    INDArray x = dsArray.getColumn(0);
    INDArray y = dsArray.getColumn(1);
    int m = x.length();
    INDArray ones = Nd4j.ones(m, 1);
    INDArray X = Nd4j.hstack(ones, x);
    INDArray theta = Nd4j.create(new double[]{0, 0}, new int[]{2, 1});

    double J = computeCost(X, y, theta);
    System.out.println("With theta = [0, 0]");
    System.out.println("Cost computed = " + J);
    System.out.println("Expected cost value (approx) 32.07");
    System.out.println();
    J = computeCost(X, y, Nd4j.create(new double[]{-1, 2}, new int[]{2, 1}));
    System.out.println("With theta = [-1, 2]");
    System.out.println("Cost computed = " + J);
    System.out.println("Expected cost value (approx) 54.24");
    System.out.println();

    System.out.println("Running Gradient Descent ...");
    int iterations = 1500;
    float alpha = 0.01f;
    theta = gradientDescent(X, y, theta, alpha, iterations);
    System.out.println("Theta found by gradient descent: " + theta);
    System.out.println("Expected theta values (approx): [-3.6303, 1.1664]");
  }

  static double computeCost(INDArray X, INDArray y, INDArray theta) {
    double child = Transforms.pow(X.mmul(theta).sub(y), 2).sumNumber()
        .doubleValue();
    return child / (2.0 * X.getColumn(0).length());
  }

  static INDArray gradientDescent(INDArray X, INDArray y, INDArray theta_src,
      float alpha, int num_iters) {
    INDArray theta = theta_src.dup();
    int m = X.getColumn(0).length();
    double temp0, temp1;
    for (int i = 0; i < num_iters; i++) {
      temp0 = theta.getDouble(0, 0)
          - alpha * X.mmul(theta).sub(y).sumNumber().doubleValue() / m;
      temp1 = theta.getDouble(1, 0)
          - alpha * X.mmul(theta).sub(y).mul(X.getColumn(1)).sumNumber()
          .doubleValue() / m;
      theta.putScalar(0, 0, temp0);
      theta.putScalar(1, 0, temp1);
    }
    return theta;
  }
}
